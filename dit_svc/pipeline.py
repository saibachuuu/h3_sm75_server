"""单任务流水线：cond → refine → 参考帧本地解码+VAE 编码 → 采样 → 解码 → ffmpeg mp4。

移植自 infer_h3_turing.py 的 main() 后半段，三处刻意不同：
  1. **无 warmup**（DiT 部分 warmup 已按需求删除）；
  2. token_refiner 常驻，逐任务 refine（不走 identity fast-path hack）；
  3. 输出直接 pipe 进 ffmpeg 出 mp4 —— 不落 PNG 帧序列、不落 wav（wav 仅作
     ffmpeg 的临时输入，mux 后即删）。

cond 文件是 slim 的（只含文本编码器输出）；参考帧由本机从任务 image_b64
本地解码 + 缩放（目标尺寸与 cond 侧 Qwen 输入共用 target_frame_size）后
送 VAE 编码。

POST /api/jobs/cond（job.cond_source 非空）直接喂 cond：内联 base64 或
dit 机器上已有的文件，**不回调 cond server**（不需要 ssh -R 隧道）；该入口
参考图可选，一张都不给就只带文本条件（不建 minimax_refs / minimax_keyframes）。

解码与 DiT 常驻的显存关系已被 10/9 基线日志证明安全（解码峰值 16.70/17.10GB，
refiner 常驻 +1.52GB 后 ≈18.6GB < 22.5GB）。
"""
import base64
import binascii
import gc
import io
import json
import os
import subprocess
import time
import traceback

from common.protocol import COND_KEYS_REQUIRED, target_frame_size


def _stage(job, name, t0):
    dt = time.perf_counter() - t0
    job.timings[name] = round(dt, 2)
    job.log.append("[%.1fs] stage done: %s" % (dt, name))
    print("[job %s] stage %s took %.2fs" % (job.id, name, dt), flush=True)


def _append_veda_status(job, runtime):
    """把 Veda 工况写进 job.cond_stats.veda_calls + 任务日志（vea 档才生效）。

    结构见 veda_attn_star7.status_report：
      backend / backend_report（启动自检）/ cumulative（跨 run 累计）/ current。
    非 veda 档或读取失败只记日志，不影响任务。
    """
    if getattr(runtime, "attn", None) != "veda":
        return
    try:
        from dit_svc.veda_attn_star7 import status_report
        report = status_report(getattr(runtime, "veda_patch", None))
    except Exception as e:  # noqa: BLE001
        job.log.append("veda status 读取失败: %s: %s" % (type(e).__name__, e))
        return
    if report is None:
        job.log.append("veda: 未启用（或已回退 int8）")
        return
    job.cond_stats = job.cond_stats or {}
    job.cond_stats["veda_calls"] = report
    cum = report.get("cumulative") or {}
    cur = report.get("current") or {}
    job.log.append(
        "veda: backend=%s | 累计 sparse=%s dense=%s runs=%s "
        "compute_fraction_last=%s kept_last=%s | current sparse=%s%s"
        % (report.get("backend"), cum.get("sparse"), cum.get("dense"),
           cum.get("runs"), cum.get("compute_fraction_last"),
           cum.get("kept_fraction_last"), cur.get("sparse"),
           (" | failed=%s" % cum.get("failed")) if cum.get("failed") else ""))


def run_job(job, ctx):
    """ctx = {"runtime": Runtime, "cond_url": str, "out_dir": str, "entry_args": argparse.Namespace}"""
    runtime = ctx["runtime"]
    rt = runtime.rt
    torch = runtime.torch
    mm = runtime.mm
    device = runtime.device
    params = job.params
    job.stage = "cond"

    # ---------- 1. 取条件 ----------
    # 默认走本机编码器（cond server，缓存命中即时）；/api/jobs/cond 直接喂 cond 时
    # 完全跳过 cond server —— 不依赖 ssh -R 隧道。
    t0 = time.perf_counter()
    if getattr(job, "cond_source", None):
        cond_bytes, cstats = _load_direct_cond(job)
    else:
        from dit_svc import cond_client
        cond_bytes, cstats = cond_client.fetch(ctx["cond_url"], {
            "prompt": params["prompt"],
            "image_b64s": params.get("image_b64s") or [params["image_b64"]],
            "mode": params["mode"],
            "width": params["width"],
            "height": params["height"],
            "ref_image_size": params["ref_image_size"],
        })
    job.cond_stats = cstats
    job.log.append("cond: %s" % json.dumps(cstats, ensure_ascii=False))
    _stage(job, "cond", t0)

    tmp_dir = tmp_dir_of(ctx["out_dir"])
    cond_path = os.path.join(tmp_dir, job.id + ".cond.safetensors")
    with open(cond_path, "wb") as f:
        f.write(cond_bytes)

    try:
        _run_with_cond(job, ctx, cond_path)
    finally:
        try:
            os.remove(cond_path)
        except OSError:
            pass


def tmp_dir_of(out_dir):
    """out_dir 下的临时目录（cond 文件、mux 前的临时 wav）。"""
    d = os.path.join(out_dir, ".tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _unhook_dit(runtime, mm, torch):
    """把 DiT 的 wrapper 从 comfy 加载表摘出（与 runner 同款技巧）：防止 VAE
    编解码时 comfy 自动把 DiT 挪去 CPU 触发 is_dead NoneType 问题。
    **不 del** unet —— 服务器要跨任务常驻复用。"""
    for loaded in list(mm.current_loaded_models):
        if loaded.model is runtime.unet or loaded.model is runtime.model_shifted:
            mm.current_loaded_models.remove(loaded)
    gc.collect()
    torch.cuda.empty_cache()


def _frame_probe(torch, frames):
    """逐帧统计 min/max/nan/inf（frames 为 [T,H,W,C] 或 [T,...]）。

    整块 frames.float() 是 124×960×640×3 fp32 ≈ 0.91GB，再加 isnan/isinf 的 bool
    中间量就能把解码后的余量吃光（10/11 真任务实测 OOM）；逐帧只多一帧的临时量
    （≈7MB），统计口径与整块一致。vae_probe 复用同一口径。
    """
    vmin, vmax, vnan, vinf = float("inf"), float("-inf"), 0, 0
    with torch.no_grad():
        for i in range(frames.shape[0]):
            f = frames[i].float()
            vmin = min(vmin, float(f.min()))
            vmax = max(vmax, float(f.max()))
            vnan += int(torch.isnan(f).sum())
            vinf += int(torch.isinf(f).sum())
            del f
    return {"min": round(vmin, 4), "max": round(vmax, 4), "nan": vnan, "inf": vinf}


def _self_cuda_us(entry):
    """torch profiler 的 per-kernel CUDA 时间（兼容新旧字段名）。"""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(entry, attr, None)
        if v:
            return float(v)
    return 0.0


class _AttnWindowProfiler:
    """只对前 `calls` 次**注意力调用**开 profiler（窗口 = 几次调用，不是整步）。

    为什么不能用整步窗口：CPU activity + `with_stack` + `record_shapes` 会把每一步里
    每个算子连同 python 栈与所有张量形状全记下来。2026-10-11 实测：16GB 主机被吃到
    只剩 2.3GB，且 16 帧的任务（249s）比 124 帧（224s）还慢。缩到单次调用后事件量小
    两个数量级，仍然能回答"注意力里跑了哪些 kernel、各自多久、谁调的、张量多大"。

    另外装一个**轻量 bf16 matmul 探针**（不依赖 profiler）：命中 bf16 矩阵乘就记一次
    调用栈 + 所属模块，用来定位 sm75 上走 MAGMA fp32 仿算的那只 GEMM。
    """

    def __init__(self, runtime, calls=2, limit_bf16=8):
        self.runtime = runtime
        self.calls = max(1, int(calls))
        self.limit_bf16 = limit_bf16
        self.profs = []
        self.bf16_hits = []
        self._restore = []
        self._n = 0

    # ---------- 安装 / 卸载 ----------
    def install(self, log):
        torch = self.runtime.torch
        from torch.profiler import ProfilerActivity, profile as _profile
        state = self

        def wrap(fn, label):
            def profiled(*a, **k):
                if state._n >= state.calls:
                    return fn(*a, **k)
                state._n += 1
                prof = _profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                with_stack=True, record_shapes=True)
                prof.__enter__()
                try:
                    return fn(*a, **k)
                finally:
                    prof.__exit__(None, None, None)
                    state.profs.append(prof)
                    log("attn profiler: 第 %d/%d 次调用已录（%s）" % (state._n, state.calls, label))
            try:                     # 保留 override 自带属性（prepared_attention_executor 等）
                profiled.__dict__.update(getattr(fn, "__dict__", {}))
            except Exception:
                pass
            prepared = getattr(fn, "prepared_attention_executor", None)
            if prepared is not None:
                profiled.prepared_attention_executor = wrap(prepared, label + ".prepared")
            return profiled

        def patch(holder, key, label):
            fn = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
            if fn is None:
                return
            wrapped = wrap(fn, label)
            if isinstance(holder, dict):
                holder[key] = wrapped
                self._restore.append(lambda h=holder, k=key, o=fn: h.__setitem__(k, o))
            else:
                setattr(holder, key, wrapped)
                self._restore.append(lambda h=holder, k=key, o=fn: setattr(h, k, o))
            log("attn profiler: 已包住 %s" % label)

        patch(getattr(self.runtime, "attention", None), "optimized_attention",
              "attention.optimized_attention")
        try:                     # H3 真正的调用点：comfy.ldm.minimax.model.optimized_attention
            import comfy.ldm.minimax.model as mm_model
            patch(mm_model, "optimized_attention", "comfy.ldm.minimax.model.optimized_attention")
        except Exception as e:
            log("attn profiler: 包装 minimax 调用点失败: %s: %s" % (type(e).__name__, e))
        for name in ("unet", "model_shifted"):
            model = getattr(self.runtime, name, None)
            try:
                topts = model.model_options["transformer_options"]
            except Exception:
                continue
            if isinstance(topts, dict):
                patch(topts, "optimized_attention_override", name + ".transformer_options.override")
        self._install_bf16_spy(torch, log)

    def restore(self):
        for undo in reversed(self._restore):
            try:
                undo()
            except Exception:
                pass
        self._restore = []

    # ---------- bf16 matmul 探针 ----------
    def _install_bf16_spy(self, torch, log):
        import sys
        import torch.nn.functional as F
        hits = self.bf16_hits
        limit = self.limit_bf16

        def spy(name, target, holder, attr):
            def wrapped(*a, **k):
                if len(hits) < limit:
                    dtypes = [getattr(x, "dtype", None) for x in a]
                    if any(d is torch.bfloat16 for d in dtypes):
                        module = None
                        chain = []
                        f = sys._getframe(1)
                        for _ in range(14):
                            if f is None:
                                break
                            chain.append("%s:%d %s"
                                         % (f.f_code.co_filename.replace("\\", "/").split("/")[-1],
                                            f.f_lineno, f.f_code.co_name))
                            if module is None:
                                slf = f.f_locals.get("self")
                                if slf is not None and hasattr(slf, "weight") \
                                        and hasattr(slf, "forward"):
                                    module = type(slf).__name__
                            f = f.f_back
                        rec = {"op": name, "module": module,
                               "dtypes": [str(d) for d in dtypes],
                               "shapes": [tuple(x.shape) for x in a if hasattr(x, "shape")][:4],
                               "stack": chain[:8]}
                        hits.append(rec)
                        log("[bf16-spy] %s module=%s dtypes=%s shapes=%s"
                            % (name, module, rec["dtypes"], rec["shapes"]))
                    return target(*a, **k)
                return target(*a, **k)
            setattr(holder, attr, wrapped)
            self._restore.append(lambda h=holder, x=attr, o=target: setattr(h, x, o))

        spy("F.linear", F.linear, F, "linear")
        for attr in ("matmul", "bmm", "baddbmm"):
            target = getattr(torch, attr, None)
            if target is not None:
                spy("torch." + attr, target, torch, attr)

    # ---------- 汇总 ----------
    def report(self):
        agg = {}
        for prof in self.profs:
            for e in prof.key_averages(group_by_stack_n=6):
                cur = agg.setdefault(str(e.key), {"ms": 0.0, "n": 0, "shapes": None, "stack": None})
                cur["ms"] += _self_cuda_us(e) / 1000.0
                cur["n"] += int(e.count)
                if cur["shapes"] is None and getattr(e, "input_shapes", None):
                    cur["shapes"] = list(e.input_shapes)[:2]
                if cur["stack"] is None and getattr(e, "stack", None):
                    cur["stack"] = list(e.stack)[:5]
        lines = ["attn profile（前 %d 次注意力调用；按 self CUDA 排序，共 %d 种 kernel）"
                 % (len(self.profs), len(agg))]
        for key, v in sorted(agg.items(), key=lambda kv: -kv[1]["ms"])[:18]:
            shape_txt = ("; ".join(str(s) for s in (v["shapes"] or [])))[:100] or "-"
            lines.append("  %-42s %9.2f ms x%-5d %s" % (key[:42], v["ms"], v["n"], shape_txt))
            for frame in (v["stack"] or [])[:4]:
                lines.append("        @ %s" % str(frame).replace("\n", " ")[:150])
        lines.append("bf16 matmul 探针：命中 %d 次（上限 %d）" % (len(self.bf16_hits), self.limit_bf16))
        for rec in self.bf16_hits:
            lines.append("  %s module=%s dtypes=%s" % (rec["op"], rec["module"], rec["dtypes"]))
            lines.append("        shapes=%s" % (rec["shapes"],))
            for frame in rec["stack"][:6]:
                lines.append("        @ %s" % frame)
        return lines


def _start_attn_profiler(job, params, runtime):
    """profile_attn=true：录前 N 次注意力调用的内核明细 + bf16 matmul 调用栈。

    窗口刻意只覆盖**几次注意力调用**（不是整个采样步）：整步 + CPU/栈/形状在 16GB
    主机上会吃掉 ~10GB 并把任务拖慢到比 124 帧还慢（实测）。
    """
    if not params.get("profile_attn"):
        return None
    try:
        state = _AttnWindowProfiler(runtime, calls=2)
        state.install(lambda s: job.log.append(s))
        job.log.append("attn profiler: 已装（只录前 2 次注意力调用；CPU+CUDA+栈+形状）+ bf16 matmul 探针")
        return state
    except Exception as e:
        job.log.append("attn profiler 启动失败: %s: %s" % (type(e).__name__, e))
        return None


def _finish_attn_profiler(job, state):
    """收尾：还原被包的入口，把报告写进 job.cond_stats["attn_profile"] 与任务日志。"""
    if state is None:
        return
    try:
        state.restore()
    except Exception as e:
        job.log.append("attn profiler 还原失败: %s: %s" % (type(e).__name__, e))
    try:
        lines = state.report()
        job.cond_stats = job.cond_stats or {}
        job.cond_stats["attn_profile"] = "\n".join(lines)
        job.log.extend(lines)
    except Exception as e:
        job.log.append("attn profile 汇总失败: %s: %s" % (type(e).__name__, e))


def _load_direct_cond(job):
    """直接喂 cond（POST /api/jobs/cond）：内联 base64 或 dit 机器上已有的文件。

    返回 (cond bytes, stats)。内容契约由 _run_with_cond 的 COND_KEYS_REQUIRED 兜底。
    """
    src = job.cond_source
    if src["kind"] == "b64":
        return src["bytes"], {"source": "direct:b64", "bytes": len(src["bytes"])}
    with open(src["path"], "rb") as f:
        data = f.read()
    return data, {"source": "direct:file", "file": os.path.basename(src["path"]),
                  "bytes": len(data)}


def _run_with_cond(job, ctx, cond_path):
    runtime = ctx["runtime"]
    rt = runtime.rt
    torch = runtime.torch
    mm = runtime.mm
    device = runtime.device
    params = job.params
    compute_dtype = torch.bfloat16

    from safetensors.torch import load_file

    # ---------- 2. 读 cond（slim 契约：只含文本编码器输出） ----------
    job.stage = "load_cond"
    t0 = time.perf_counter()
    sd = load_file(cond_path)
    missing = [k for k in COND_KEYS_REQUIRED if k not in sd]
    if missing:
        raise RuntimeError("cond 文件缺少必需字段: %s" % missing)
    cond_tensor = sd["cond"].to(device="cuda", dtype=compute_dtype)
    tags = sd["minimax_token_tags"].cuda().long()
    job.log.append("cond tensor %s tags %s" % (tuple(cond_tensor.shape), tuple(tags.shape)))
    _stage(job, "load_cond", t0)

    # ---------- 3. 文本 refine（refiner 常驻，逐任务跑） ----------
    job.stage = "refine"
    t0 = time.perf_counter()
    with torch.no_grad():
        refined = runtime.diff_model.preprocess_text_embeds(cond_tensor)
    job.log.append("refined %s %s" % (tuple(refined.shape), refined.dtype))
    _stage(job, "refine", t0)
    del cond_tensor
    torch.cuda.empty_cache()

    # ---------- 4. 参考帧（可选）：本地解码 image_b64s + 缩放 + 逐张 VAE 编码 ----------
    # ref2va 1–9 张 → minimax_refs 多块（顺序即 <Picture i>）；fl2va 仅单图 → keyframes。
    # 一张都没给（/api/jobs/cond 允许）→ 不建 ref/keyframe 块，纯文本条件。
    job.stage = "vae_encode"
    t0 = time.perf_counter()
    b64s = params.get("image_b64s") or []
    ref_latents = []  # [(latent, latent_h, latent_w)]
    if not b64s:
        job.log.append("无参考图：纯文本条件（不建 minimax_refs/minimax_keyframes）")
    for i, b64 in enumerate(b64s):
        frame = _decode_ref_frame(params, b64)  # CPU [H,W,3] float 0..1，已缩到目标尺寸
        with torch.no_grad():
            z = runtime.vvae.encode(frame.unsqueeze(0).cuda().float())
        zl = z.to(device="cuda", dtype=compute_dtype)
        with torch.no_grad():
            zf = zl.float()
            enc_probe = {"min": round(float(zf.min()), 4), "max": round(float(zf.max()), 4),
                         "nan": int(torch.isnan(zf).sum()), "inf": int(torch.isinf(zf).sum())}
            del zf
        if enc_probe["nan"] or enc_probe["inf"]:
            raise RuntimeError("参考帧[%d] VAE 编码输出含 NaN/Inf: %s" % (i, enc_probe))
        ref_latents.append((zl, int(zl.shape[3]), int(zl.shape[4])))
        job.log.append("ref[%d] frame %s latent %s probe=%s"
                       % (i, tuple(frame.shape), tuple(zl.shape), enc_probe))
        del frame, z
    _stage(job, "vae_encode", t0)
    torch.cuda.empty_cache()

    # ---------- 5. 组装 positive（ref2va / fl2va / 无参考图三种） ----------
    job.stage = "build_cond"
    values = {"minimax_token_tags": tags}
    if ref_latents:
        if params["mode"] == "ref2va":
            values["minimax_refs"] = [{"kind": "image", "latent_h": lh, "latent_w": lw,
                                       "latent": zl} for zl, lh, lw in ref_latents]
        else:
            values["minimax_keyframes"] = [{"resolved_frame_index": 0,
                                            "latent": ref_latents[0][0]}]
    for k in ("attention_mask", "pooled_output"):
        if k in sd:
            values[k] = sd[k].cuda()
    positive = [[refined, values]]
    del sd

    from comfy_extras.nodes_minimax_h3 import _empty_av_latent
    latent, frame_count = _empty_av_latent(params["width"], params["height"], params["length"])
    job.log.append("empty latent video=%s audio=%s frames=%d"
                   % (tuple(latent["samples"].tensors[0].shape),
                      tuple(latent["samples"].tensors[1].shape), frame_count))

    # ---------- 6. 采样（第一遍 + 可选 refine 第二遍；无 warmup） ----------
    job.stage = "sample"
    t0 = time.perf_counter()
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
    )
    model_shifted = runtime.model_shifted

    if params.get("sigmas"):
        sigmas = torch.tensor([float(s) for s in params["sigmas"].split(",") if s.strip()],
                              dtype=torch.float32)
        job.log.append("manual sigmas: %s" % [round(float(s), 6) for s in sigmas])
    else:
        sigmas = rt.unpack(BasicScheduler.execute(
            model_shifted, params["scheduler"], params["steps"], 1.0))[0]
    sampler_obj = rt.unpack(KSamplerSelect.execute(params["sampler"]))[0]
    noise = rt.unpack(RandomNoise.execute(params["seed"]))[0]
    guider = rt.unpack(BasicGuider.execute(model_shifted, positive))[0]
    job.log.append("sampling: %s/%s steps=%d seed=%d"
                   % (params["sampler"], params["scheduler"], params["steps"], params["seed"]))
    rt.log_memory("job pre-sample")
    profiler = _start_attn_profiler(job, params, runtime)
    try:
        out1 = rt.unpack(SamplerCustomAdvanced.execute(
            noise, guider, sampler_obj, sigmas, latent))[0]
    finally:
        _finish_attn_profiler(job, profiler)
    job.timings["sample_first"] = round(time.perf_counter() - t0, 2)

    if params["refine_steps"] > 0:
        job.stage = "refine_pass"
        t1 = time.perf_counter()
        sigmas2 = rt.unpack(BasicScheduler.execute(
            model_shifted, params["scheduler"], params["refine_steps"],
            params["refine_denoise"]))[0]
        noise2 = rt.unpack(RandomNoise.execute(params["seed"] + 1))[0]
        guider2 = rt.unpack(BasicGuider.execute(model_shifted, positive))[0]
        out2 = rt.unpack(SamplerCustomAdvanced.execute(
            noise2, guider2, sampler_obj, sigmas2, out1))[0]
        job.timings["sample_refine"] = round(time.perf_counter() - t1, 2)
    else:
        out2 = out1
    _stage(job, "sample", t0)
    del out1, latent, positive, refined, tags, ref_latents
    # Veda 工况（每个采样 run 结束时 star7 的 cleanup 已把统计抓进累计，
    # 这里读的是"含本任务"的口径；稀疏真跑次数看 veda.calls.sparse）
    _append_veda_status(job, runtime)

    # ---------- 7. 归因探针：latent 坏了就直接失败，不解码 ----------
    job.stage = "probe"
    with torch.no_grad():
        p = out2["samples"].unbind()[0].float()
        probe = {
            "min": round(float(p.min()), 4), "max": round(float(p.max()), 4),
            "nan": int(torch.isnan(p).sum()), "inf": int(torch.isinf(p).sum()),
        }
        del p
    job.latent_probe = probe
    job.log.append("latent probe: %s" % probe)
    if probe["nan"] or probe["inf"]:
        raise RuntimeError("采样输出含 NaN/Inf，拒绝送入解码: %s" % probe)

    # ---------- 8. comfy 模型簿记：把 DiT 的 wrapper 摘出加载表 ----------
    _unhook_dit(runtime, mm, torch)
    rt.log_memory("job post-sample")

    # ---------- 9. 视频解码（DiT 权重常驻，基线已证安全） ----------
    job.stage = "decode_video"
    t0 = time.perf_counter()
    runtime.vvae.first_stage_model.to(device)
    with torch.no_grad():
        decoded = runtime.vvae.decode(out2["samples"].unbind()[0])
    if decoded.ndim == 5:
        decoded = decoded.reshape(-1, *decoded.shape[-3:])
    if decoded.shape[0] > params["length"]:
        decoded = decoded[:params["length"]]
    # 逐帧探针（口径与 vae_probe 一致；整块 fp32 探针会在 124 帧上 OOM）
    dec_probe = _frame_probe(torch, decoded)
    job.log.append("decoded video %s %s probe=%s" % (tuple(decoded.shape), decoded.dtype, dec_probe))
    if dec_probe["nan"] or dec_probe["inf"]:
        raise RuntimeError("视频 VAE 解码输出含 NaN/Inf: %s" % dec_probe)
    _stage(job, "decode_video", t0)

    # ---------- 10. 音频解码（截到 length/24 秒） ----------
    job.stage = "decode_audio"
    t0 = time.perf_counter()
    from comfy_extras.nodes_audio import vae_decode_audio
    with torch.no_grad():
        audio = vae_decode_audio(runtime.avae, out2)
    sr = audio["sample_rate"]
    target = int(round((params["length"] / 24.0) * sr))
    if audio["waveform"].shape[-1] > target:
        audio["waveform"] = audio["waveform"][..., :target]
    job.log.append("decoded audio %s sr=%d" % (tuple(audio["waveform"].shape), sr))
    _stage(job, "decode_audio", t0)
    del out2
    torch.cuda.empty_cache()

    # ---------- 11. ffmpeg mux：rawvideo 管进、wav 管进，只落 mp4 ----------
    job.stage = "mux"
    t0 = time.perf_counter()
    out_dir = ctx["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, job.id + ".mp4")
    tmp_wav = os.path.join(tmp_dir_of(out_dir), job.id + ".tmp.wav")
    rt.save_wav(tmp_wav, audio["waveform"], sr)
    n_frames, hh, ww, _ = decoded.shape
    fps = int(params.get("fps", 24) or 24)
    cmd = [
        runtime.ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "%dx%d" % (ww, hh), "-r", str(fps), "-i", "pipe:0",
        "-i", tmp_wav,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", mp4_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with torch.no_grad():
            for i in range(n_frames):
                u8 = decoded[i].float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
                proc.stdin.write(u8.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        rc = proc.wait()
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
    if rc != 0:
        raise RuntimeError("ffmpeg 失败 rc=%d: %s" % (rc, err[-1200:]))
    job.mp4 = os.path.basename(mp4_path)
    job.log.append("mp4: %s (%.1f MB)"
                   % (job.mp4, os.path.getsize(mp4_path) / 1e6))
    _stage(job, "mux", t0)

    # ---------- 12. 参数 sidecar（画廊“参数”链接指向它） ----------
    sidecar = {
        "job_id": job.id,
        "params": {k: v for k, v in params.items() if k not in ("image_b64", "image_b64s")},
        "cond_stats": job.cond_stats,
        "latent_probe": job.latent_probe,
        "timings": job.timings,
        "attn": getattr(runtime, "attn", None),
        "output": job.mp4,
    }
    with open(os.path.join(out_dir, job.id + ".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    del decoded, audio
    gc.collect()
    torch.cuda.empty_cache()
    rt.log_memory("job done")


def _decode_ref_frame(params, image_b64):
    """从单张 image_b64 本地解码参考帧：PIL 解码 → 目标尺寸缩放 → CPU float 张量。

    目标尺寸与 cond 侧 Qwen 输入共用 target_frame_size（两侧一致）；缩放用
    comfy.common_upscale lanczos（与 cond 侧 _resize 相同）。返回 [H,W,3] float 0..1。
    """
    import numpy as np
    import torch
    from PIL import Image

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_b64 不是合法 base64") from None
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    frame = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3]
    raw_h, raw_w = int(frame.shape[1]), int(frame.shape[2])

    tw, th = target_frame_size(params["mode"], raw_w, raw_h,
                               params["width"], params["height"],
                               params.get("ref_image_size", "match"))
    import comfy.utils as cu
    samples = frame[..., :3].movedim(-1, 1)
    samples = cu.common_upscale(samples, tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)[0]
