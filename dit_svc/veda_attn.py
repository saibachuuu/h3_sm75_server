"""dev 分支：Veda 稀疏 attention 接入 + 分级 warmup（防崩溃）。

设计约束（用户要求 + sage 事故教训）：
  1. 任何 warmup 负载必须分级可控，默认只跑最小 probe；
  2. 高负载 warmup 必须能跳过（--veda-warmup off），且每级之间显式同步 + 显存水位检查；
  3. Veda 初始化失败必须回退到 int8（安全默认），绝不让服务带病跑；
  4. NVML 全零/断流 = 驱动已病（sage 事故判据），warmup 中检测到直接中止并报错。

Veda 依赖（wjie98 _clone_svdint4）：
  - Python 侧：comfyui_turing_utils.adapters.minimax.veda.{integration, predictor, engine}
  - kernel 侧：_sage_qattn_sm75.pyd 含 veda_* 符号（0.43.0 已有，见 pybind_sm75.cpp）
  - bundle：本地 .safetensors（ComfyUI/models/veda/ 或 --veda-bundle 指定，无自动下载）
"""

# warmup 分级：每级 = (名, q_tokens, 说明)。负载逐级放大，默认只跑 probe。
WARMUP_TIERS = {
    "probe": (256, "最小 probe（与现有 sage/int8 probe 同量级）"),
    "tier1": (2048, "短序列（文本段量级）"),
    "tier2": (8192, "中序列（短视频段量级）"),
    "tier3": (24500, "全尺寸（124 帧 packed ~24.5k，sage 崩溃量级——仅手动指定才跑）"),
}
WARMUP_ORDER = ["probe", "tier1", "tier2", "tier3"]

# Veda 需要的 kernel 符号（integration.configure 的 required 列表）
VEDA_SYMBOLS = ("veda_sparse_online_attn", "veda_gather_pool", "veda_scatter_tiles",
                "veda_pack_routes", "veda_prepare_scores", "veda_finish_selection")


def check_nvml_sane(torch):
    """NVML 健康检查：sage 事故中学到的判据——全零/断流 = 驱动已病。返回 (ok, 说明)。"""
    try:
        free, total = torch.cuda.mem_get_info()
        if free == 0 or total == 0:
            return False, "NVML 返回全零（free=%d total=%d），驱动可能已病，中止" % (free, total)
        return True, "free=%.2fGB total=%.2fGB" % (free / 1e9, total / 1e9)
    except Exception as e:
        return False, "NVML 调用失败（断流？）: %s" % e


def run_tiered_warmup(torch, device, attn_fn, impl_name, level, log=print):
    """分级 warmup：从 probe 逐级跑到 level，每级同步 + 水位检查 + NVML 检查。

    attn_fn(q, k, v, heads) 为待测 attention（含 skip_reshape=True 调用形态）。
    任一级失败/水位异常即抛 RuntimeError，调用方回退。level='off' 直接返回。
    """
    import math
    if level == "off":
        log("[warmup] 跳过（--veda-warmup off）")
        return
    if level not in WARMUP_TIERS:
        raise ValueError("未知 warmup 级别: %s（可选 off/probe/tier1/tier2/tier3）" % level)
    target = WARMUP_ORDER.index(level)
    heads, dim = 56, 128  # H3 真实形态
    for name in WARMUP_ORDER[:target + 1]:
        n, desc = WARMUP_TIERS[name]
        ok, vram = check_nvml_sane(torch)
        if not ok:
            raise RuntimeError("[warmup %s] NVML 异常，中止：%s" % (name, vram))
        log("[warmup %s] n=%d（%s）| %s" % (name, n, desc, vram))
        # 分块构造，避免一次性大分配误伤：每 4k tokens 一块拼起来
        parts = []
        for s in range(0, n, 4096):
            m = min(4096, n - s)
            parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
        q = torch.cat(parts, dim=2)
        del parts
        k = q
        v = q.clone()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = attn_fn(q, k, v, heads)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        amax = float(out.float().abs().max())
        ok, vram2 = check_nvml_sane(torch)
        log("[warmup %s] %.1fms finite=%s amax=%.4g | %s"
            % (name, ms, finite, amax, vram2))
        del q, k, v, out
        torch.cuda.empty_cache()
        if not finite or not math.isfinite(amax):
            raise RuntimeError("[warmup %s] 输出非 finite，中止" % name)
        if not ok:
            raise RuntimeError("[warmup %s] 跑后 NVML 异常，中止：%s" % (name, vram2))
    log("[warmup] 到 %s 全过" % level)


def run_single_warmup(torch, device, attn_fn, tokens, log=print, repeat=1):
    """单发 warmup：指定 token 数跑 repeat 次。返回 dict（每轮 ms/finite/amax + 显存）。

    给 POST /api/warmup 用（队列内串行执行，不重启）。失败抛 RuntimeError。
    上限 tokens 40000（超过 packed 上限太多无意义，且防手滑炸卡），
    repeat 上限 50（防长时间占卡；真任务 4 步约等于 repeat=4×56 层）。
    QKV 只分配一次、循环复用（allocator 热态，贴近真任务步间状态，轮间无停顿）；
    empty_cache 只放最后，NVML 每轮查。
    """
    import math
    if not isinstance(tokens, int) or not (16 <= tokens <= 40000):
        raise ValueError("tokens 须为 16–40000 的整数，当前 %r" % (tokens,))
    if not isinstance(repeat, int) or not (1 <= repeat <= 50):
        raise ValueError("repeat 须为 1–50 的整数，当前 %r" % (repeat,))
    heads, dim = 56, 128  # H3 真实形态
    ok, vram0 = check_nvml_sane(torch)
    if not ok:
        raise RuntimeError("NVML 异常，中止：%s" % vram0)
    log("[warmup once] n=%d x%d | %s" % (tokens, repeat, vram0))
    # 一次分配、循环复用：首轮建图后 allocator 全热，后续轮无分配停顿
    parts = []
    for s in range(0, tokens, 4096):
        m = min(4096, tokens - s)
        parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
    q = torch.cat(parts, dim=2)
    del parts
    v = q.clone()
    out_probe = attn_fn(q, q, v, heads)  # 建图轮（不计时，排除编译抖动）
    del out_probe
    torch.cuda.synchronize()
    rounds = []
    for i in range(repeat):
        ok, vram = check_nvml_sane(torch)
        if not ok:
            raise RuntimeError("[warmup %d/%d] NVML 异常，中止：%s" % (i + 1, repeat, vram))
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = attn_fn(q, q, v, heads)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        amax = float(out.float().abs().max())
        ok2, vram2 = check_nvml_sane(torch)
        del out
        r = {"ms": round(ms, 1), "finite": finite,
             "amax": amax if math.isfinite(amax) else None}
        log("[warmup %d/%d] %s | %s" % (i + 1, repeat, r, vram2))
        if not finite or r["amax"] is None:
            raise RuntimeError("[warmup %d/%d] 输出非 finite: %s" % (i + 1, repeat, r))
        if not ok2:
            raise RuntimeError("[warmup %d/%d] 跑后 NVML 异常：%s" % (i + 1, repeat, vram2))
        rounds.append(r)
    del q, v
    torch.cuda.empty_cache()
    result = {"tokens": tokens, "repeat": repeat, "rounds": rounds,
              "ms_min": min(r["ms"] for r in rounds),
              "ms_max": max(r["ms"] for r in rounds),
              "vram_before": vram0, "vram_after": vram2}
    log("[warmup once] done %s" % result)
    return result

def bench_attention(torch, device, veda_call, dense_call, tokens, repeat=3,
                    profile=False, log=print):
    """同一对 Q/K/V 上对比 Veda 稀疏 vs 稠密 int8 的注意力耗时（可选 profiler 明细）。

    veda_call / dense_call：fn(q, k, v) -> out，q/k/v 为 (1, heads, tokens, dim) fp16。
    每档先跑一轮建图（不计时），再跑 repeat 轮取 min/max；每轮校验输出 finite。
    profile=True 时额外用 torch.profiler 抓 CUDA kernel 明细（按 cuda_time_total 排序），
    这是把"到底谁吃时间"从推测变成数字的唯一手段。
    """
    import time as _time
    heads, dim = 56, 128  # H3 真实形态
    parts = []
    for s in range(0, tokens, 4096):
        m = min(4096, tokens - s)
        parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
    q = torch.cat(parts, dim=2)
    del parts
    k = q.clone()
    v = q.clone()

    def _run(fn):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = fn(q, k, v)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        del out
        return round(ms, 2), finite

    result = {"tokens": tokens, "heads": heads, "dim": dim, "repeat": repeat}
    measured = {}
    for name, fn in (("veda", veda_call), ("int8", dense_call)):
        try:
            _run(fn)                       # 建图轮：allocator/内核都热起来，不计时
        except Exception as e:
            result[name] = {"error": "%s: %s" % (type(e).__name__, e)}
            log("[bench] %s 预热失败: %s" % (name, e))
            continue
        rounds = []
        for _ in range(max(1, repeat)):
            ms, finite = _run(fn)
            if not finite:
                raise RuntimeError("%s 注意力输出含 NaN/Inf（n=%d）" % (name, tokens))
            rounds.append(ms)
            log("[bench %s] n=%d round %d: %.1f ms" % (name, tokens, len(rounds), ms))
        measured[name] = rounds
        result[name] = {"rounds": rounds, "ms_min": min(rounds), "ms_max": max(rounds),
                        "ms_avg": round(sum(rounds) / len(rounds), 2)}
    if "veda" in measured and "int8" in measured:
        result["ratio_veda_over_int8"] = round(
            result["veda"]["ms_min"] / max(result["int8"]["ms_min"], 1e-9), 2)
    if profile:
        for name, fn in (("veda", veda_call), ("int8", dense_call)):
            if name not in measured:
                continue
            t0 = _time.perf_counter()
            try:
                from torch.profiler import ProfilerActivity, profile as _profile
                with _profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn(q, k, v)
                    torch.cuda.synchronize()
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            except Exception as e:
                table = "profiler 失败: %s: %s" % (type(e).__name__, e)
            result[name + "_profile"] = table
            log("[bench %s] profile (%.1fs):\n%s" % (name, _time.perf_counter() - t0, table))
    del q, k, v
    torch.cuda.empty_cache()
    log("[bench] 汇总: veda=%s int8=%s ratio=%s"
        % (result.get("veda", {}).get("ms_min"), result.get("int8", {}).get("ms_min"),
           result.get("ratio_veda_over_int8")))
    return result


def build_veda_override(args, unet, torch, log=print):
    """构建 Veda override。成功返回 (override, bundle_info)；失败抛 RuntimeError（调用方回退 int8）。

    步骤：sys.path → 符号检查 → bundle 加载 → 维度对齐 → configure → 最小 probe。
    注意：configure 需要 model 级对象；server 侧只有 unet（ModelPatcher），
    这里复用 wjie98 的 make_override + VedaConfig，绕过 ComfyUI 节点层。
    """
    import os
    import sys

    bundle_path = getattr(args, "veda_bundle", None)
    if not bundle_path or not os.path.isfile(bundle_path):
        raise RuntimeError("--attn veda 需要 --veda-bundle 指向本地 predictor bundle "
                           "(.safetensors)，未找到: %r" % bundle_path)
    precision = getattr(args, "veda_precision", "w8a8") or "w8a8"
    keep = float(getattr(args, "veda_keep", 0.0) or 0.0)
    if keep != 0.0 and not (0.0 < keep <= 1.0):
        raise RuntimeError("--veda-keep 须为 0（用 bundle 默认）或 (0,1]，当前 %r" % keep)

    # TURING_UTILS_DIR 复用 runner 的探测（与 sage/w8a8 同源）
    import infer_h3_turing as rt
    if rt.TURING_UTILS_DIR not in sys.path:
        sys.path.insert(0, rt.TURING_UTILS_DIR)

    from comfyui_turing_utils.kernel_api import kernel_extension_has_symbol
    missing = [s for s in VEDA_SYMBOLS
               if not kernel_extension_has_symbol(s, "_sage_qattn_sm75")]
    if missing:
        raise RuntimeError("kernel 缺 Veda 符号 %s（需重建 Turing Utils kernel）" % missing)

    from comfyui_turing_utils.adapters.minimax.veda.predictor import load_bundle
    bundle = load_bundle(bundle_path, precision)
    log("[veda] bundle: %d层 x %d头 x d%d keep=%.4f precision=%s"
        % (bundle.num_layers, bundle.num_heads, bundle.head_dim,
           bundle.keep_ratio, precision))

    blocks = unet.model.diffusion_model.blocks
    if len(blocks) != bundle.num_layers or any(
            b.attn.heads != bundle.num_heads or b.attn.head_dim != bundle.head_dim
            for b in blocks):
        raise RuntimeError("Veda bundle 维度（%d层/%d头/d%d）与当前 DiT（%d层/%d头/d%d）不匹配" %
                           (bundle.num_layers, bundle.num_heads, bundle.head_dim,
                            len(blocks), blocks[0].attn.heads, blocks[0].attn.head_dim))

    # 走 integration.configure（与 ComfyUI 节点同路径，含 layout provider 安装）。
    # configure 只认 folder_paths 里的 "veda" 目录：把 bundle 所在目录临时注册，
    # 这样 --veda-bundle 传任意完整路径都能用（portable embed 版无 models/veda）。
    from comfyui_turing_utils.adapters.minimax.veda.integration import (
        configure, register_predictor_folder)
    import folder_paths
    register_predictor_folder()
    folder_paths.add_model_folder_path("veda", os.path.dirname(os.path.abspath(bundle_path)))
    model = configure(unet, predictor_name=os.path.basename(bundle_path),
                      predictor_precision=precision,
                      keep_ratio=keep, reference_keep_ratio=keep,
                      debug=bool(getattr(args, "veda_debug", False)))
    override = model.model_options["transformer_options"]["optimized_attention_override"]
    info = {"bundle": os.path.basename(bundle_path), "precision": precision,
            "keep": keep or bundle.keep_ratio}
    log("[veda] override 就绪: %s" % getattr(
        override, "turing_utils_attention_implementation", "?"))
    return override, info
