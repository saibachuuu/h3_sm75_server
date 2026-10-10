"""单任务流水线：cond → refine → 参考帧本地解码+VAE 编码 → 采样 → 解码 → ffmpeg mp4。

移植自 infer_h3_turing.py 的 main() 后半段，三处刻意不同：
  1. **无 warmup**（DiT 部分 warmup 已按需求删除）；
  2. token_refiner 常驻，逐任务 refine（不走 identity fast-path hack）；
  3. 输出直接 pipe 进 ffmpeg 出 mp4 —— 不落 PNG 帧序列、不落 wav（wav 仅作
     ffmpeg 的临时输入，mux 后即删）。

cond 文件是 slim 的（只含文本编码器输出）；参考帧由本机从任务 image_b64
本地解码 + 缩放（目标尺寸与 cond 侧 Qwen 输入共用 target_frame_size）后
送 VAE 编码。

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


def run_job(job, ctx):
    """ctx = {"runtime": Runtime, "cond_url": str, "out_dir": str, "entry_args": argparse.Namespace}"""
    runtime = ctx["runtime"]
    rt = runtime.rt
    torch = runtime.torch
    mm = runtime.mm
    device = runtime.device
    params = job.params
    job.stage = "cond"

    # ---------- 1. 取条件（本机编码器常驻，缓存命中即时） ----------
    from dit_svc import cond_client
    t0 = time.perf_counter()
    cond_bytes, cstats = cond_client.fetch(ctx["cond_url"], {
        "prompt": params["prompt"],
        "image_b64": params["image_b64"],
        "mode": params["mode"],
        "width": params["width"],
        "height": params["height"],
        "ref_image_size": params["ref_image_size"],
    })
    job.cond_stats = cstats
    job.log.append("cond: %s" % json.dumps(cstats, ensure_ascii=False))
    _stage(job, "cond", t0)

    tmp_dir = os.path.join(ctx["out_dir"], ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
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

    # ---------- 4. 参考帧：本地解码 image_b64 + 缩放 + VAE 编码 ----------
    job.stage = "vae_encode"
    t0 = time.perf_counter()
    frame = _decode_ref_frame(params, t0)  # CPU [H,W,3] float 0..1，已缩到目标尺寸
    with torch.no_grad():
        z = runtime.vvae.encode(frame.unsqueeze(0).cuda().float())
    encoded_latent = z.to(device="cuda", dtype=compute_dtype)
    job.log.append("ref frame %s latent %s"
                   % (tuple(frame.shape), tuple(encoded_latent.shape)))
    _stage(job, "vae_encode", t0)
    del frame, z
    torch.cuda.empty_cache()

    # ---------- 5. 组装 positive（区分 ref2va / fl2va） ----------
    job.stage = "build_cond"
    if params["mode"] == "ref2va":
        ref_block = {
            "kind": "image",
            "latent_h": encoded_latent.shape[3],
            "latent_w": encoded_latent.shape[4],
            "latent": encoded_latent,
        }
        values = {"minimax_token_tags": tags, "minimax_refs": [ref_block]}
    else:
        values = {"minimax_token_tags": tags,
                  "minimax_keyframes": [{"resolved_frame_index": 0, "latent": encoded_latent}]}
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
    out1 = rt.unpack(SamplerCustomAdvanced.execute(
        noise, guider, sampler_obj, sigmas, latent))[0]
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
    del out1, latent, positive, refined, tags, encoded_latent
    torch.cuda.empty_cache()

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
    #（与 runner 相同的技巧：防止 VAE 解码时 comfy 自动把 DiT 挪去 CPU 触发
    #  is_dead NoneType 问题；**不 del** unet —— 服务器要跨任务常驻复用）
    for loaded in list(mm.current_loaded_models):
        if loaded.model is runtime.unet or loaded.model is runtime.model_shifted:
            mm.current_loaded_models.remove(loaded)
    gc.collect()
    torch.cuda.empty_cache()
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
    job.log.append("decoded video %s %s" % (tuple(decoded.shape), decoded.dtype))
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
        "created": job.created,
        "params": {k: v for k, v in params.items() if k != "image_b64"},
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


def _decode_ref_frame(params, t0):
    """从任务 image_b64 本地解码参考帧：PIL 解码 → 目标尺寸缩放 → CPU float 张量。

    目标尺寸与 cond 侧 Qwen 输入共用 target_frame_size（两侧一致）；缩放用
    comfy.common_upscale lanczos（与 cond 侧 _resize 相同）。返回 [H,W,3] float 0..1。
    """
    import numpy as np
    import torch
    from PIL import Image

    try:
        image_bytes = base64.b64decode(params["image_b64"], validate=True)
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
