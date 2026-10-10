"""VAE-only 探针：只跑视频 VAE 的 encode / decode（不进 DiT、不采样）。

给 POST /api/vae_probe 用 —— 目的是把"编解码段"从 220s 级采样里剥出来单测：

- 编码探针（给了 image_b64）：参考帧本地解码 + 缩放 → `vvae.encode` → latent 探针；
- round-trip（同上，write_png 控制）：把 latent 解回像素，与缩放后源图比
  MAE / PSNR / max|diff|，并把源图与重建图各写一张 PNG —— Light VAE 与官方 VAE
  的画质 A/B 就靠这组数字 + 目视；
- 解码探针（decode=true，默认）：按 width/height/length 造生产同几何的 latent
  （noise=seed 随机，贴近 sample 输出的量级；zeros=_empty_av_latent 原样）→
  `vvae.decode` → 逐帧探针 + 可选 mp4（rawvideo 管进 ffmpeg，与生产同款）。

推理口径与 pipeline 对齐：参考帧尺寸走 `target_frame_size`、解码前把 DiT 从 comfy
加载表摘出（`_unhook_dit`）、VAE 权重上卡、逐帧探针（`_frame_probe`）；只是不建
positive、不进 sampler，所以单次只要几十秒。
"""
import json
import math
import os
import subprocess
import time

from dit_svc.pipeline import _decode_ref_frame, _frame_probe, _unhook_dit


def _vae_info(runtime):
    """报告当前加载的是哪个视频 VAE（文件/解码器深度/tile 几何）—— A/B 的判据。"""
    fs = getattr(runtime.vvae, "first_stage_model", None)
    dec = getattr(fs, "decoder", None)
    blocks = getattr(dec, "transformer_blocks", None)
    return {
        "file": os.path.basename(getattr(runtime.args, "video_vae", "") or "") or "?",
        "decoder_layers": len(blocks) if blocks is not None else None,
        "tile_size": getattr(fs, "tile_size", None),
        "tile_overlap_min": getattr(fs, "tile_overlap_min", None),
        "vae_ratio": getattr(fs, "vae_ratio", None),
    }


def _vram(torch):
    free, total = torch.cuda.mem_get_info()
    return {"alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2)}


def _tensor_probe(torch, t):
    """整张张量的 min/max/nan/inf（小张量用：latent 只有 4.4 万个元素）。"""
    f = t.float()
    out = {"min": round(float(f.min()), 4), "max": round(float(f.max()), 4),
           "mean": round(float(f.mean()), 4), "std": round(float(f.std()), 4),
           "nan": int(torch.isnan(f).sum()), "inf": int(torch.isinf(f).sum()),
           "dtype": str(t.dtype), "shape": list(t.shape)}
    del f
    return out


def _as_frames(decoded, length=None):
    """comfy VAE.decode 输出 [B,T,H,W,C] → [T,H,W,C]（与 pipeline 同款处理）。"""
    d = decoded
    if d.ndim == 5:
        d = d.reshape(-1, *d.shape[-3:])
    if length:
        d = d[:length]
    return d


def _save_png(frames, path, torch):
    """frames [T,H,W,C] float 0..1 → 只存第 0 帧的 PNG。"""
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        arr = frames[0].float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    Image.fromarray(np.asarray(arr)).save(path)


def _write_mp4(frames, path, fps, ffmpeg, torch):
    """rawvideo 管进 ffmpeg（无音频轨版；生产的 mux 另有音频）。"""
    n, hh, ww, _ = frames.shape
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (ww, hh), "-r", str(fps), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    with torch.no_grad():
        for i in range(n):
            u8 = frames[i].float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            proc.stdin.write(u8.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg 失败 rc=%d: %s" % (rc, err[-800:]))


def _psnr(a, b):
    """0..1 像素域的 PSNR（dB）；完全一致返回 None。"""
    mse = float((a - b).pow(2).mean())
    if mse <= 0:
        return None
    return round(10.0 * math.log10(1.0 / mse), 3)


def run_vae_probe(job, ctx):
    """ctx 同 pipeline（runtime/out_dir）；只碰 runtime.vvae。"""
    runtime = ctx["runtime"]
    torch = runtime.torch
    params = job.params
    out_dir = ctx["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_memory = getattr(runtime.rt, "log_memory", None)

    result = {"vae": _vae_info(runtime),
              "params": {k: v for k, v in params.items() if k != "image_b64"}}
    job.log.append("vae probe: %s" % json.dumps(result["vae"], ensure_ascii=False))

    if getattr(runtime, "mm", None) is not None:
        _unhook_dit(runtime, runtime.mm, torch)
    runtime.vvae.first_stage_model.to(runtime.device)

    # ---------- 1. 编码探针（+ round-trip） ----------
    frame = None
    z = None
    if params.get("image_b64"):
        job.stage = "vae 探针：encode"
        t0 = time.perf_counter()
        frame = _decode_ref_frame(params, params["image_b64"])  # [H,W,3] CPU float 0..1
        with torch.no_grad():
            z = runtime.vvae.encode(frame.unsqueeze(0).cuda().float())
        z = z.to(device=runtime.device, dtype=torch.bfloat16)
        enc = _tensor_probe(torch, z)
        enc["frame_shape"] = list(frame.shape)
        enc["seconds"] = round(time.perf_counter() - t0, 2)
        enc["vram"] = _vram(torch)
        result["encode"] = enc
        job.log.append("encode: frame %s -> latent %s seconds=%.2f"
                       % (tuple(frame.shape), tuple(z.shape), enc["seconds"]))
        if log_memory:
            log_memory("vae_probe encode")

        if params.get("write_png"):
            job.stage = "vae 探针：round-trip"
            t0 = time.perf_counter()
            with torch.no_grad():
                back = _as_frames(runtime.vvae.decode(z))    # [1,H,W,C]
            src = frame.clamp(0, 1)
            rec = back[0].float().cpu().clamp(0, 1)
            if tuple(rec.shape) != tuple(src.shape):
                raise RuntimeError("round-trip 尺寸不一致: 源 %s vs 重建 %s"
                                   % (tuple(src.shape), tuple(rec.shape)))
            rt = {"seconds": round(time.perf_counter() - t0, 2),
                  "shape": list(rec.shape),
                  "mae": round(float((rec - src).abs().mean()), 6),
                  "max_abs_diff": round(float((rec - src).abs().max()), 6),
                  "psnr_db": _psnr(rec, src),
                  "probe": _frame_probe(torch, back)}
            src_png = os.path.join(out_dir, job.id + ".source.png")
            rec_png = os.path.join(out_dir, job.id + ".roundtrip.png")
            _save_png(frame.unsqueeze(0), src_png, torch)
            _save_png(back, rec_png, torch)
            rt["files"] = {"source_png": "/outputs/" + os.path.basename(src_png),
                           "roundtrip_png": "/outputs/" + os.path.basename(rec_png)}
            job.files.update({"source_png": rt["files"]["source_png"],
                              "roundtrip_png": rt["files"]["roundtrip_png"]})
            result["roundtrip"] = rt
            job.log.append("round-trip: mae=%.6f max|diff|=%.6f psnr=%s dB (%.2fs)"
                           % (rt["mae"], rt["max_abs_diff"], rt["psnr_db"], rt["seconds"]))
            del back
        del z
        torch.cuda.empty_cache()

    # ---------- 2. 解码探针（生产同几何 latent，无采样） ----------
    if params.get("decode", True):
        job.stage = "vae 探针：decode"
        t0 = time.perf_counter()
        video, frames_expected = _probe_latent(runtime, torch, params)
        latent_stats = {"shape": list(video.shape), "dtype": str(video.dtype),
                        "kind": params.get("latent", "noise"), "seed": params["seed"],
                        "frames_expected": frames_expected}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            decoded = runtime.vvae.decode(video)
        frames = _as_frames(decoded, params["length"])
        dec = {"seconds": round(time.perf_counter() - t0, 2),
               "latent": latent_stats,
               "frames_shape": list(frames.shape),
               "probe": _frame_probe(torch, frames),
               "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "vram": _vram(torch)}
        result["decode"] = dec
        job.log.append("decode: latent %s -> frames %s probe=%s (%.2fs, peak %.2fGB)"
                       % (tuple(video.shape), tuple(frames.shape), dec["probe"],
                          dec["seconds"], dec["peak_alloc_gb"]))
        if log_memory:
            log_memory("vae_probe decode")
        if dec["probe"]["nan"] or dec["probe"]["inf"]:
            raise RuntimeError("VAE 解码输出含 NaN/Inf: %s" % dec["probe"])
        if params.get("write_mp4", True):
            mp4_path = os.path.join(out_dir, job.id + ".mp4")
            _write_mp4(frames, mp4_path, int(params.get("fps", 24) or 24),
                       runtime.ffmpeg, torch)
            job.mp4 = os.path.basename(mp4_path)
            dec["mp4"] = job.mp4
            dec["mp4_mb"] = round(os.path.getsize(mp4_path) / 1e6, 2)
            job.log.append("mp4: %s (%.2f MB)" % (dec["mp4"], dec["mp4_mb"]))
        del decoded, frames, video
        torch.cuda.empty_cache()

    job.cond_stats = {"vae_probe": result}
    sidecar = {"job_id": job.id, "kind": "vae_probe", "vae_probe": result,
               "output": job.mp4}
    with open(os.path.join(out_dir, job.id + ".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)


def _probe_latent(runtime, torch, params):
    """造一条生产同几何的视频 latent：noise（seed 随机）或 zeros。"""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent
    latent, frames = _empty_av_latent(params["width"], params["height"], params["length"])
    video = latent["samples"].tensors[0].to(device=runtime.device)
    if params.get("latent", "noise") == "noise":
        g = torch.Generator(device="cpu").manual_seed(int(params["seed"]) & 0x7FFFFFFF)
        video = torch.randn(video.shape, generator=g, dtype=video.dtype).to(runtime.device)
    return video, int(frames)
