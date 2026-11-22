"""Bounded synthetic-input DiT probe; real sampler path, no decode, no artifacts.

Scale is cond tokens and latent canvas size, not a promised internal token count.

The probe runs the **DiT body only**: the synthetic cond is fed with the model's
internal width already (``hidden_size``), so ``MiniMaxH3Model._embed_and_pack``
takes its ``text_states.shape[-1] == self.hidden_size`` fast path and never calls
``token_refiner`` / ``condition_proj``. That makes token refiner optional: it may
stay absent (purged) and the probe still runs, which is exactly what a
body-only probe should measure.

Inputs are synthetic but the sampling compute is real: the latent comes from the
stock ``_empty_av_latent`` and sampling runs the stock SamplerCustomAdvanced
chain (RandomNoise/BasicGuider/KSamplerSelect/BasicScheduler). No reference
images (no VAE involved), no mp4/latent artifacts.
This module is safe to import without torch/Comfy.
"""
import time

from dit_svc.operations import OperationError
from dit_svc.te_probe import cuda_memory, require_workspace

# 本 checkpoint 的 DiT 内部宽度（MiniMaxH3Model.hidden_size）。probe 直接以该宽度
# 造 cond，避免触发 condition_proj/token_refiner 这条非 body 分支。
DEFAULT_COND_DIM = 5376


def validate_request(payload):
    from dit_svc import api_validation as api
    api._require_object(payload)
    api._reject_unknown(payload, {"scale", "steps", "repeat", "warmup", "seed",
                                  "min_free_mb", "request_id"})
    scale = payload.get("scale")
    if not isinstance(scale, dict):
        raise api.invalid("scale 必须是 JSON 对象", field="scale")
    api._reject_unknown(scale, {"cond_tokens", "width", "height", "length"})
    cleaned = {
        "cond_tokens": api._strict_int(scale, "cond_tokens", 512, lo=16, hi=8000),
        "width": api._strict_int(scale, "width", 640, lo=256, hi=1600),
        "height": api._strict_int(scale, "height", 960, lo=256, hi=1600),
        "length": api._strict_int(scale, "length", 124, lo=16, hi=400),
    }
    if cleaned["width"] % 32 or cleaned["height"] % 32:
        raise api.invalid("画布 width/height 必须是 32 的倍数", field="scale")
    return {"scale": cleaned,
            "steps": api._strict_int(payload, "steps", 1, lo=1, hi=4),
            "repeat": api._strict_int(payload, "repeat", 3, lo=1, hi=20),
            "warmup": api._strict_int(payload, "warmup", 1, lo=0, hi=5),
            "seed": api._strict_int(payload, "seed", 0, lo=0, hi=2 ** 63 - 1),
            "min_free_mb": api._strict_int(payload, "min_free_mb", 256,
                                           lo=0, hi=16384),
            "request_id": api._request_id(payload)}


def cond_dim(runtime):
    """读取 DiT 内部宽度；读不到就报错，不隐式猜一个宽度。"""
    model = getattr(runtime, "diff_model", None)
    width = getattr(model, "hidden_size", None)
    if isinstance(width, int) and width > 0:
        return width
    return DEFAULT_COND_DIM


def synthetic_cond(torch, scale, seed, width=DEFAULT_COND_DIM):
    """Random body-width cond + all-text tags from a local CPU generator.

    ``width`` must be the DiT internal width: the forward then skips
    ``condition_proj`` + ``token_refiner`` entirely (body-only probe).
    Tags are all 1 (pure text tokens — no minimax_refs blocks are built).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cond = torch.randn((1, scale["cond_tokens"], width),
                       generator=generator, dtype=torch.float32)
    tags = torch.ones(scale["cond_tokens"], dtype=torch.long)
    return cond, tags


def _latent_finite(torch, video):
    """Frame-by-frame finite check (whole-tensor float() can OOM at full size)."""
    with torch.no_grad():
        for i in range(video.shape[0]):
            frame = video[i].float()
            if not bool(torch.isfinite(frame).all()):
                return False
    return True


def _dit_resident_snapshot(runtime):
    models = runtime.weights_snapshot()["models"]
    dit = models.get("dit") or {}
    units = dit.get("units") or []
    return {"state": dit.get("state"), "counts": dit.get("counts"),
            "bytes": dit.get("bytes"),
            "devices": [unit.get("device") for unit in units]}


def _veda_calls(runtime):
    try:
        from dit_svc.veda_attn_star7 import status_report
        return status_report(runtime.veda_patch)
    except Exception:  # noqa: BLE001  工况读取失败不掩盖探针结果
        return None


def run_probe(runtime, request, operation=None):
    import torch
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    rt = runtime.rt
    backend = runtime.weights.backends["dit"]
    device = getattr(runtime, "device", None) or torch.device("cuda:0")
    started = time.perf_counter()
    warmup, runs = [], []
    index = -1
    memory_queried = False
    before = None
    cond = tags = latent = None
    frame_count = None
    try:
        before = require_workspace(torch, device, request["min_free_mb"])
        memory_queried = True
        resident_before = _dit_resident_snapshot(runtime)
        scale = request["scale"]
        backend._stage("DiT probe 合成条件（body 宽，跳过 refiner）", operation)
        width = cond_dim(runtime)
        cond, tags = synthetic_cond(torch, scale, request["seed"], width)
        cond = cond.to(device=device, dtype=torch.bfloat16)
        tags = tags.to(device)
        backend._stage("DiT probe 组装采样链", operation)
        latent, frame_count = _empty_av_latent(
            scale["width"], scale["height"], scale["length"])
        values = {"minimax_token_tags": tags}
        positive = [[cond, values]]
        sigmas = rt.unpack(BasicScheduler.execute(
            runtime.model_shifted, "simple", request["steps"], 1.0))[0]
        sampler_obj = rt.unpack(KSamplerSelect.execute("res_multistep"))[0]

        for index in range(request["warmup"] + request["repeat"]):
            is_warmup = index < request["warmup"]
            backend._stage("DiT probe %s %d"
                           % ("warmup" if is_warmup else "run", index + 1),
                           operation)
            require_workspace(torch, device, request["min_free_mb"])
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = cuda_memory(torch, device)
            tick = time.perf_counter()
            noise = rt.unpack(RandomNoise.execute(request["seed"]))[0]
            guider = rt.unpack(BasicGuider.execute(
                runtime.model_shifted, positive))[0]
            out = rt.unpack(SamplerCustomAdvanced.execute(
                noise, guider, sampler_obj, sigmas, latent))[0]
            torch.cuda.synchronize(device)
            video = out["samples"].unbind()[0]
            if not _latent_finite(torch, video):
                raise OperationError(
                    "NONFINITE_LATENT", "DiT 采样输出含 NaN/Inf", 422, False)
            elapsed = time.perf_counter() - tick
            del out, noise, guider, video  # never retain probe latents
            peak = int(torch.cuda.max_memory_allocated(device))
            row = {"iteration": index + 1, "sample_s": round(elapsed, 6),
                   "steps": request["steps"], "frames": int(frame_count),
                   "latent_finite": True,
                   "peak_allocated_bytes": peak,
                   "peak_reserved_bytes": int(
                       torch.cuda.max_memory_reserved(device)),
                   "workspace_peak_bytes": max(
                       0, peak - baseline["allocated_bytes"]),
                   "memory_after": cuda_memory(torch, device)}
            (warmup if is_warmup else runs).append(row)

        resident_after = _dit_resident_snapshot(runtime)
        if resident_after != resident_before:
            raise OperationError(
                "WEIGHT_MUTATED", "采样期间 DiT 权重驻留状态被改变", 500, False,
                {"before": resident_before, "after": resident_after})
    except Exception as exc:
        error = exc if isinstance(exc, OperationError) else OperationError.from_exception(exc)
        error.details = dict(error.details or {})
        memory = error.details.get("memory", {"queried": False})
        if memory_queried:
            try:
                memory = cuda_memory(torch, device)
            except Exception as diagnostic:  # noqa: BLE001
                memory = {"error": str(diagnostic)}
        error.details.update({"scale": request["scale"], "steps": request["steps"],
                              "failed_iteration": index + 1,
                              "warmup_completed": len(warmup),
                              "runs_completed": len(runs),
                              "memory": memory})
        if error is exc:
            raise
        raise error from exc
    finally:
        del cond, latent
    sample_times = [row["sample_s"] for row in runs]
    return {"kind": "dit_probe", "synthetic_inputs": True,
            "body_only": True,
            "token_refiner": "skipped (body-width cond)",
            "cond_dim": width,
            "scale": request["scale"], "steps": request["steps"],
            "seed": request["seed"],
            "warmup": warmup, "runs": runs,
            "veda_calls": _veda_calls(runtime),
            "summary": {"repeat": len(runs),
                        "sample_mean_s": round(
                            sum(sample_times) / len(sample_times), 6),
                        "sample_min_s": min(sample_times),
                        "sample_max_s": max(sample_times),
                        "peak_allocated_bytes": max(
                            row["peak_allocated_bytes"] for row in runs)},
            "memory_before": before,
            "memory_after": cuda_memory(torch, device),
            "elapsed_s": round(time.perf_counter() - started, 6)}
