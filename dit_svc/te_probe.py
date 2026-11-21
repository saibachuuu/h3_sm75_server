"""Bounded synthetic-input TE probe; real encoder/projection, no fake matmul.

Scale is input characters and image pixels, not a promised encoder token count.
This module is safe to import without torch/Comfy. Probe results are metrics only,
not conditioning artifacts. All execution uses the existing resident TE encoder.
"""
import time

from dit_svc.operations import OperationError


def validate_request(payload):
    from dit_svc import api_validation as api
    api._require_object(payload)
    api._reject_unknown(payload, {"scale", "repeat", "warmup", "seed", "min_free_mb",
                                  "request_id"})
    scale = payload.get("scale")
    if not isinstance(scale, dict):
        raise api.invalid("scale 必须是 JSON 对象", field="scale")
    api._reject_unknown(scale, {"text_chars", "image_width", "image_height", "images"})
    cleaned = {
        "text_chars": api._strict_int(scale, "text_chars", 512, lo=1, hi=api.MAX_PROMPT_CHARS),
        "image_width": api._strict_int(scale, "image_width", 512, lo=32, hi=1600),
        "image_height": api._strict_int(scale, "image_height", 512, lo=32, hi=1600),
        "images": api._strict_int(scale, "images", 1, lo=1, hi=9),
    }
    if cleaned["image_width"] % 32 or cleaned["image_height"] % 32:
        raise api.invalid("测试图片宽高必须是 32 的倍数", field="scale")
    return {"scale": cleaned,
            "repeat": api._strict_int(payload, "repeat", 3, lo=1, hi=20),
            "warmup": api._strict_int(payload, "warmup", 1, lo=0, hi=5),
            "seed": api._strict_int(payload, "seed", 0, lo=0, hi=2 ** 63 - 1),
            "min_free_mb": api._strict_int(payload, "min_free_mb", 256, lo=0, hi=16384),
            "request_id": api._request_id(payload)}


def synthetic_inputs(scale, seed):
    import torch
    # Local generator only: do not change the server's global sampling RNG.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pattern = "A camera follows a person walking through a sunlit garden. "
    text = (pattern * (scale["text_chars"] // len(pattern) + 1))[:scale["text_chars"]]
    text = text[:-1] + "."  # no tokenizer .strip() length ambiguity
    refs = [{"type": "image", "data": torch.rand(
        (1, scale["image_height"], scale["image_width"], 3),
        generator=generator, dtype=torch.float32)} for _ in range(scale["images"])]
    return text, refs


def cuda_memory(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": int(free), "total_bytes": int(total),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device))}


def require_workspace(torch, device, min_free_mb):
    memory = cuda_memory(torch, device)
    # Unused allocator reserve can be reused without moving any model.
    available = memory["free_bytes"] + max(
        0, memory["reserved_bytes"] - memory["allocated_bytes"])
    if available < min_free_mb * 1024 ** 2:
        raise OperationError("INSUFFICIENT_VRAM", "可用显存低于请求的最小余量",
                             409, True, {"memory": memory, "min_free_mb": min_free_mb,
                                         "available_bytes": available})
    return memory


def run_probe(backend, request, operation=None):
    import torch
    device = torch.device("cuda:0")
    started = time.perf_counter()
    warmup, runs = [], []
    index = -1
    memory_queried = False
    try:
        # One context protects setup, warmup, measured runs and exit audit. There
        # are no unguarded gaps between iterations for helper-thread eviction.
        with backend._encode_session(operation) as (encode, proof):
            before = require_workspace(torch, device, request["min_free_mb"])
            memory_queried = True
            text, refs = synthetic_inputs(request["scale"], request["seed"])
            for index in range(request["warmup"] + request["repeat"]):
                is_warmup = index < request["warmup"]
                backend._stage("TE probe %s %d" % ("warmup" if is_warmup else "run", index + 1),
                               operation)
                require_workspace(torch, device, request["min_free_mb"])
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline = cuda_memory(torch, device)
                tick = time.perf_counter()
                out, stats = encode(text, refs, "ref2va")
                torch.cuda.synchronize(device)
                if not torch.isfinite(out["cond"]).all():
                    raise OperationError("NONFINITE_COND", "TE 条件输出包含 NaN/Inf", 422, False)
                elapsed = time.perf_counter() - tick
                del out  # never retain cond tensors or write probe artifacts
                peak = int(torch.cuda.max_memory_allocated(device))
                row = {"iteration": index + 1, "wall_s": round(elapsed, 6),
                       "tokenize_s": stats["tokenize_s"], "encode_s": stats["encode_s"],
                       "output_cond_tokens": stats["n_tokens"],
                       "output_cond_shape": stats["cond_shape"], "output_finite": True,
                       "peak_allocated_bytes": peak,
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                       "workspace_peak_bytes": max(0, peak - baseline["allocated_bytes"]),
                       "memory_after": cuda_memory(torch, device)}
                (warmup if is_warmup else runs).append(row)
    except Exception as exc:
        error = exc if isinstance(exc, OperationError) else OperationError.from_exception(exc)
        error.details = dict(error.details or {})
        memory = error.details.get("memory", {"queried": False})
        if memory_queried:
            try:
                memory = cuda_memory(torch, device)
            except Exception as diagnostic:
                memory = {"error": str(diagnostic)}
        error.details.update({"scale": request["scale"], "failed_iteration": index + 1,
                              "warmup_completed": len(warmup), "runs_completed": len(runs),
                              "memory": memory})
        if error is exc:
            raise
        raise error from exc
    encode_times = [row["encode_s"] for row in runs]
    return {"kind": "te_probe", "synthetic_inputs": True, "scale": request["scale"],
            "actual_input": {"text_chars": len(text), "images": len(refs),
                             "reference_hw": [[int(ref["data"].shape[1]),
                                                int(ref["data"].shape[2])] for ref in refs]},
            "token_count_note": "output_cond_tokens 是输出条件长度，不保证等于编码器内部计算序列长度",
            "seed": request["seed"], "warmup": warmup, "runs": runs, "residence": proof,
            "summary": {"repeat": len(runs),
                        "encode_mean_s": round(sum(encode_times) / len(encode_times), 6),
                        "encode_min_s": min(encode_times), "encode_max_s": max(encode_times),
                        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in runs)},
            "memory_before": before, "memory_after": cuda_memory(torch, device),
            "elapsed_s": round(time.perf_counter() - started, 6)}
