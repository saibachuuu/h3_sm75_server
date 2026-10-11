"""Star7 VEDA 的运行期辅助：NVML 健康检查 + warmup 分级常量 + 统计快照。

从旧 `dit_svc/veda_attn.py`（最小 fork 路线）保留下来的、与实现无关的部分：
  * check_nvml_sane —— sage 事故判据：显存查询全零/断流 = 驱动已病；
  * run_single_warmup / bench_attention 的框架思路被 _veda_warmup 取代
    （旧实现是"裸调 override"，结构性只能测到 dense 回退）。

新增（路线 B）：
  * snapshot(patch) —— 把 star7 VedaPatch.run.calls / engines 统计压成
    /api/status 可读的 dict，直接回答"稀疏到底跑没跑"。
"""
from __future__ import annotations

import math

# veda warmup 分级（tokens）。tier3=全尺寸 24.5k 是 sage 崩溃量级，
# 仅手动指定才跑。这是 **dense 回退 + 通路** 验证，不是稀疏率验证。
WARMUP_TIERS_VEDA = {"probe": 256, "tier1": 2048,
                     "tier2": 8192, "tier3": 24500}


def check_nvml_sane(torch):
    """NVML 健康检查：sage 事故判据——全零/断流 = 驱动已病。返回 (ok, 说明)。"""
    try:
        free, total = torch.cuda.mem_get_info()
        if free == 0 or total == 0:
            return False, "NVML 返回全零（free=%d total=%d），驱动可能已病，中止" % (free, total)
        return True, "free=%.2fGB total=%.2fGB" % (free / 1e9, total / 1e9)
    except Exception as e:  # noqa: BLE001
        return False, "NVML 调用失败（断流？）: %s" % e


def snapshot(patch) -> dict:
    """star7 VedaPatch 的运行期统计（/api/status.veda_calls）。

    VedaPatch.run.calls 是 collections.Counter：'sparse' 计稀疏次数，
    其余键是 dense 原因（'other attention' / 'layer outside the predictor' /
    'head mismatch' / 'full-attention layer' / 'full-attention step' /
    'sparsity 0%' / 'no sparse kernel' / 'layout' / 'error'）。
    engines 侧还有 kept_fraction / compute_fraction（占全量注意力的比例）。
    """
    if patch is None:
        return {"active": False}
    run = getattr(patch, "run", None)
    calls = dict(run.calls) if run is not None else {}
    engines = [e for e in getattr(patch, "_engines", {}).values() if e is not None]
    out = {
        "active": True,
        "calls": calls,
        "sparse": calls.get("sparse", 0),
        "dense": sum(n for r, n in calls.items() if r != "sparse"),
        "evaluations": getattr(run, "evaluations", 0) if run else 0,
        "failed": getattr(run, "failed", None) if run else None,
        "backend": (engines[0].backend.display if engines else "unresolved"),
        "video": getattr(run, "video", None) if run else None,
    }
    kept = [e.stats.kept_fraction() for e in engines]
    kept = [k for k in kept if k is not None]
    if kept:
        out["kept_fraction"] = round(sum(kept) / len(kept), 6)
    work = [e.stats.compute_fraction() for e in engines]
    work = [w for w in work if w is not None]
    if work:
        share = sum(work) / len(work)
        dense = out["dense"]
        if dense:  # dense 调用按全量计
            share = (share * out["sparse"] + dense) / (out["sparse"] + dense)
        out["compute_fraction"] = round(min(1.0, share), 6)
    for e in engines:
        chunking = getattr(e, "chunking", None)
        if chunking:
            out.setdefault("chunking", dict(chunking))
    return out


def warmup_tier_tokens(level: str) -> int:
    if level not in WARMUP_TIERS_VEDA:
        raise ValueError("未知 warmup 级别: %s" % level)
    return WARMUP_TIERS_VEDA[level]


def is_finite_ok(value) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
