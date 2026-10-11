"""Star7 VEDA 的运行期辅助：NVML 检查 + 统计抓取/累计 + logger 接出。

路线 B 的"探针"设计目标：**反映正版 Veda 工况**，即回答三件事
  1. SM75 稀疏内核到底可用吗（backend resolve + self_test，启动即知）；
  2. 每个采样 run 里稀疏真跑了几次、相对全量注意力算了几成；
  3. 为什么没走稀疏（dense 原因计数）。

为什么需要"累计"层：star7 的 VedaPatch.on_cleanup() 会在每次采样 run 结束
（comfy/samplers.py:1267 的 finally 必调）把 self.run 清零。WebUI 每 2s 轮询
/api/status，靠原始 run 计数会任务一结束就归零，永远看不到真数字。所以
bridge 包一层 on_cleanup：**在清零之前抓取**并累计到模块级 CUMULATIVE。
"""
from __future__ import annotations

import logging
import sys
import threading

# veda warmup 分级（tokens）。tier3=全尺寸 24.5k 是 sage 崩溃量级，
# 仅手动指定才跑。这是 **dense 回退 + 通路** 验证，不是稀疏率验证。
WARMUP_TIERS_VEDA = {"probe": 256, "tier1": 2048,
                     "tier2": 8192, "tier3": 24500}

_LOGGER_NAME = "Star7-H3-VEDA"   # star7 vendored 包的 logger（status.py/backends）
_lock = threading.Lock()

# 启动时 backend resolve 的结果（self_test 过 = SM75 kernel 可用）
BACKEND_INFO = {"resolved": False, "ok": False, "display": None, "report": None}

# 跨 run 累计（在 patch.on_cleanup 之前抓取，故不会被清零）
CUMULATIVE = {
    "runs": 0,
    "sparse": 0,
    "dense": 0,
    "reasons": {},            # dense 原因 -> 次数
    "evaluations": 0,         # 到达 layer0 的 model forward 次数
    "failed": None,           # 最近一次失败原因（成功为 None）
    "compute_fraction_last": None,
    "kept_fraction_last": None,
    "video_last": None,
    "backend_last": None,
    "sparse_per_run": [],     # 每 run 的 sparse 次数（最近 8 单）
}


def install_logger(log=print) -> bool:
    """把 star7 的诊断 logger 接到 stdout。

    star7 在每个 run 结束时会打一行正版工况摘要（status.py 的
    'Attention computed X% of full attention (Y% skipped)'），不接出来就
    全被 logging 默认配置吞掉。返回是否成功接管。
    """
    try:
        lg = logging.getLogger(_LOGGER_NAME)
        lg.setLevel(logging.INFO)
        if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            lg.addHandler(h)
        lg.propagate = False
        log("[veda-probe] star7 诊断日志已接到 stdout（INFO）")
        return True
    except Exception as e:  # noqa: BLE001
        log("[veda-probe] star7 日志接管失败: %s: %s" % (type(e).__name__, e))
        return False


def check_nvml_sane(torch):
    """NVML 健康检查：sage 事故判据——全零/断流 = 驱动已病。返回 (ok, 说明)。"""
    try:
        free, total = torch.cuda.mem_get_info()
        if free == 0 or total == 0:
            return False, ("NVML 返回全零（free=%d total=%d），驱动可能已病，中止"
                           % (free, total))
        return True, "free=%.2fGB total=%.2fGB" % (free / 1e9, total / 1e9)
    except Exception as e:  # noqa: BLE001
        return False, "NVML 调用失败（断流？）: %s" % e


def run_snapshot(patch) -> dict:
    """**单个 run** 的快照（VedaPatch.run.calls 等）。patch 可为 None。

    calls 键：'sparse' 是稀疏真跑次数；其余键是 dense 原因：
      other attention / layer outside the predictor / head mismatch /
      full-attention layer / full-attention step / sparsity 0% /
      no sparse kernel / layout / error
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
        "video": getattr(run, "video", None) if run else None,
        "backend": (engines[0].backend.display if engines else
                    (BACKEND_INFO.get("display") or "unresolved")),
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


def _record_run(snapshot: dict) -> None:
    """把一个 run 的快照累计进 CUMULATIVE（在 on_cleanup 清零前调用）。"""
    with _lock:
        CUMULATIVE["runs"] += 1
        CUMULATIVE["sparse"] += int(snapshot.get("sparse", 0))
        CUMULATIVE["dense"] += int(snapshot.get("dense", 0))
        CUMULATIVE["evaluations"] += int(snapshot.get("evaluations", 0))
        for reason, n in (snapshot.get("calls") or {}).items():
            if reason == "sparse":
                continue
            CUMULATIVE["reasons"][reason] = CUMULATIVE["reasons"].get(reason, 0) + int(n)
        CUMULATIVE["failed"] = snapshot.get("failed")
        CUMULATIVE["compute_fraction_last"] = snapshot.get("compute_fraction")
        CUMULATIVE["kept_fraction_last"] = snapshot.get("kept_fraction")
        CUMULATIVE["video_last"] = snapshot.get("video")
        CUMULATIVE["backend_last"] = snapshot.get("backend")
        hist = CUMULATIVE["sparse_per_run"]
        hist.append(int(snapshot.get("sparse", 0)))
        del hist[:-8]                     # 只留最近 8 单


def wrap_cleanup(patch, log=print) -> bool:
    """包住 patch.on_cleanup：先抓统计再交给原实现（原实现会清零 run）。

    star7 的回调是 `lambda model_patcher: patch.on_cleanup()` —— 运行时才查
    属性，所以事后替换 patch.on_cleanup 生效。返回是否成功包装。
    """
    orig = getattr(patch, "on_cleanup", None)
    if not callable(orig) or getattr(patch, "_h3_stats_wrapped", False):
        return False

    def _wrapped(*args, **kwargs):
        snap = run_snapshot(patch)
        if snap.get("active"):
            _record_run(snap)
            log("[veda-probe] run #%d 收尾: sparse=%s dense=%s "
                "compute_fraction=%s | 累计 sparse=%d/%d runs"
                % (CUMULATIVE["runs"], snap.get("sparse"), snap.get("dense"),
                   snap.get("compute_fraction"), CUMULATIVE["sparse"],
                   CUMULATIVE["runs"]))
        return orig(*args, **kwargs)

    patch.on_cleanup = _wrapped
    patch._h3_stats_wrapped = True
    return True


def status_report(patch) -> dict | None:
    """/api/status.veda_calls 的结构（反映了正版工况的三件事）。"""
    if patch is None and not BACKEND_INFO.get("resolved"):
        return None
    with _lock:
        cum = dict(CUMULATIVE)
        cum["reasons"] = dict(CUMULATIVE["reasons"])
        cum["sparse_per_run"] = list(CUMULATIVE["sparse_per_run"])
    report = {
        "active": patch is not None,
        "backend": BACKEND_INFO.get("display"),
        "backend_report": BACKEND_INFO.get("report"),
        "cumulative": cum,
        # 当前正在跑的 run（未 cleanup）；空闲时全 0
        "current": run_snapshot(patch) if patch is not None else {"active": False},
    }
    return report


def reset_stats() -> None:
    """（测试用）清空累计与 backend 信息。"""
    with _lock:
        CUMULATIVE["runs"] = 0
        CUMULATIVE["sparse"] = 0
        CUMULATIVE["dense"] = 0
        CUMULATIVE["evaluations"] = 0
        CUMULATIVE["reasons"] = {}
        CUMULATIVE["failed"] = None
        CUMULATIVE["compute_fraction_last"] = None
        CUMULATIVE["kept_fraction_last"] = None
        CUMULATIVE["video_last"] = None
        CUMULATIVE["backend_last"] = None
        CUMULATIVE["sparse_per_run"] = []
    BACKEND_INFO.update(resolved=False, ok=False, display=None, report=None)
