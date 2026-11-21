"""远端 probe 公共件：版本/哈希/peak 上报、JSON 报告（只读 + 显式 unit）。

安全规则：probe 只接受显式 unit 选择器，拒绝 mode=all；
任何实际权重读取都经 access-log 记录；本文件自身不读权重。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

EXPECTED_SCHEMA = {
    "torch": "2.13.0+cu132",
    "safetensors": "0.7.0",
    "comfy-kitchen": "0.2.40",
}

# 参与构造/绑定的源码文件（相对 comfy-root 或 minimax-root 的候选位置，
# probe 按存在性逐个哈希，缺失记 missing，不中断）。
SOURCE_FILES = [
    "comfy/sd.py",
    "comfy/model_base.py",
    "comfy/model_detection.py",
    "comfy/model_management.py",
    "comfy/model_patcher.py",
    "comfy/supported_models.py",
    "comfy/ops.py",
    "comfy/quant_ops.py",
    "minimax/model.py",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_sources(roots: list) -> dict:
    out = {}
    for rel in SOURCE_FILES:
        found = None
        for r in roots:
            p = os.path.join(r, rel)
            if os.path.isfile(p):
                found = p
                break
        if found is None:
            out[rel] = {"status": "missing"}
        else:
            out[rel] = {"status": "ok", "path": found,
                        "sha256": sha256_file(found),
                        "size": os.path.getsize(found)}
    return out


def dist_version(*names: str) -> str:
    """只读发行版元数据，不 import 被测包（无代码执行、无 CUDA）。"""
    try:
        from importlib import metadata as _md
    except ImportError:
        return "unknown"
    for n in names:
        try:
            return str(_md.version(n))
        except Exception:  # noqa: BLE001
            continue
    return "missing"


def env_info(include_cuda: bool = False) -> dict:
    """版本只读（importlib.metadata，不 import 包）。

    include_cuda=True 仅限显式 GPU probe（21）调用；00/10/20 必须 False：
    torch.cuda.is_available()/mem_get_info 会初始化 CUDA，与“不触 GPU”
    承诺冲突。CUDA 字段缺席即证明未碰。
    """
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    info["torch"] = dist_version("torch")
    info["safetensors"] = dist_version("safetensors")
    info["comfy-kitchen"] = dist_version("comfy-kitchen", "comfy_kitchen")
    if include_cuda:
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_devices"] = torch.cuda.device_count()
                free, total = torch.cuda.mem_get_info(0)
                info["cuda_free_bytes"] = free
                info["cuda_total_bytes"] = total
        except Exception as e:  # noqa: BLE001
            info["cuda_available"] = "query-failed:%s" % type(e).__name__
    return info


def schema_check(env: dict) -> dict:
    notes = []
    for k, want in EXPECTED_SCHEMA.items():
        got = env.get(k)
        if got != want:
            notes.append("%s: want %r, got %r" % (k, want, got))
    verdict = "ok" if not notes else "unverified"
    return {"expected": EXPECTED_SCHEMA, "mismatches": notes,
            "verdict": verdict}


def require_schema(env: dict, report: dict) -> None:
    """版本 schema 不符：不许 verdict ok。显式 unverified + exit 2。"""
    sc = schema_check(env)
    report["schema_check"] = sc
    if sc["verdict"] != "ok":
        report["verdict"] = "unverified"
        raise SystemExit(2)


def rss_bytes() -> int:
    """当前 RSS（不是 peak；peak 另见 rss_peak_bytes）。"""
    return rss_current_bytes()


def rss_current_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                             wintypes.DWORD]
            GetProcessMemoryInfo.restype = wintypes.BOOL
            current = wintypes.HANDLE(0xFFFFFFFFFFFFFFFF)
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if GetProcessMemoryInfo(current, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
        except Exception:  # noqa: BLE001
            pass
        return -1
    try:  # Linux /proc（无 psutil 时）
        with open("/proc/self/statm") as f:
            parts = f.read().split()
        import os as _os
        return int(parts[1]) * _os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return -1


def rss_peak_bytes() -> int:
    """进程真实 CPU peak（ru_maxrss / PeakWorkingSetSize）。"""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                             wintypes.DWORD]
            GetProcessMemoryInfo.restype = wintypes.BOOL
            current = wintypes.HANDLE(0xFFFFFFFFFFFFFFFF)
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if GetProcessMemoryInfo(current, ctypes.byref(pmc), pmc.cb):
                return int(pmc.PeakWorkingSetSize)
        except Exception:  # noqa: BLE001
            pass
        return -1
    try:  # Linux/macOS getrusage（Linux 单位 KB）
        import resource
        scale = 1024 if sys.platform.startswith("linux") else 1
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
    except Exception:  # noqa: BLE001
        return -1


def base_parser(name: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=name)
    ap.add_argument("--header", required=True,
                    help="DiT safetensors 路径（只读 header + 所选 keys）")
    ap.add_argument("--model-id", default="dit")
    ap.add_argument("--comfy-root", default="",
                    help="ComfyUI 目录（含 comfy/ 与 ldm/minimax 或 minimax/）")
    ap.add_argument("--units", default="dit.blocks.0",
                    help="逗号分隔的显式 unit 选择器；禁止 all")
    ap.add_argument("--out", default="",
                    help="JSON 报告输出路径（默认 stdout 旁 weights_probe_*.json）")
    return ap


def parse_units(s: str) -> list:
    units = [u.strip() for u in s.split(",") if u.strip()]
    if not units:
        raise SystemExit("empty --units")
    if any(u == "all" for u in units):
        raise SystemExit("refusing mode=all in probe")
    return units


# 低负载上限：resolve 后的 unit 数与源端字节（防 range 展开成全量）。
MAX_PROBE_UNITS = 2
MAX_PROBE_SOURCE_BYTES = 1 << 30  # 所选 units 源端合计 1GiB


def enforce_probe_budget(catalog, resolved: list) -> dict:
    """resolve 后强制预算：unit 数 + 源端字节。超限直接拒绝运行。"""
    if not resolved:
        raise SystemExit("probe refuses empty resolve")
    if len(resolved) > MAX_PROBE_UNITS:
        raise SystemExit("probe refuses %d resolved units (max %d); "
                         "range expanded too far?" % (len(resolved),
                                                      MAX_PROBE_UNITS))
    want = catalog.expected_bytes_of(resolved)
    if want > MAX_PROBE_SOURCE_BYTES:
        raise SystemExit("probe refuses %d source bytes (max %d)" % (
            want, MAX_PROBE_SOURCE_BYTES))
    return {"resolved": list(resolved),
            "expected_source_bytes": want,
            "budget_units": MAX_PROBE_UNITS,
            "budget_bytes": MAX_PROBE_SOURCE_BYTES}


def prepare_comfy_import(comfy_root: str, *, cpu: bool) -> dict:
    """Comfy import 前钳制（必须在 import comfy.* 之前调用）。

    - sys.argv 只留程序名：防 comfy.cli_args 误读 probe 参数；
    - comfy.cli_args.args.cpu/gpu_only 按需设定（缺失则如实记录，不硬编）；
    - 调用方在 import 前后 assert torch.cuda.is_initialized() is False（CPU probe）。
    """
    note: dict = {"argv_cleared": True, "cli_args": "unknown"}
    if comfy_root and comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    prog = sys.argv[0] if sys.argv else "probe"
    sys.argv = [prog]
    try:
        import comfy.cli_args as _ca
        _ca.args.cpu = bool(cpu)
        _ca.args.gpu_only = not bool(cpu)
        note["cli_args"] = "cpu=%s gpu_only=%s" % (_ca.args.cpu,
                                                  _ca.args.gpu_only)
    except Exception as e:  # noqa: BLE001
        note["cli_args"] = "missing-or-unset:%s" % type(e).__name__
    return note


def release_staging_proof(objs: list) -> dict:
    """staging 释放证明：weakref 采样 + gc 后断言无强引用残留。

    调用方在 purge 后、purge audit 前传入全部 staging 张量对象；
    返回 {"staged_tensors", "alive_after_release"}，alive 必须为 0。
    """
    import gc
    import weakref
    refs = [weakref.ref(v) for v in objs if v is not None]
    n = len(refs)
    objs.clear()
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    return {"staged_tensors": n, "alive_after_release": alive}


def write_report(report: dict, out: str, default_name: str) -> str:
    if not out:
        out = os.path.join(os.getcwd(), default_name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("REPORT: %s" % out)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])
    return out
