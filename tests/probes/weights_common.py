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


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_devices"] = torch.cuda.device_count()
            free, total = torch.cuda.mem_get_info(0)
            info["cuda_free_bytes"] = free
            info["cuda_total_bytes"] = total
    except Exception as e:  # noqa: BLE001
        info["torch"] = "import-failed:%s" % type(e).__name__
    try:
        import safetensors
        info["safetensors"] = safetensors.__version__
    except Exception:  # noqa: BLE001
        info["safetensors"] = "missing"
    try:
        import comfy_kitchen
        info["comfy-kitchen"] = getattr(comfy_kitchen, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        info["comfy-kitchen"] = "missing"
    return info


def schema_check(env: dict) -> dict:
    notes = []
    for k, want in EXPECTED_SCHEMA.items():
        got = env.get(k)
        if got != want:
            notes.append("%s: want %r, got %r" % (k, want, got))
    return {"expected": EXPECTED_SCHEMA, "mismatches": notes}


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        try:
            import resource
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
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


def write_report(report: dict, out: str, default_name: str) -> str:
    if not out:
        out = os.path.join(os.getcwd(), default_name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("REPORT: %s" % out)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])
    return out
