"""probe 00：版本与源码基线（CPU only，不读权重）。

零 CUDA 接触证明：本文件只用 importlib.metadata 读发行版元数据，
不 import torch/safetensors 包，更不调用任何 torch.cuda.*。
CUDA 字段缺席即证据；GPU 细节只允许 21（显式 GPU probe）上报。

schema 不符：verdict=unverified 且 exit 2，不许 ok。

用法（远端，主代理运行）：
  python tests/probes/weights_00_versions.py --comfy-root C:\\...\\ComfyUI

输出 JSON：torch/safetensors/comfy-kitchen 版本、2.13 分支 schema 比对、
9 个构造相关源码 sha256。供 P0 证据链存档。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import weights_common as C  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="weights probe 00: versions")
    ap.add_argument("--comfy-root", default="")
    ap.add_argument("--extra-roots", default="",
                    help="逗号分隔的额外源码根（含 minimax/ 时）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    roots = [r for r in [args.comfy_root] + args.extra_roots.split(",")
             if r.strip()]
    env = C.env_info(include_cuda=False)
    assert "cuda_available" not in env and "cuda_free_bytes" not in env, env
    report = {
        "probe": "weights_00_versions",
        "cuda_touch": "none (importlib.metadata only; no torch.cuda.* called)",
        "env": env,
        "sources": C.collect_sources(roots),
        "verdict": "ok",
    }
    try:
        C.require_schema(env, report)
    except SystemExit:
        C.write_report(report, args.out, "weights_probe_00.json")
        raise
    C.write_report(report, args.out, "weights_probe_00.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
