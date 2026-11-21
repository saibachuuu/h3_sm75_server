"""probe 00：版本与源码基线（CPU only，不读权重）。

零 CUDA 接触证明：本文件只用 importlib.metadata 读发行版元数据，
不 import torch/safetensors 包，更不调用任何 torch.cuda.*。
CUDA 字段缺席即证据；GPU 细节只允许 21（显式 GPU probe）上报。
CPU 钳制：main 第一行设 CUDA_VISIBLE_DEVICES='-1'（远端实测：PS 设 ''
无效，'-1' 才隐藏 GPU）；失败也写报告（finally）。

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import argparse
    ap = argparse.ArgumentParser(description="weights probe 00: versions")
    ap.add_argument("--comfy-root", default="")
    ap.add_argument("--extra-roots", default="",
                    help="逗号分隔的额外源码根（含 minimax/ 时）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    report: dict = {
        "probe": "weights_00_versions",
        "cuda_visible_devices": "-1",
        "cuda_touch": "none (importlib.metadata only; no torch.cuda.* called)",
    }
    rc = 0
    try:
        roots = [r for r in [args.comfy_root] + args.extra_roots.split(",")
                 if r.strip()]
        env = C.env_info(include_cuda=False)
        assert "cuda_available" not in env and "cuda_free_bytes" not in env, env
        report["env"] = env
        report["sources"] = C.collect_sources(roots)
        C.require_schema(env, report)  # 不符 -> unverified + exit 2
        report["verdict"] = "ok"
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        if report.get("verdict") not in ("unverified", "refused"):
            report["verdict"] = "refused"
            report["error"] = "SystemExit: %s" % e
    except BaseException as e:  # noqa: BLE001
        rc = 1
        report["verdict"] = "failed"
        report["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        C.write_report(report, args.out, "weights_probe_00.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
