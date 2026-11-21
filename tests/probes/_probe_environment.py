"""Read-only dependency/header inventory; never constructs a model or imports CUDA.

Run with the actual service interpreter. Output is JSON on stdout unless --out
is explicitly supplied. This script does not start/stop services or change git.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys


PACKAGES = ("torch", "safetensors", "psutil", "comfy-kitchen", "gguf", "imageio-ffmpeg")
SOURCE_FILES = (
    "comfy/sd.py", "comfy/model_management.py", "comfy/model_patcher.py",
    "comfy/model_detection.py", "comfy/supported_models.py", "comfy/model_base.py",
    "comfy/ldm/minimax/model.py", "comfy/ldm/modules/attention.py",
    "comfy/ops.py", "comfy/quant_ops.py", "comfy/nested_tensor.py",
    "comfy_extras/nodes_minimax_h3.py",
    "custom_nodes/clipproj/clipproj_nodes.py",
    "custom_nodes/clipproj/clipproj_projection.py",
)


def file_record(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "bytes": stat.st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def safetensors_header(path):
    """Inspect the bounded JSON header, not tensor payload or CUDA storage."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("incomplete safetensors header length")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 32 * 1024 * 1024:
            raise ValueError("header length outside probe bounds")
        header = stream.read(size)
        if len(header) != size:
            raise ValueError("incomplete safetensors header")
    data = json.loads(header)
    if not isinstance(data, dict):
        raise ValueError("header must be an object")
    tensors = {key: value for key, value in data.items() if key != "__metadata__"}
    for key, descriptor in tensors.items():
        if not isinstance(descriptor, dict):
            raise ValueError("invalid tensor descriptor: " + key)
    stat = path.stat()
    return {"path": str(path.resolve()), "file_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "header_bytes": size,
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "metadata": data.get("__metadata__", {}), "tensors": tensors}


def collect(comfy_dir=None, runner_dir=None, checkpoints=()):
    report = {"python": sys.version, "executable": sys.executable,
              "platform": sys.platform, "packages": {}, "sources": {},
              "checkpoints": [], "models_constructed": 0,
              "cuda_imported_by_probe": False}
    for package in PACKAGES:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    if comfy_dir:
        report["sources"] = {relative: file_record(Path(comfy_dir) / relative)
                             for relative in SOURCE_FILES}
    if runner_dir:
        report["runner"] = {name: file_record(Path(runner_dir) / name)
                            for name in ("infer_h3_turing.py", "h3_mlp_chunk.py")}
    for path in checkpoints:
        report["checkpoints"].append(safetensors_header(path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-dir")
    parser.add_argument("--runner-dir")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--out", help="Explicit report destination; otherwise stdout only")
    args = parser.parse_args()
    report = collect(args.comfy_dir, args.runner_dir, args.checkpoint)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        destination = Path(args.out).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
