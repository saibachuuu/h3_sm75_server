"""Submit a configurable TE inference probe to an EXISTING server.

No models are loaded/offloaded, no server is started/stopped by this client.
Prepare TE explicitly with /api/weights/te/* first. Other resident models stay put.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_remote_smoke import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8190")
    parser.add_argument("--text-chars", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-free-mb", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"scale": {"text_chars": args.text_chars, "image_width": args.image_width,
                         "image_height": args.image_height, "images": args.images},
               "repeat": args.repeat, "warmup": args.warmup, "seed": args.seed,
               "min_free_mb": args.min_free_mb}
    client = Client(base_url=args.url)
    status, receipt = client.request("POST", "/api/infer/te_probe", payload)
    if status != 202:
        report = {"status": status, "error": receipt}
    else:
        report = client.wait_operation(receipt["operation_id"], args.timeout)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == 202 and report["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
