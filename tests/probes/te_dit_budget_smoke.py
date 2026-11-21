"""DiT 部分常驻 + TE 上卡 + TE encode 的显存预算探针。

先 disk2ram 全量 56 units，再只把“除前 N 个 block 之外”的 unit 上卡，
然后让 TE 上卡并跑真实 prompt/参考图的条件编码，报告显存与状态。

用法（先 20 层，再逐步减少）：
  python tests/probes/te_dit_budget_smoke.py --server-root . \
      --defaults-file ... --offload-layers 20 \
      --prompt-file ... --image-file ... --out ...
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_remote_smoke import Client, free_port, wait_server  # noqa: E402


def action(client, model, name, body, timeout, report):
    started = time.perf_counter()
    status, receipt = client.request(
        "POST", "/api/weights/%s/%s" % (model, name), body)
    if status != 202:
        raise RuntimeError("%s %s -> %s: %s" % (model, name, status, receipt))
    operation = client.wait_operation(receipt["operation_id"], timeout)
    entry = {"model": model, "action": name, "status": operation["status"],
             "elapsed_s": round(time.perf_counter() - started, 2)}
    report.setdefault("actions", []).append(entry)
    print("[budget] %s %s -> %s (%.1fs)"
          % (model, name, operation["status"], entry["elapsed_s"]), flush=True)
    if operation["status"] != "done":
        raise RuntimeError("%s %s failed: %s"
                           % (model, name, operation["error"]))
    return operation


def weights_of(client, model):
    status, body = client.request("GET", "/api/weights/" + model)
    if status != 200:
        raise RuntimeError("weights %s -> %s: %s" % (model, status, body))
    return body["model"]


def vram_of(client):
    status, body = client.request("GET", "/api/status")
    return body.get("vram") if status == 200 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--defaults-file", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--offload-layers", type=int, required=True,
                        help="前 N 个 dit.blocks.N 留 RAM，其余上卡")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    log_path = report_path.with_suffix(".log")
    port = free_port()
    report = {"probe": "te_dit_budget_smoke", "offload_layers": args.offload_layers,
              "port": port, "verdict": "failed"}
    client = Client(port, timeout=60)
    process = None
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig")
        image = Path(args.image_file).read_bytes()

        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "h3_dit_server.py",
                 "--host", "127.0.0.1", "--port", str(port),
                 "--defaults-file", args.defaults_file,
                 "--out-dir", str(root / "outputs")],
                cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            report["created_pid"] = process.pid
            wait_server(client, time.monotonic() + 240, process)

            action(client, "dit", "disk2ram", {"mode": "all"}, 1800, report)
            units = [u["id"] for u in weights_of(client, "dit")["units"]]
            offloaded = ["dit.blocks.%d" % i for i in range(args.offload_layers)]
            keep = [u for u in units if u not in offloaded]
            report["keep_units"] = len(keep)
            report["offload_units"] = len(offloaded)
            action(client, "dit", "ram2vram",
                   {"mode": "units", "units": keep}, 900, report)
            dit_before = weights_of(client, "dit")
            report["dit_before"] = {
                "state": dit_before["state"],
                "counts": dit_before["counts"],
                "cuda_bytes": dit_before["bytes"]["resident_cuda"]}
            report["vram_after_dit"] = vram_of(client)

            action(client, "te", "disk2ram", {"mode": "all"}, 900, report)
            action(client, "te", "ram2vram", {"mode": "all"}, 300, report)
            te_before = weights_of(client, "te")
            assert te_before["state"] == "vram", te_before
            report["te_before"] = {"state": te_before["state"],
                                   "cuda_bytes": te_before["bytes"]["resident_cuda"]}
            report["vram_after_te"] = vram_of(client)

            payload = {
                "prompt": prompt,
                "image_b64s": [base64.b64encode(image).decode("ascii")],
                "mode": "ref2va",
                "width": args.width, "height": args.height,
                "ref_image_size": "match",
            }
            started = time.perf_counter()
            status, receipt = client.request("POST", "/api/infer/encode", payload)
            if status != 202:
                raise RuntimeError("encode -> %s: %s" % (status, receipt))
            operation = client.wait_operation(receipt["operation_id"], 900)
            elapsed = round(time.perf_counter() - started, 2)
            report["encode"] = {"status": operation["status"],
                                "elapsed_s": elapsed,
                                "error": operation["error"]}
            print("[budget] encode -> %s (%.1fs)"
                  % (operation["status"], elapsed), flush=True)
            if operation["status"] != "done":
                report["vram_after_encode"] = vram_of(client)
                raise RuntimeError("encode failed: %s" % (operation["error"],))
            result = operation["result"]
            report["encode"]["stats"] = result["stats"]
            report["encode"]["artifact"] = {
                "id": result["artifact"]["id"],
                "filename": result["artifact"]["filename"],
                "bytes": result["artifact"]["bytes"]}
            dit_after = weights_of(client, "dit")
            report["dit_after"] = {
                "state": dit_after["state"],
                "counts": dit_after["counts"],
                "cuda_bytes": dit_after["bytes"]["resident_cuda"]}
            report["vram_after_encode"] = vram_of(client)
            assert dit_after["bytes"]["resident_cuda"] == \
                dit_before["bytes"]["resident_cuda"], (
                    dit_before["bytes"]["resident_cuda"],
                    dit_after["bytes"]["resident_cuda"])
            report["verdict"] = "ok"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report["error"] = "%s: %s" % (type(error).__name__, error)
        try:
            report["actual"] = {name: weights_of(client, name)["state"]
                                for name in ("dit", "te")}
        except Exception:
            pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            report["created_process_stopped"] = True
        report["log_path"] = str(log_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        summary = {k: report.get(k) for k in
                   ("verdict", "error", "offload_layers", "keep_units",
                    "dit_before", "te_before", "vram_after_dit",
                    "vram_after_te", "encode", "vram_after_encode")}
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
