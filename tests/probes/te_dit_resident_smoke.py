"""DiT 与 TE 同时驻留显存时跑 TE 条件编码。

验证目标：DiT 全量 56 units 已在 cuda 的情况下，TE encode 仍能完成，
且 DiT 没有被 ComfyUI 的 load_models_gpu 隐式卸载/搬迁。

用法：
  python tests/probes/te_dit_resident_smoke.py --server-root . \
      --defaults-file C:\\Users\\sheepserver\\Desktop\\h3_explicit_defaults.json \
      --prompt-file C:\\Users\\sheepserver\\Desktop\\TEST_PROMPT.txt \
      --image-file C:\\Users\\sheepserver\\Desktop\\_cond_ref_check.png \
      --out C:\\Users\\sheepserver\\Desktop\\_te_dit_resident.json
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
    print("[smoke] %s %s -> %s (%.1fs)"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--defaults-file", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    log_path = report_path.with_suffix(".log")
    port = free_port()
    report = {"probe": "te_dit_resident_smoke", "port": port,
              "width": args.width, "height": args.height, "verdict": "failed"}
    client = Client(port, timeout=60)
    process = None
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig")
        image = Path(args.image_file).read_bytes()
        report["prompt_chars"] = len(prompt)
        report["image_bytes"] = len(image)

        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "h3_dit_server.py",
                 "--host", "127.0.0.1", "--port", str(port),
                 "--defaults-file", args.defaults_file,
                 "--out-dir", str(root / "outputs")],
                cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            report["created_pid"] = process.pid
            wait_server(client, time.monotonic() + 240, process)

            # 1) DiT 全量上卡
            action(client, "dit", "disk2ram", {"mode": "all"}, 1800, report)
            action(client, "dit", "ram2vram", {"mode": "all"}, 900, report)
            dit_before = weights_of(client, "dit")
            report["dit_before"] = {"state": dit_before["state"],
                                    "counts": dit_before["counts"],
                                    "bytes": dit_before["bytes"]}
            assert dit_before["state"] == "vram", dit_before["state"]
            assert dit_before["counts"]["vram"] == 56, dit_before["counts"]

            # 2) TE 也上卡（4.86GB），与 DiT 同时驻留
            action(client, "te", "disk2ram", {"mode": "all"}, 900, report)
            action(client, "te", "ram2vram", {"mode": "all"}, 300, report)
            te_before = weights_of(client, "te")
            report["te_before"] = {"state": te_before["state"],
                                   "counts": te_before["counts"],
                                   "bytes": te_before["bytes"]}
            assert te_before["state"] == "vram", te_before["state"]

            # 3) TE encode（参考图 + 真实 prompt）
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
                                "operation_id": operation["id"],
                                "error": operation["error"]}
            print("[smoke] encode -> %s (%.1fs)"
                  % (operation["status"], elapsed), flush=True)
            if operation["status"] != "done":
                raise RuntimeError("encode failed: %s" % (operation["error"],))
            result = operation["result"]
            report["encode"]["stats"] = result["stats"]
            report["encode"]["artifact"] = {
                "id": result["artifact"]["id"],
                "filename": result["artifact"]["filename"],
                "bytes": result["artifact"]["bytes"]}
            report["encode"]["audit"] = result["audit"]["units"]

            # 4) DiT / TE 都仍在显存，且 DiT 未被搬走
            dit_after = weights_of(client, "dit")
            te_after = weights_of(client, "te")
            report["dit_after"] = {"state": dit_after["state"],
                                   "counts": dit_after["counts"],
                                   "bytes": dit_after["bytes"]}
            report["te_after"] = {"state": te_after["state"],
                                  "counts": te_after["counts"],
                                  "bytes": te_after["bytes"]}
            assert dit_after["state"] == "vram", dit_after["state"]
            assert dit_after["counts"]["vram"] == 56, dit_after["counts"]
            assert dit_after["bytes"]["resident_cuda"] == \
                dit_before["bytes"]["resident_cuda"]
            assert te_after["state"] == "vram", te_after["state"]
            status, body = client.request("GET", "/api/status")
            report["vram"] = body.get("vram")
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
                   ("verdict", "error", "encode", "dit_before", "dit_after",
                    "te_before", "te_after", "vram")}
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
