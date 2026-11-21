"""真实远端冒烟：空壳启动 → TE 四动作 → resident encode → 释放。

只操作它自己启动的进程；不终止用户既有服务。每一步都轮询 operation，
失败时保留实际状态与错误 envelope。

用法（远端 Comfy py 3.12）：
  python tests/probes/te_remote_smoke.py --server-root . \
      --defaults-file C:\\Users\\sheepserver\\Desktop\\h3_explicit_defaults.json \
      --out C:\\Users\\sheepserver\\Desktop\\_te_smoke.json
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Client:
    def __init__(self, port, timeout=30):
        self.base = "http://127.0.0.1:%d" % port
        self.timeout = timeout

    def request(self, method, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            payload = error.read().decode("utf-8", "replace")
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"raw": payload}
            return error.code, payload

    def wait_operation(self, operation_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, body = self.request("GET", "/api/operations/" + operation_id)
            if status != 200:
                raise RuntimeError("operation GET %s: %s" % (status, body))
            operation = body["operation"]
            if operation["status"] in ("done", "failed", "cancelled"):
                return operation
            time.sleep(0.5)
        raise TimeoutError("operation %s exceeded %.0fs"
                           % (operation_id, timeout))


def wait_server(client, deadline, process):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited %s" % process.returncode)
        try:
            status, body = client.request("GET", "/api/status")
            if status == 200 and body.get("env_ready"):
                return body
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.3)
    raise TimeoutError("server did not become ready")


def tiny_image():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (96, 96), (96, 128, 160)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_action(client, model, action, body, timeout, report):
    started = time.perf_counter()
    status, receipt = client.request(
        "POST", "/api/weights/%s/%s" % (model, action), body)
    if status != 202:
        raise RuntimeError("%s %s -> %s: %s" % (model, action, status, receipt))
    operation = client.wait_operation(receipt["operation_id"], timeout)
    entry = {
        "action": action,
        "status": operation["status"],
        "elapsed_s": round(time.perf_counter() - started, 2),
        "operation_id": operation["id"],
        "result": operation["result"],
        "error": operation["error"],
    }
    report.setdefault("actions", []).append(entry)
    print("[smoke] %s %s -> %s (%.1fs)"
          % (model, action, operation["status"], entry["elapsed_s"]), flush=True)
    if operation["status"] != "done":
        raise RuntimeError("%s %s failed: %s" % (model, action, operation["error"]))
    return operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--defaults-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-purge", action="store_true")
    args = parser.parse_args()

    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    log_path = report_path.with_suffix(".log")
    port = free_port()
    report = {"probe": "te_remote_smoke", "port": port,
              "server_root": str(root), "verdict": "failed"}
    client = Client(port)
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "h3_dit_server.py",
                 "--host", "127.0.0.1", "--port", str(port),
                 "--defaults-file", args.defaults_file,
                 "--out-dir", str(root / "outputs")],
                cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            report["created_pid"] = process.pid
            status = wait_server(client, time.monotonic() + 240, process)
            report["boot"] = status
            assert status["executor"]["state"] == "idle", status["executor"]

            _, weights = client.request("GET", "/api/weights")
            te = weights["models"]["te"]
            report["weights_before"] = te
            assert te["catalog_ready"] is True, te
            assert te["state"] == "absent", te
            assert len(te["units"]) == 2, te

            run_action(client, "te", "disk2ram", {"mode": "all"}, 900, report)
            _, weights = client.request("GET", "/api/weights/te")
            report["after_disk2ram"] = weights["model"]["state"]
            assert weights["model"]["state"] == "ram", weights["model"]

            run_action(client, "te", "ram2vram", {"mode": "all"}, 240, report)
            _, weights = client.request("GET", "/api/weights/te")
            report["after_ram2vram"] = weights["model"]
            model = weights["model"]
            units = {unit["id"]: unit for unit in model["units"]}
            # TE 主权重必须在显存；ClipProj 的 304MB 加速副本由首次 encode 惰性创建，
            # 所以此时 model.state 可以是 partial（vram + ram），不是失败。
            assert units["te.model"]["state"] == "vram", model
            assert model["state"] in ("vram", "partial"), model
            cuda_bytes = model["bytes"]["resident_cuda"]
            assert cuda_bytes > 4 * 1024 ** 3, cuda_bytes

            if not args.skip_encode:
                body = {"prompt": "a red cube on a wooden table",
                        "image_b64s": [tiny_image()],
                        "mode": "ref2va", "width": 640, "height": 960,
                        "ref_image_size": "match"}
                started = time.perf_counter()
                status, receipt = client.request("POST", "/api/infer/encode", body)
                if status != 202:
                    raise RuntimeError("encode -> %s: %s" % (status, receipt))
                operation = client.wait_operation(receipt["operation_id"], 600)
                report["encode"] = {
                    "status": operation["status"],
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "operation_id": operation["id"],
                    "error": operation["error"],
                }
                print("[smoke] encode -> %s (%.1fs)"
                      % (operation["status"], report["encode"]["elapsed_s"]),
                      flush=True)
                if operation["status"] != "done":
                    raise RuntimeError("encode failed: %s" % (operation["error"],))
                result = operation["result"]
                artifact = result["artifact"]
                report["encode"]["stats"] = result["stats"]
                report["encode"]["artifact"] = {
                    "id": artifact["id"], "filename": artifact["filename"],
                    "bytes": artifact["bytes"]}
                audit = result["audit"]["units"]["te.model"]
                assert audit["device"] == "cuda:0", audit
                report["encode"]["audit"] = audit
                _, manifest = client.request("GET", "/api/artifacts/" + artifact["id"])
                assert manifest["artifact"]["bytes"] == artifact["bytes"], manifest
                with urllib.request.urlopen(
                        client.base + "/outputs/stage/" + artifact["filename"],
                        timeout=60) as response:
                    payload = response.read()
                report["encode"]["downloaded_bytes"] = len(payload)
                assert len(payload) == artifact["bytes"]

            run_action(client, "te", "vram2ram", {"mode": "all"}, 240, report)
            _, weights = client.request("GET", "/api/weights/te")
            report["after_vram2ram"] = weights["model"]
            assert weights["model"]["state"] == "ram", weights["model"]
            assert weights["model"]["bytes"]["resident_cuda"] == 0, weights["model"]

            if not args.skip_purge:
                run_action(client, "te", "purgefromram", {"mode": "all"}, 180,
                           report)
                _, weights = client.request("GET", "/api/weights/te")
                report["after_purge"] = weights["model"]
                assert weights["model"]["state"] == "absent", weights["model"]
                assert weights["model"]["bytes"]["resident_cpu"] == 0, weights["model"]
            report["verdict"] = "ok"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report["error"] = "%s: %s" % (type(error).__name__, error)
        try:
            _, weights = client.request("GET", "/api/weights/te")
            report["actual_weights"] = weights.get("model")
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
        print(json.dumps({"verdict": report["verdict"],
                          "error": report.get("error"),
                          "report": str(report_path)}, ensure_ascii=False,
                         indent=2), flush=True)
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
