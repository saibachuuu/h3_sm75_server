"""权重 API 的 HTTP 契约自测（不构造模型、不读权重体）。

覆盖：
* 五类模型都在 GET /api/weights 里，未配置的是 absent + addressable=false；
* POST /api/weights/<model>/<action> 统一返回 202 operation 回执；
* dry_run 走同一 worker，返回 estimate 且不改状态；
* 未知 model/unit/动作、URL 与 body.model 冲突都是 4xx；
* infer.encode 在 TE 未上卡时返回 409 + 有序 suggested。
"""
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h3_dit_server as entry
from dit_svc.http_api import make_handler
from dit_svc.job_queue import QueueManager
from dit_svc.runtime import Runtime


def main():
    with tempfile.TemporaryDirectory() as directory:
        te_path = os.path.join(directory, "te.safetensors")
        proj_path = os.path.join(directory, "proj.safetensors")
        for path in (te_path, proj_path):
            with open(path, "wb") as handle:
                handle.write(b"placeholder")
        defaults = os.path.join(directory, "defaults.json")
        with open(defaults, "w", encoding="utf-8") as handle:
            json.dump({"models": {"te": {"path": te_path, "kind": "4b",
                                         "proj": proj_path}}}, handle)

        args = entry.parse_args(["--out-dir", directory,
                                 "--defaults-file", defaults])
        runtime = Runtime(args, None)
        runtime.env_ready = True
        runtime.weights.prepare()
        ctx = {"runtime": runtime, "cond_url": args.cond_url,
               "out_dir": directory, "entry_args": args}
        queue = QueueManager(runtime, ctx)
        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                     make_handler(runtime, queue, ctx))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=10)
            try:
                data = json.dumps(body) if body is not None else None
                connection.request(method, path, body=data,
                                   headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        def wait_operation(operation_id, timeout=10):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                status, body = request("GET", "/api/operations/" + operation_id)
                assert status == 200, (status, body)
                if body["operation"]["status"] in ("done", "failed", "cancelled"):
                    return body["operation"]
                time.sleep(0.02)
            raise AssertionError("operation did not finish")

        try:
            with patch("dit_svc.cond_client.health", return_value=None):
                status, body = request("GET", "/api/status")
            assert status == 200 and body["env_ready"], body
            models = body["weights"]
            assert set(models) == {"dit", "vae.video", "vae.audio", "te",
                                   "veda.bundle"}, models
            assert models["te"]["catalog_ready"] is True, models["te"]
            assert models["te"]["state"] == "absent", models["te"]
            assert models["dit"]["addressable"] is False, models["dit"]

            status, body = request("POST", "/api/weights/te/disk2ram",
                                   {"dry_run": True})
            assert status == 202 and body["ok"], (status, body)
            operation = wait_operation(body["operation_id"])
            assert operation["status"] == "done", operation
            estimate = operation["result"]["estimate"]
            assert set(estimate["would_change"]) == {"te.model", "te.proj"}, estimate
            assert estimate["noop"] is False, estimate
            status, body = request("GET", "/api/weights/te")
            assert status == 200 and body["model"]["state"] == "absent", body

            status, body = request("POST", "/api/weights/te/purgefromram",
                                   {"dry_run": True})
            assert status == 202, (status, body)
            operation = wait_operation(body["operation_id"])
            assert operation["result"]["estimate"]["noop"] is True, operation

            status, body = request("POST", "/api/weights/te/disk2ram",
                                   {"mode": "units", "units": ["te.proj"]})
            assert status == 202, (status, body)
            operation = wait_operation(body["operation_id"])
            assert operation["status"] == "failed", operation
            assert operation["error"]["code"] == "SHARED_UNIT_REQUIRED", operation

            status, body = request("POST", "/api/weights/te/disk2ram",
                                   {"mode": "units", "units": ["te.blocks.99"]})
            assert status == 400, (status, body)
            assert body["error"]["code"] == "INVALID_SELECTOR", body

            status, body = request("POST", "/api/weights/dit/ram2vram", {})
            assert status == 409, (status, body)
            assert body["error"]["code"] == "DIT_NOT_CONFIGURED", body

            status, body = request("POST", "/api/weights/unknown/disk2ram", {})
            assert status == 404 and body["error"]["code"] == "MODEL_NOT_FOUND", body

            status, body = request("POST", "/api/weights/te/disk2ram",
                                   {"model": "dit"})
            assert status == 400, (status, body)

            status, body = request("POST", "/api/weights/te/nope", {})
            assert status == 400, (status, body)

            status, body = request("POST", "/api/infer/encode", {
                "prompt": "a cat", "image_b64s": ["AAAA"]})
            assert status == 409, (status, body)
            assert body["error"]["code"] == "MISSING_UNITS", body
            suggested = body["error"]["details"]["suggested"]
            assert [item["action"] for item in suggested][0] == "disk2ram", suggested

            status, body = request("POST", "/api/infer/encode", {"prompt": "a cat"})
            assert status == 400, (status, body)

            # Legacy TE operations may not race the worker through HTTP threads.
            entered, release = threading.Event(), threading.Event()
            legacy_reply = []
            def legacy_probe(payload, operation):
                assert threading.current_thread().name == "h3-worker"
                entered.set()
                assert release.wait(5)
                return {"ok": True, "cycles": payload["cycles"]}
            runtime.run_te_probe = legacy_probe
            caller = threading.Thread(target=lambda: legacy_reply.append(request(
                "POST", "/api/te/probe", {"cycles": 1})))
            caller.start()
            try:
                assert entered.wait(5)
                status, body = request("POST", "/api/weights/te/disk2ram",
                                       {"dry_run": True})
                assert status == 409, (status, body)
            finally:
                release.set()
                caller.join(5)
            assert legacy_reply == [(200, {"ok": True, "cycles": 1})], legacy_reply
            runtime.run_te_legacy = lambda payload, op: {"ok": True, "action": "reset"}
            status, body = request("POST", "/api/te/step", {"action": "reset"})
            assert status == 200 and body["action"] == "reset", (status, body)
        finally:
            queue.stop(5)
            server.shutdown()
            server.server_close()
            thread.join(5)
    print("ALL OK: unified weights API across five models; no torch/GPU")


if __name__ == "__main__":
    main()
