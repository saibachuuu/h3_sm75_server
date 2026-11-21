"""HTTP smoke with an empty Runtime and the single real QueueManager worker."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
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
        args = entry.parse_args(["--out-dir", directory])
        runtime = Runtime(args, None)
        runtime.env_ready = True
        ctx = {"runtime": runtime, "cond_url": args.cond_url, "out_dir": directory,
               "entry_args": args}
        queue = QueueManager(runtime, ctx)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, queue, ctx))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                data = json.dumps(body) if body is not None else None
                connection.request(method, path, body=data,
                                   headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        try:
            with patch("dit_svc.cond_client.health", return_value=None):
                status, body = request("GET", "/api/status")
            assert status == 200 and body["env_ready"] and not body["ready"], body
            assert body["executor"]["state"] == "idle", body
            status, body = request("GET", "/api/weights")
            assert status == 200 and len(body["models"]) == 5, body
            assert all(model["state"] == "absent" for model in body["models"].values())
            status, body = request("GET", "/api/weights/dit")
            assert status == 200 and not body["model"]["catalog_ready"], body
            status, body = request("GET", "/api/weights/unknown")
            assert status == 404 and body["error"]["code"] == "MODEL_NOT_FOUND", body
            status, body = request("GET", "/api/operations?limit=-1")
            assert status == 400, body
            status, body = request("GET", "/api/operations/missing")
            assert status == 404 and body["error"]["code"] == "OPERATION_NOT_FOUND", body
            finished = threading.Event()

            def callback(op):
                op.set_stage("test")
                finished.set()
                return {"value": 7}

            op = queue.submit_operation("test", {}, callback)
            assert finished.wait(5)
            # stop waits for callback finalization and avoids polling/background timers.
            assert queue.stop(5)
            status, body = request("GET", "/api/operations/" + op.id)
            assert status == 200 and body["operation"]["status"] == "done", body
            assert body["operation"]["result"] == {"value": 7}, body
            status, body = request("GET", "/api/operations")
            assert status == 200 and body["ok"] and body["counts"]["done"] == 1, body
            status, body = request("POST", "/api/operations/" + op.id + "/cancel", {})
            assert status == 409 and body["error"]["code"] == "OPERATION_NOT_CANCELLABLE", body
        finally:
            queue.stop(5)
            server.shutdown()
            server.server_close()
            thread.join(5)
    print("ALL OK: empty status/weights and operation HTTP views; no torch/GPU")


if __name__ == "__main__":
    main()
