"""Local real-HTTP artifact query/download, ranges, traversal and read leases."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dit_svc.artifacts import ArtifactStore, ArtifactError
from dit_svc.http_api import make_handler, byte_range
from dit_svc.job_queue import QueueManager
from dit_svc.runtime import Runtime


class ArtifactHttpTests(unittest.TestCase):
    def test_byte_ranges(self):
        self.assertEqual(byte_range("bytes=1-3", 6), (1, 3, 206))
        self.assertEqual(byte_range("bytes=-2", 6), (4, 5, 206))
        self.assertEqual(byte_range("bytes=4-", 6), (4, 5, 206))
        for invalid in ("bytes=-0", "bytes=6-", "bytes=-1-3", "bytes=0-1,3-4"):
            with self.assertRaises(ValueError):
                byte_range(invalid, 6)

    def test_download_lease_protects_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root, disk_reserve_bytes=0, ttl_s=0)
            artifact = store.commit("cond", "leased.safetensors", b"123")
            with store.download_lease(artifact["filename"]):
                self.assertEqual(store.refcount(artifact["id"]), 1)
                self.assertEqual(store.cleanup(), [])
                with self.assertRaises(ArtifactError):
                    store.delete(artifact["id"])
            self.assertEqual(store.refcount(artifact["id"]), 0)
            self.assertEqual(store.cleanup(), [artifact["id"]])

    def test_real_http_artifact_and_safe_stage_download(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root, disk_reserve_bytes=0)
            artifact = store.commit("cond", "clip.safetensors", b"abcdef")
            Path(root, "existing.mp4").write_bytes(b"movie")
            runtime = Runtime(SimpleNamespace(attn="none"), None)
            queue = QueueManager(runtime, {})
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
                runtime, queue, {"out_dir": root, "artifacts": store}))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def request(path, headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                try:
                    connection.request("GET", path, headers=headers or {})
                    response = connection.getresponse()
                    return response.status, dict(response.getheaders()), response.read()
                finally:
                    connection.close()

            try:
                code, _, body = request("/api/artifacts/" + artifact["id"])
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["artifact"]["bytes"], 6)
                code, _, body = request("/outputs/stage/clip.safetensors")
                self.assertEqual((code, body), (200, b"abcdef"))
                code, headers, body = request("/outputs/stage/clip.safetensors", {"Range": "bytes=-2"})
                self.assertEqual((code, body), (206, b"ef"))
                self.assertEqual(headers["Content-Range"], "bytes 4-5/6")
                code, _, _ = request("/outputs/stage/clip.safetensors", {"Range": "bytes=99-"})
                self.assertEqual(code, 416)
                self.assertEqual(request("/outputs/existing.mp4")[2], b"movie")
                for path in ("/outputs/stage/../existing.mp4", "/outputs/stage/%2e%2e%2fexisting.mp4",
                             "/outputs/foo/existing.mp4", "/outputs/stage/C:/secrets"):
                    self.assertEqual(request(path)[0], 400)
                self.assertEqual(request("/api/artifacts/missing")[0], 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(3)
                queue.stop(3)


if __name__ == "__main__":
    unittest.main()
