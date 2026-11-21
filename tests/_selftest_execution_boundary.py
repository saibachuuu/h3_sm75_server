"""Primary review regressions for cross-module errors and legacy GPU lifetime."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dit_svc.job_queue import QueueManager
from dit_svc.operations import OperationError, detached_error


class SubsystemError(Exception):
    code = "INVALID_ARTIFACT"
    http_status = 422
    retryable = False
    details = {"field": "audio"}


class BoundaryTests(unittest.TestCase):
    def test_subsystem_protocol_error_survives_worker(self):
        clean = detached_error(SubsystemError("invalid audio"))
        self.assertEqual(clean.code, "INVALID_ARTIFACT")
        self.assertEqual(clean.http_status, 422)
        self.assertEqual(clean.details, {"field": "audio"})
        self.assertIsNone(clean.__traceback__)

    def test_legacy_job_retains_reservation_until_sync(self):
        entered, release = threading.Event(), threading.Event()
        runtime = SimpleNamespace(error=None, ready=True, env_ready=True)

        def synchronize(job):
            self.assertEqual(job.status, "running")
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test synchronization timeout")

        runtime.job_synchronize = synchronize
        queue = QueueManager(runtime, {})
        payload = {"prompt": "test", "image_b64s": ["aGVsbG8="], "width": 640,
                   "height": 960, "length": 124, "steps": 4, "seed": 1}
        try:
            with patch("dit_svc.pipeline.run_job", return_value=None):
                job = queue.submit(payload)
                self.assertTrue(entered.wait(5))
                self.assertEqual(queue.executor_state()["state"], "running")
                with self.assertRaises(OperationError) as caught:
                    queue.submit_operation("control", {}, lambda op: {})
                self.assertEqual(caught.exception.code, "BUSY")
                release.set()
                self.assertTrue(queue.stop(5))
                self.assertEqual(job.status, "done")
        finally:
            release.set()
            queue.stop(5)


if __name__ == "__main__":
    unittest.main()
