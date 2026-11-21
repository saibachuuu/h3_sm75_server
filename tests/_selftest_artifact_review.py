"""Primary-review regressions: disk reserve and full publication transaction."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dit_svc.artifacts import ArtifactStore, ArtifactError


class ArtifactReviewTests(unittest.TestCase):
    def test_actual_write_cannot_consume_disk_reserve(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root, quota_bytes=10000, disk_reserve_bytes=2000)
            calls = []

            def free_bytes():
                calls.append(True)
                return 4000 if len(calls) == 1 else 1000

            with patch.object(store, "free_bytes", side_effect=free_bytes):
                with self.assertRaises(ArtifactError) as caught:
                    store.commit_stream("cond", "oversized.safetensors",
                                        lambda path: Path(path).write_bytes(b"x" * 2000),
                                        expected_bytes=None)
            self.assertEqual(caught.exception.code, "RESOURCE_EXHAUSTED")
            self.assertFalse((Path(root) / "stage" / "oversized.safetensors").exists())
            self.assertFalse(list((Path(root) / "stage").glob(".tmp_*")))

    def test_second_writer_waits_for_manifest_not_only_data_publish(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root, quota_bytes=10000, disk_reserve_bytes=0)
            hashing, release, second_started = (threading.Event() for _ in range(3))
            first_errors, second_errors = [], []

            def hash_file(path):
                if Path(path).name == "first.safetensors":
                    hashing.set()
                    if not release.wait(5):
                        raise RuntimeError("test hash timeout")
                return "0" * 64

            def first():
                try:
                    store.commit("cond", "first.safetensors", b"one")
                except Exception as exc:
                    first_errors.append(exc)

            def second():
                try:
                    store.commit_stream("cond", "second.safetensors",
                                        lambda path: (second_started.set(),
                                                      Path(path).write_bytes(b"two")),
                                        expected_bytes=3)
                except Exception as exc:
                    second_errors.append(exc)

            with patch("dit_svc.artifacts._sha256_file", side_effect=hash_file):
                t1, t2 = threading.Thread(target=first), threading.Thread(target=second)
                try:
                    t1.start()
                    self.assertTrue(hashing.wait(5))
                    t2.start()
                    self.assertFalse(second_started.wait(0.1))
                    self.assertEqual(store.list(), [])  # GET does not block on write lock
                    release.set()
                    t1.join(5)
                    t2.join(5)
                    self.assertFalse(t1.is_alive() or t2.is_alive())
                    self.assertFalse(first_errors or second_errors)
                    self.assertEqual(len(store.list()), 2)
                    self.assertEqual(store._pending, {})
                finally:
                    release.set()
                    t1.join(5)
                    if t2.ident is not None:
                        t2.join(5)


if __name__ == "__main__":
    unittest.main()
