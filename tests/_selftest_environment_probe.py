"""Stdlib tests for the read-only environment probe."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


PATH = Path(__file__).parent / "probes" / "_probe_environment.py"
SPEC = importlib.util.spec_from_file_location("environment_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class EnvironmentProbeTests(unittest.TestCase):
    def test_header_only_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            header = json.dumps({"__metadata__": {"test": "yes"},
                                 "weight": {"dtype": "F32", "shape": [2],
                                            "data_offsets": [0, 8]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
            report = probe.safetensors_header(path)
            self.assertEqual(report["metadata"], {"test": "yes"})
            self.assertEqual(report["tensors"]["weight"]["shape"], [2])
            self.assertNotIn("torch", sys.modules)

    def test_invalid_header_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", 1 << 50))
            with self.assertRaises(ValueError):
                probe.safetensors_header(path)

    def test_collect_never_imports_cuda_or_constructs_models(self):
        report = probe.collect()
        self.assertEqual(report["models_constructed"], 0)
        self.assertFalse(report["cuda_imported_by_probe"])
        self.assertNotIn("torch", sys.modules)
        self.assertIn("torch", report["packages"])


if __name__ == "__main__":
    unittest.main()
