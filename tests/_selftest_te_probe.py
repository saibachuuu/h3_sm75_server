"""Synthetic TE probe contracts; CPU tensors plus fake CUDA metrics, no real GPU."""
import contextlib
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dit_svc.api_validation import ApiError
from dit_svc.operations import OperationError
from dit_svc.te_probe import run_probe, synthetic_inputs, validate_request


class FakeCuda:
    def __init__(self, free=1024 ** 3):
        self.free = free
        self.resets = 0
        self.syncs = 0
        self.owner = None
        self.queries = 0

    def mem_get_info(self, device):
        self.queries += 1
        return self.free, 2 * 1024 ** 3

    def memory_allocated(self, device):
        return 1000

    def memory_reserved(self, device):
        return 2000

    def synchronize(self, device):
        self.syncs += 1

    def reset_peak_memory_stats(self, device):
        if self.owner is not None:
            assert self.owner.session_active
        self.resets += 1

    def max_memory_allocated(self, device):
        return 3000

    def max_memory_reserved(self, device):
        return 4000


class Backend:
    def __init__(self):
        self.calls = []
        self.session_active = False
        self.runtime = types.SimpleNamespace(te_residence=types.SimpleNamespace(
            require=lambda backend: None))

    def preflight_encode(self):
        pass

    @contextlib.contextmanager
    def _encode_session(self, operation):
        self.preflight_encode()
        self.runtime.te_residence.require(self)
        self.session_active = True
        proof = {}
        try:
            yield (lambda text, refs, mode: self._encode_tokens(text, refs, mode, operation)), proof
        finally:
            self.session_active = False
            proof["unchanged"] = True

    def _stage(self, name, operation):
        assert self.session_active

    def _encode_tokens(self, text, refs, mode, operation):
        self.calls.append((len(text), [list(ref["data"].shape) for ref in refs], mode))
        return {"cond": torch.zeros((1, 12, 16))}, {
            "n_tokens": 12, "cond_shape": [1, 12, 16],
            "tokenize_s": 0.01, "encode_s": 0.02,
            "residence": {"unchanged": True}}


class ProbeTests(unittest.TestCase):
    def test_defaults_and_strict_limits(self):
        req = validate_request({"scale": {}})
        self.assertEqual(req["scale"]["text_chars"], 512)
        self.assertEqual(req["repeat"], 3)
        invalid = [{}, {"scale": []}, {"scale": {"tokens": 1024}},
                   {"scale": {"image_width": 33}}, {"scale": {"images": 0}},
                   {"scale": {"images": 10}}, {"scale": {"text_chars": 8001}},
                   {"scale": {}, "repeat": True}, {"scale": {}, "warmup": 6},
                   {"scale": {}, "repeat": 0}, {"scale": {}, "seed": -1},
                   {"scale": {}, "unknown": 1}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ApiError):
                validate_request(payload)

    def test_generated_inputs_exact_shape_deterministic_and_no_global_rng_change(self):
        scale = {"text_chars": 59, "images": 2, "image_width": 64, "image_height": 96}
        before = torch.get_rng_state().clone()
        text, refs = synthetic_inputs(scale, 7)
        text2, refs2 = synthetic_inputs(scale, 7)
        self.assertEqual(len(text), 59)
        self.assertEqual(text, text2)
        self.assertEqual(list(refs[0]["data"].shape), [1, 96, 64, 3])
        self.assertTrue(torch.equal(refs[0]["data"], refs2[0]["data"]))
        self.assertFalse(torch.equal(refs[0]["data"], refs[1]["data"]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(synthetic_inputs(dict(scale, text_chars=1), 7)[0]), 1)

    def test_real_encoder_entry_repeated_metrics_only(self):
        backend = Backend()
        request = validate_request({"scale": {"text_chars": 200, "images": 2,
                                               "image_width": 64, "image_height": 96},
                                    "warmup": 1, "repeat": 3})
        cuda = FakeCuda()
        cuda.owner = backend
        with patch.object(torch, "cuda", cuda):
            result = run_probe(backend, request)
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(backend.calls[0], (200, [[1, 96, 64, 3]] * 2, "ref2va"))
        self.assertEqual(len(result["warmup"]), 1)
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(result["runs"][0]["workspace_peak_bytes"], 2000)
        self.assertEqual(result["summary"]["encode_mean_s"], 0.02)
        self.assertNotIn("artifact", result)
        self.assertNotIn("cond", result)
        self.assertEqual(result["actual_input"]["reference_hw"], [[96, 64]] * 2)
        self.assertIn("不保证", result["token_count_note"])
        self.assertEqual(cuda.resets, 4)
        self.assertTrue(result["residence"]["unchanged"])
        self.assertFalse(backend.session_active)
        json.dumps(result)

    def test_insufficient_memory_refuses_before_inputs_or_encoder(self):
        backend = Backend()
        request = validate_request({"scale": {}})
        with patch.object(torch, "cuda", FakeCuda(free=0)):
            with self.assertRaises(OperationError) as error:
                run_probe(backend, request)
        self.assertEqual(error.exception.code, "INSUFFICIENT_VRAM")
        self.assertEqual(backend.calls, [])

    def test_missing_te_is_not_implicitly_loaded(self):
        backend = Backend()
        def missing():
            raise OperationError("MISSING_UNITS", "TE absent", 409, False)
        backend.preflight_encode = missing
        cuda = FakeCuda()
        with patch.object(torch, "cuda", cuda), self.assertRaises(OperationError) as error:
            run_probe(backend, validate_request({"scale": {}}))
        self.assertEqual(error.exception.code, "MISSING_UNITS")
        self.assertEqual(backend.calls, [])
        self.assertEqual(cuda.queries, 0)

    def test_nonfinite_output_cannot_be_reported_as_success(self):
        backend = Backend()
        normal = backend._encode_tokens
        def nonfinite(*args):
            out, stats = normal(*args)
            out["cond"][0, 0, 0] = float("nan")
            return out, stats
        backend._encode_tokens = nonfinite
        with patch.object(torch, "cuda", FakeCuda()):
            with self.assertRaises(OperationError) as error:
                run_probe(backend, validate_request({"scale": {}, "warmup": 0}))
        self.assertEqual(error.exception.code, "NONFINITE_COND")
        self.assertEqual(len(backend.calls), 1)

    def test_oom_stops_without_reducing_scale_or_retrying(self):
        backend = Backend()
        def oom(*args):
            backend.calls.append(args)
            raise OperationError("CUDA_OOM", "workspace", 409, True)
        backend._encode_tokens = oom
        request = validate_request({"scale": {}, "warmup": 0, "repeat": 5})
        with patch.object(torch, "cuda", FakeCuda()):
            with self.assertRaises(OperationError) as error:
                run_probe(backend, request)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(error.exception.details["runs_completed"], 0)
        self.assertEqual(error.exception.details["failed_iteration"], 1)
        self.assertEqual(error.exception.details["scale"], request["scale"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
