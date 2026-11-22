"""Synthetic DiT probe contracts; CPU tensors + fake sampler chain, no real GPU.

Covers the dit_probe validator, synthetic cond determinism, the full fake
SamplerCustomAdvanced run (warmup/repeat metrics only, no artifacts), memory
gate, non-finite latent rejection, weight-mutation audit, and Runtime
admission/capabilities for infer.dit_probe.
"""
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # CPU build in the isolated test env

from dit_svc.api_validation import ApiError
from dit_svc.dit_probe import (COND_DIM, run_probe, synthetic_cond,
                               validate_request)
from dit_svc.operations import OperationError


class FakeCuda:
    def __init__(self, free=1024 ** 3):
        self.free = free
        self.resets = 0
        self.syncs = 0
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
        self.resets += 1

    def max_memory_allocated(self, device):
        return 3000

    def max_memory_reserved(self, device):
        return 4000


class _FakeNodes:
    """comfy_extras fake: records sampler calls, optional non-finite latents."""

    def __init__(self, nonfinite=False):
        self.nonfinite = nonfinite
        self.sampler_calls = 0
        self.empty_latent_calls = []
        self.guider_calls = 0
        self.modules = self._build()

    def _build(self):
        fake = types.ModuleType("comfy_extras.nodes_custom_sampler")
        nodes = self

        class BasicScheduler:
            @staticmethod
            def execute(model, scheduler, steps, denom):
                return ([torch.tensor([1.0, 0.0])],)

        class KSamplerSelect:
            @staticmethod
            def execute(name):
                return (("sampler_obj",),)

        class RandomNoise:
            @staticmethod
            def execute(seed):
                return (("noise",),)

        class BasicGuider:
            @staticmethod
            def execute(model, positive):
                nodes.guider_calls += 1
                return (("guider",),)

        class SamplerCustomAdvanced:
            @staticmethod
            def execute(noise, guider, sampler, sigmas, latent):
                nodes.sampler_calls += 1
                video = torch.zeros(2, 4, 6, 8)
                if nodes.nonfinite:
                    video = video.clone()
                    video[0, 0, 0, 0] = float("nan")
                out = {"samples": SimpleNamespace(unbind=lambda: (video,))}
                return (out,)

        for cls in (BasicScheduler, KSamplerSelect, RandomNoise, BasicGuider,
                    SamplerCustomAdvanced):
            setattr(fake, cls.__name__, cls)

        minimax = types.ModuleType("comfy_extras.nodes_minimax_h3")

        def _empty_av_latent(width, height, length):
            nodes.empty_latent_calls.append((width, height, length))
            return {"samples": "latent"}, 16

        minimax._empty_av_latent = _empty_av_latent
        self.minimax = minimax
        return fake


class _FakeComfyExtrasContext:
    def __init__(self, nodes):
        self.nodes = nodes
        self._saved = {}

    def __enter__(self):
        for name, module in (("comfy_extras", types.ModuleType("comfy_extras")),
                             ("comfy_extras.nodes_custom_sampler", self.nodes.modules),
                             ("comfy_extras.nodes_minimax_h3", self.nodes.minimax)):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self

    def __exit__(self, *exc):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        return False


def make_runtime(nonfinite=False, mutate_after=None):
    nodes = _FakeNodes(nonfinite=nonfinite)
    dit_backend = SimpleNamespace(_stage=lambda message, op=None: None)
    snapshots = [{"state": "vram", "counts": {"vram": 56},
                  "bytes": {"resident_cuda": 1},
                  "units": [{"device": "cuda:0"}]}]
    if mutate_after is not None:
        snapshots.append({"state": "vram", "counts": {"vram": 55, "ram": 1},
                         "bytes": {"resident_cuda": 1},
                         "units": [{"device": "cpu"}]})
    calls = {"n": 0}

    def weights_snapshot():
        if mutate_after is None:
            return {"models": {"dit": snapshots[0]}}
        if calls["n"] < mutate_after:
            calls["n"] += 1
            return {"models": {"dit": snapshots[min(calls["n"], len(snapshots) - 1)]}}
        return {"models": {"dit": snapshots[-1]}}

    if mutate_after is not None:
        weights_snapshot = lambda: {"models": {  # noqa: E731
            "dit": snapshots[0] if calls["n"] == 0 else snapshots[-1]}}

    runtime = SimpleNamespace(
        rt=SimpleNamespace(unpack=lambda value: value),
        diff_model=SimpleNamespace(
            preprocess_text_embeds=lambda cond: cond),
        model_shifted=object(),
        device=torch.device("cpu"),
        weights=SimpleNamespace(backends={"dit": dit_backend}),
        weights_snapshot=weights_snapshot,
        veda_patch=None)
    return runtime, nodes


class ValidationTests(unittest.TestCase):
    def test_defaults_and_strict_limits(self):
        req = validate_request({"scale": {}})
        self.assertEqual(req["scale"]["cond_tokens"], 512)
        self.assertEqual(req["scale"]["width"], 640)
        self.assertEqual(req["scale"]["height"], 960)
        self.assertEqual(req["scale"]["length"], 124)
        self.assertEqual(req["steps"], 1)
        invalid = [{}, {"scale": []}, {"scale": {"tokens": 10}},
                   {"scale": {"width": 641}}, {"scale": {"length": 15}},
                   {"scale": {"cond_tokens": 15}}, {"steps": 5},
                   {"steps": True}, {"scale": {}, "repeat": 0},
                   {"scale": {}, "seed": -1}, {"scale": {}, "unknown": 1}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ApiError):
                validate_request(payload)


class SyntheticTests(unittest.TestCase):
    def test_cond_shape_deterministic_no_global_rng(self):
        before = torch.get_rng_state().clone()
        cond, tags = synthetic_cond(torch, {"cond_tokens": 24}, 7)
        cond2, tags2 = synthetic_cond(torch, {"cond_tokens": 24}, 7)
        self.assertEqual(list(cond.shape), [1, 24, COND_DIM])
        self.assertTrue(torch.equal(cond, cond2))
        self.assertEqual(int(tags.sum()), 24)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class ProbeTests(unittest.TestCase):
    def test_real_sampler_chain_metrics_only(self):
        runtime, nodes = make_runtime()
        request = validate_request({"scale": {"cond_tokens": 32, "width": 640,
                                              "height": 960, "length": 16},
                                    "warmup": 1, "repeat": 3})
        cuda = FakeCuda()
        with _FakeComfyExtrasContext(nodes), patch.object(torch, "cuda", cuda):
            result = run_probe(runtime, request)
        self.assertEqual(nodes.sampler_calls, 4)       # warmup + 3 runs
        self.assertEqual(nodes.empty_latent_calls, [(640, 960, 16)])
        self.assertEqual(len(result["warmup"]), 1)
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(result["runs"][0]["workspace_peak_bytes"], 2000)
        self.assertEqual(result["runs"][0]["steps"], 1)
        self.assertEqual(result["runs"][0]["frames"], 16)
        self.assertEqual(result["summary"]["repeat"], 3)
        self.assertAlmostEqual(result["summary"]["sample_mean_s"],
                               sum(row["sample_s"] for row in result["runs"]) / 3,
                               places=5)
        self.assertIsNone(result["veda_calls"])        # 非 veda 档
        self.assertNotIn("artifact", result)
        self.assertEqual(cuda.resets, 4)
        import json
        json.dumps(result)

    def test_insufficient_memory_refuses_before_sampling(self):
        runtime, nodes = make_runtime()
        request = validate_request({"scale": {}})
        with _FakeComfyExtrasContext(nodes), patch.object(
                torch, "cuda", FakeCuda(free=0)):
            with self.assertRaises(OperationError) as error:
                run_probe(runtime, request)
        self.assertEqual(error.exception.code, "INSUFFICIENT_VRAM")
        self.assertEqual(nodes.sampler_calls, 0)

    def test_nonfinite_latent_cannot_be_reported_as_success(self):
        runtime, nodes = make_runtime(nonfinite=True)
        request = validate_request({"scale": {}, "warmup": 0, "repeat": 2})
        with _FakeComfyExtrasContext(nodes), patch.object(torch, "cuda", FakeCuda()):
            with self.assertRaises(OperationError) as error:
                run_probe(runtime, request)
        self.assertEqual(error.exception.code, "NONFINITE_LATENT")
        self.assertEqual(nodes.sampler_calls, 1)
        self.assertEqual(error.exception.details["runs_completed"], 0)
        self.assertEqual(error.exception.details["failed_iteration"], 1)
        self.assertEqual(error.exception.details["steps"], 1)

    def test_weight_mutation_fails_closed(self):
        # 第二次 snapshot（采样后）返回不同驻留 -> WEIGHT_MUTATED
        runtime, nodes = make_runtime()
        request = validate_request({"scale": {}, "warmup": 0, "repeat": 1})
        snap = {"state": "vram", "counts": {"vram": 56},
                "bytes": {"resident_cuda": 1}, "units": [{"device": "cuda:0"}]}
        mutated = {"state": "partial", "counts": {"vram": 55, "ram": 1},
                   "bytes": {"resident_cuda": 1}, "units": [{"device": "cpu"}]}
        state = {"flip": False}
        runtime.weights_snapshot = lambda: {"models": {
            "dit": mutated if state["flip"] else snap}}

        with _FakeComfyExtrasContext(nodes):
            with patch.object(torch, "cuda", FakeCuda()):
                # 通过 monkeypatch 在采样后翻转 snapshot：包住 sampler 调用
                orig_exec = nodes.modules.SamplerCustomAdvanced.execute
                def flip_exec(*args, **kwargs):
                    out = orig_exec(*args, **kwargs)
                    state["flip"] = True
                    return out
                nodes.modules.SamplerCustomAdvanced.execute = staticmethod(flip_exec)
                with self.assertRaises(OperationError) as error:
                    run_probe(runtime, request)
        self.assertEqual(error.exception.code, "WEIGHT_MUTATED")

    def test_seed_and_steps_flow_into_chain(self):
        runtime, nodes = make_runtime()
        request = validate_request({"scale": {"length": 32}, "steps": 3,
                                   "warmup": 0, "repeat": 1, "seed": 99})
        with _FakeComfyExtrasContext(nodes), patch.object(torch, "cuda", FakeCuda()):
            run_probe(runtime, request)
        self.assertEqual(nodes.empty_latent_calls, [(640, 960, 32)])


class AdmissionTests(unittest.TestCase):
    def test_dit_probe_requires_full_vram(self):
        from dit_svc.runtime import Runtime
        runtime = Runtime(SimpleNamespace(attn="none"), None)
        runtime.env_ready = True
        runtime.device_healthy = True
        with self.assertRaises(OperationError) as ctx:
            runtime.operation_admission("infer.dit_probe", {})
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")

        def full_snap(dit_state):
            return {"models": {
                name: {"state": dit_state if name == "dit" else "absent"}
                for name in ("dit", "te", "vae.video", "vae.audio",
                             "veda.bundle")}}

        runtime.weights_snapshot = lambda: full_snap("vram")
        runtime.operation_admission("infer.dit_probe", {})
        caps = runtime.capabilities()
        self.assertTrue(caps["dit_probe"]["ready"])
        self.assertEqual(caps["dit_probe"]["missing"], [])
        runtime.weights_snapshot = lambda: full_snap("partial")
        self.assertFalse(runtime.capabilities()["dit_probe"]["ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
