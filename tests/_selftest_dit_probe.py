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
from dit_svc.dit_probe import (DEFAULT_COND_DIM, cond_dim, run_probe,
                               synthetic_cond, validate_request)
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


def make_runtime(nonfinite=False, hidden_size=None):
    nodes = _FakeNodes(nonfinite=nonfinite)
    dit_backend = SimpleNamespace(_stage=lambda message, op=None: None)
    diff_model = SimpleNamespace()
    if hidden_size is not None:
        diff_model.hidden_size = hidden_size
    snapshots = [{"state": "vram", "counts": {"vram": 56},
                  "bytes": {"resident_cuda": 1},
                  "units": [{"device": "cuda:0"}]}]

    def weights_snapshot():
        return {"models": {"dit": snapshots[0]}}

    runtime = SimpleNamespace(
        rt=SimpleNamespace(unpack=lambda value: value),
        diff_model=diff_model,
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
        cond, tags = synthetic_cond(torch, {"cond_tokens": 24}, 7, 5376)
        cond2, tags2 = synthetic_cond(torch, {"cond_tokens": 24}, 7, 5376)
        self.assertEqual(list(cond.shape), [1, 24, 5376])
        self.assertTrue(torch.equal(cond, cond2))
        self.assertEqual(int(tags.sum()), 24)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cond_dim_prefers_model_hidden_size(self):
        runtime, _ = make_runtime(hidden_size=5376)
        self.assertEqual(cond_dim(runtime), 5376)
        runtime, _ = make_runtime()          # 未暴露 hidden_size 时用默认值
        self.assertEqual(cond_dim(runtime), DEFAULT_COND_DIM)

    def test_probe_does_not_touch_token_refiner(self):
        """body-only：cond 直接以 body 宽度喂入，不需要 preprocess_text_embeds。"""
        runtime, nodes = make_runtime(hidden_size=5376)
        called = {"refiner": 0}

        def explode(cond):
            called["refiner"] += 1
            raise AssertionError("probe 不得调用 token_refiner 路径")

        runtime.diff_model.preprocess_text_embeds = explode
        request = validate_request({"scale": {"cond_tokens": 16},
                                    "warmup": 0, "repeat": 1})
        with _FakeComfyExtrasContext(nodes), patch.object(torch, "cuda", FakeCuda()):
            result = run_probe(runtime, request)
        self.assertEqual(called["refiner"], 0)
        self.assertTrue(result["body_only"])
        self.assertEqual(result["cond_dim"], 5376)
        self.assertIn("skipped", result["token_refiner"])


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
        # 采样后快照变化 -> WEIGHT_MUTATED
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
    def _runtime(self):
        from dit_svc.runtime import Runtime
        runtime = Runtime(SimpleNamespace(attn="none"), None)
        runtime.env_ready = True
        runtime.device_healthy = True
        return runtime

    @staticmethod
    def _snap(dit_units, catalog_ready=True):
        return {"models": {
            "dit": {"state": "partial", "catalog_ready": catalog_ready,
                    "units": [{"id": uid, "state": state}
                              for uid, state in dit_units],
                    "counts": {}, "bytes": {}},
            "te": {"state": "absent"},
            "vae.video": {"state": "absent"},
            "vae.audio": {"state": "absent"},
            "veda.bundle": {"state": "absent"}}}

    def test_dit_probe_allows_refiner_absent(self):
        runtime = self._runtime()
        # refiner 缺席，其余全在显存 -> 通过
        snap = self._snap([("dit.blocks.0", "vram"),
                           ("dit.blocks.1", "vram"),
                           ("dit.refiner", "absent")])
        runtime.weights_snapshot = lambda: snap
        runtime.operation_admission("infer.dit_probe", {})
        self.assertTrue(runtime.capabilities()["dit_probe"]["ready"])

    def test_dit_probe_rejects_missing_body_unit(self):
        runtime = self._runtime()
        snap = self._snap([("dit.blocks.0", "vram"),
                           ("dit.blocks.1", "ram"),
                           ("dit.refiner", "absent")])
        runtime.weights_snapshot = lambda: snap
        with self.assertRaises(OperationError) as ctx:
            runtime.operation_admission("infer.dit_probe", {})
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertEqual(ctx.exception.details["missing"], ["dit.blocks.1"])
        caps = runtime.capabilities()["dit_probe"]
        self.assertFalse(caps["ready"])
        self.assertEqual(caps["missing"], ["dit.blocks.1"])

    def test_dit_probe_requires_catalog(self):
        runtime = self._runtime()
        runtime.weights_snapshot = lambda: self._snap([], catalog_ready=False)
        with self.assertRaises(OperationError) as ctx:
            runtime.operation_admission("infer.dit_probe", {})
        self.assertEqual(ctx.exception.details["missing"], ["dit"])

    def test_sample_still_requires_full_vram(self):
        """真实采样依旧要求 dit 全量 vram（refiner 不可缺席）。"""
        runtime = self._runtime()
        snap = self._snap([("dit.blocks.0", "vram"),
                           ("dit.refiner", "absent")])
        snap["models"]["dit"]["state"] = "partial"
        snap["models"]["vae.video"] = {"state": "ram"}
        snap["models"]["vae.audio"] = {"state": "ram"}
        runtime.weights_snapshot = lambda: snap
        job = SimpleNamespace(kind="video")
        with self.assertRaises(OperationError) as ctx:
            runtime.job_admission(job)
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertIn("dit", ctx.exception.details["missing"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
