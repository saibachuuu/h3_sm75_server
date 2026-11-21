"""TE 显式权重管理器自测（stdlib + 可控 fake，不读真实权重）。

覆盖：
* recipe 缺失/catalog 登记/共享 unit 拒绝；
* dry_run 不构造、不改状态；
* disk2ram → ram2vram → vram2ram → purgefromram 的完整状态迁移与 purge 门卫；
* encode 的 resident 前置条件与 artifact 落盘；
* 真实 torch 模型的 storage 账（物理字节非 numel 估算）。
"""
import contextlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dit_svc import weights as W                      # noqa: E402
from dit_svc.operations import OperationError         # noqa: E402
from dit_svc.te_manager import (                      # noqa: E402
    MODEL_ID, TeManager, UNIT_MODEL, UNIT_PROJ)


class FakeArgs:
    def __init__(self, recipes=None):
        self.model_recipes = recipes or {}


class FakeStore:
    def __init__(self):
        self.commits = []

    def commit(self, kind, filename, data, *, metadata=None, fmt=None):
        self.commits.append({"kind": kind, "filename": filename,
                             "bytes": len(data), "metadata": metadata,
                             "fmt": fmt})
        return {"id": "cond_%032x" % len(self.commits), "kind": kind,
                "filename": filename, "bytes": len(data), "format": fmt,
                "metadata": metadata}


class FakeResidence:
    @contextlib.contextmanager
    def inference(self, backend, operation=None):
        yield {"unchanged": True}

    def unregister(self):
        pass


class FakeRuntime:
    def __init__(self, recipes=None):
        self.args = FakeArgs(recipes)
        self.stage = None
        self.artifacts = FakeStore()
        self.device_healthy = True
        self.te_residence = FakeResidence()

    def set_stage(self, name):
        self.stage = name


class FakePatcher:
    def __init__(self, model):
        self.model = model


class FakeClip:
    """足够让 _inventory/_dominant_device 正常工作的最小 CLIP。"""

    def __init__(self, model=None):
        import torch
        self.patcher = FakePatcher(model if model is not None else torch.nn.Linear(4, 4))

    def tokenize(self, prompt, **kwargs):
        return {"fake": prompt, "kwargs": sorted(kwargs)}

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        cond = torch.zeros((1, 3, 8), dtype=torch.float32)
        tags = torch.arange(3, dtype=torch.long)
        return [[cond, {"minimax_token_tags": tags}]]


def inventory(cpu=0, cuda=0, device="cpu", proj_cpu=0, proj_cuda=0):
    return {"units": {
                UNIT_MODEL: {"device": device, "cpu": cpu, "cuda": cuda},
                UNIT_PROJ: {"device": "cuda:0" if proj_cuda else "cpu",
                            "cpu": proj_cpu, "cuda": proj_cuda},
            },
            "uninspectable": [],
            "storage_digest": (["cpu:1:%d" % (cpu + cuda)]
                               if cpu + cuda else [])}


class TeManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.te_path = os.path.join(self.tmp.name, "te.safetensors")
        self.proj_path = os.path.join(self.tmp.name, "proj.safetensors")
        with open(self.te_path, "wb") as f:
            f.write(b"te-placeholder")
        with open(self.proj_path, "wb") as f:
            f.write(b"proj-placeholder")
        self.recipes = {"te": {"path": self.te_path, "kind": "4b",
                               "proj": self.proj_path}}

    def tearDown(self):
        self.tmp.cleanup()

    def _manager(self, recipes=None):
        runtime = FakeRuntime(recipes if recipes is not None else self.recipes)
        registry = W.WeightRegistry()
        return TeManager(runtime, registry), runtime, registry

    # -------------------------------------------------- recipe / catalog
    def test_prepare_without_config_is_not_configured(self):
        manager, _, _ = self._manager({})
        self.assertFalse(manager.prepare())
        self.assertFalse(manager.configured())
        with self.assertRaises(OperationError) as ctx:
            manager.run_action("disk2ram", {"mode": "all", "units": []})
        self.assertEqual(ctx.exception.code, "TE_NOT_CONFIGURED")
        self.assertEqual(ctx.exception.http_status, 409)

    def test_prepare_registers_model_and_proj_units(self):
        manager, _, registry = self._manager()
        self.assertTrue(manager.prepare())
        self.assertEqual(manager.catalog.unit_ids(), [UNIT_MODEL, UNIT_PROJ])
        state = registry.model_state(MODEL_ID)
        self.assertEqual(state["state"], "absent")
        self.assertEqual(state["counts"]["absent"], 2)
        self.assertEqual(state["recipe"]["kind"], "4b")

    def test_prepare_is_idempotent(self):
        manager, _, registry = self._manager()
        manager.prepare()
        generation = registry.model_state(MODEL_ID)["generation"]
        manager.prepare()
        self.assertEqual(registry.model_state(MODEL_ID)["generation"], generation)

    def test_proj_cannot_be_selected_alone(self):
        manager, _, _ = self._manager()
        manager.prepare()
        with self.assertRaises(OperationError) as ctx:
            manager.run_action("disk2ram", {"mode": "units",
                                            "units": [UNIT_PROJ]})
        self.assertEqual(ctx.exception.code, "SHARED_UNIT_REQUIRED")

    def test_model_without_proj_is_rejected_when_configured(self):
        manager, _, _ = self._manager()
        manager.prepare()
        with self.assertRaises(OperationError) as ctx:
            manager.run_action("disk2ram", {"mode": "units",
                                            "units": [UNIT_MODEL]})
        self.assertEqual(ctx.exception.code, "SHARED_UNIT_REQUIRED")
        self.assertEqual(ctx.exception.details["missing"], [UNIT_PROJ])

    # -------------------------------------------------- dry_run
    def test_dry_run_does_not_construct_or_change_state(self):
        manager, _, registry = self._manager()
        manager.prepare()
        calls = []
        manager.construct_cpu = lambda operation=None: calls.append(1)
        result = manager.run_action("disk2ram", {"mode": "all", "units": [],
                                                 "dry_run": True})
        self.assertEqual(calls, [])
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["estimate"]["would_change"],
                         [UNIT_MODEL, UNIT_PROJ])
        self.assertEqual(registry.model_state(MODEL_ID)["state"], "absent")

    # -------------------------------------------------- lifecycle
    def _install_fake_model(self, manager, inventory_value):
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        manager.inventory = lambda: dict(inventory_value)

    def _attach(self, manager):
        manager._clip = FakeClip()
        manager.pc = manager._clip
        manager._projected = False

    def test_full_lifecycle_and_purge_gate(self):
        manager, _, registry = self._manager()
        manager.prepare()
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        manager.inventory = lambda: dict(inventory(cpu=4096, proj_cpu=512))

        summary = manager.run_action("disk2ram", {"mode": "all", "units": []})
        self.assertEqual(sorted(summary["completed"]), [UNIT_MODEL, UNIT_PROJ])
        state = registry.model_state(MODEL_ID)
        self.assertEqual(state["state"], "ram")
        self.assertEqual(state["bytes"]["resident_cpu"], 4608)

        manager.move_to = lambda device: None
        manager.inventory = lambda: dict(inventory(cpu=0, cuda=4096,
                                                    device="cuda:0",
                                                    proj_cpu=512, proj_cuda=256))
        summary = manager.run_action("ram2vram", {"mode": "all", "units": []})
        state = registry.model_state(MODEL_ID)
        self.assertEqual(state["state"], "vram")
        self.assertEqual(state["bytes"]["resident_cuda"], 4352)
        self.assertEqual(summary["audit"]["units"][UNIT_MODEL]["device"],
                         "cuda:0")

        with self.assertRaises(W.TransitionError) as ctx:
            manager.run_action("purgefromram", {"mode": "all", "units": []})
        self.assertIn("CUDA", str(ctx.exception))

        manager.inventory = lambda: dict(inventory(cpu=4096, proj_cpu=512))
        manager.run_action("vram2ram", {"mode": "all", "units": []})
        self.assertEqual(registry.model_state(MODEL_ID)["state"], "ram")

        manager.inventory = lambda: dict(inventory())
        summary = manager.run_action("purgefromram", {"mode": "all",
                                                      "units": []})
        self.assertEqual(registry.model_state(MODEL_ID)["state"], "absent")
        self.assertEqual(summary["proof"]["weakrefs_alive"], [])
        self.assertIsNone(manager.pc)

    def test_purge_is_noop_when_already_absent(self):
        manager, _, registry = self._manager()
        manager.prepare()
        summary = manager.run_action("purgefromram", {"mode": "all", "units": []})
        self.assertTrue(summary.get("noop"))
        self.assertEqual(registry.model_state(MODEL_ID)["state"], "absent")

    def test_uninspectable_inventory_fails_closed(self):
        manager, _, registry = self._manager()
        manager.prepare()
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        bad = inventory(cpu=1024)
        bad["uninspectable"] = ["te.model/wrapper"]
        manager.inventory = lambda: dict(bad)
        with self.assertRaises(OperationError) as ctx:
            manager.run_action("disk2ram", {"mode": "all", "units": []})
        self.assertEqual(ctx.exception.code, "UNINSPECTABLE_WEIGHTS")
        self.assertEqual(registry.model_state(MODEL_ID)["state"], "error")

    # -------------------------------------------------- encode
    def test_encode_requires_vram(self):
        manager, _, _ = self._manager()
        manager.prepare()
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        manager.inventory = lambda: dict(inventory(cpu=1024))
        manager.run_action("disk2ram", {"mode": "all", "units": []})
        with self.assertRaises(OperationError) as ctx:
            manager.run_encode(self._encode_request())
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertEqual(ctx.exception.details["suggested"][0]["action"],
                         "ram2vram")

    def test_encode_writes_artifact_and_keeps_device(self):
        manager, runtime, registry = self._manager()
        manager.prepare()
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        manager.inventory = lambda: dict(inventory(cpu=4096, proj_cpu=512))
        manager.run_action("disk2ram", {"mode": "all", "units": []})
        manager.move_to = lambda device: None
        manager.inventory = lambda: dict(inventory(cpu=0, cuda=4096,
                                                    device="cuda:0",
                                                    proj_cpu=512, proj_cuda=256))
        manager.run_action("ram2vram", {"mode": "all", "units": []})
        manager._preprocess = lambda *args: ([], [{"type": "image",
                                                   "data": None}], [[32, 32]])

        result = manager.run_encode(self._encode_request())
        self.assertEqual(result["stats"]["n_tokens"], 3)
        self.assertEqual(len(runtime.artifacts.commits), 1)
        commit = runtime.artifacts.commits[0]
        self.assertEqual(commit["kind"], "cond")
        self.assertEqual(commit["fmt"], "h3.cond.v1")
        self.assertEqual(commit["metadata"]["format"], "h3.cond.v1")
        self.assertGreater(commit["bytes"], 0)
        self.assertEqual(manager.encoded, 1)
        self.assertEqual(registry.unit_record(MODEL_ID, UNIT_MODEL)["state"],
                         "vram")

    def test_encode_detects_weight_mutated_by_inner_loader(self):
        manager, _, registry = self._manager()
        manager.prepare()
        manager.construct_cpu = lambda operation=None: self._attach(manager)
        manager.inventory = lambda: dict(inventory(cpu=4096))
        manager.run_action("disk2ram", {"mode": "all", "units": []})
        manager.move_to = lambda device: None
        manager.inventory = lambda: dict(inventory(cpu=0, cuda=4096,
                                                    device="cuda:0"))
        manager.run_action("ram2vram", {"mode": "all", "units": []})
        manager._preprocess = lambda *args: ([], [{"type": "image",
                                                   "data": None}], [[32, 32]])
        manager.inventory = lambda: dict(inventory(cpu=4096, device="cpu"))
        with self.assertRaises(OperationError) as ctx:
            manager.run_encode(self._encode_request())
        self.assertEqual(ctx.exception.code, "WEIGHT_MUTATED")
        self.assertEqual(ctx.exception.details["units"][UNIT_MODEL]["device"],
                         "cpu")

    @staticmethod
    def _encode_request():
        return {"prompt": "a cat", "image_b64s": ["AAAA"], "mode": "ref2va",
                "width": 640, "height": 960, "ref_image_size": "match"}

    # -------------------------------------------------- 物理账
    def test_storage_inventory_uses_real_bytes(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
        manager, _, _ = self._manager()
        manager._clip = FakeClip(model=model)
        manager.pc = manager._clip
        manager._projected = False
        inv = manager.inventory()
        expected = sum(p.numel() * p.element_size() for p in model.parameters())
        unit = inv["units"][UNIT_MODEL]
        self.assertEqual(unit["cpu"], expected)
        self.assertEqual(unit["cuda"], 0)
        self.assertEqual(unit["device"], "cpu")
        self.assertTrue(inv["storage_digest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
