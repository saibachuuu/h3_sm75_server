"""CPU contracts for TE native registration, eviction guards and storage audits."""
import contextlib
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dit_svc.comfy_bindings import ImplicitLoadRefused
from dit_svc.operations import OperationError
from dit_svc.te_residency import TeResidency


class Patcher:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.load_device = "cuda:0"
        self.offload_device = "cpu"
        self.parent = None
        self.gpu = False

    def is_dynamic(self):
        return False

    def model_patches_models(self):
        return []


class Backend:
    def __init__(self, patcher):
        self.obj = types.SimpleNamespace(patcher=patcher)

    def _clip_model(self):
        return self.obj.patcher.model

    def inventory(self):
        gpu = self.obj.patcher.gpu
        return {"units": {"te.model": {"device": "cuda:0" if gpu else "cpu",
                                         "cpu": 0 if gpu else 24,
                                         "cuda": 24 if gpu else 0}},
                "uninspectable": [], "storage_digest": []}


class OtherBackend:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)

    def inventory(self):
        from dit_svc import weights as W
        inv = W.storage_inventory(self.model)
        return {"units": {"dit.blocks.0": {"device": "cpu", "cpu": inv.total_cpu,
                                              "cuda": 0}},
                "storage_digest": [entry["storage_id"] for entry in inv.entries],
                "uninspectable": inv.uninspectable}


def setup():
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []
    calls = {"global_load": 0, "free": 0, "native_load": 0, "unload": 0}

    def load_models_gpu(models, memory_required=0, force_patch_weights=False,
                        minimum_memory_required=None, force_full_load=False):
        calls["global_load"] += 1
        raise AssertionError("TE must not call the global loader")

    def free_memory(memory_required, device, keep_loaded=[], for_dynamic=False,
                    pins_required=0, ram_required=0):
        calls["free"] += 1
        mm.current_loaded_models.clear()

    class Loaded:
        def __init__(self, patcher):
            self.model = patcher
            self.device = patcher.load_device
            self.currently_used = True

        def model_load(self):
            calls["native_load"] += 1
            self.model.gpu = True

        def model_unload(self):
            calls["unload"] += 1
            self.model.gpu = False

    mm.load_models_gpu = load_models_gpu
    mm.free_memory = free_memory
    mm.LoadedModel = Loaded
    backend = Backend(Patcher())
    other = OtherBackend()
    runtime = types.SimpleNamespace(mm=mm, device_healthy=True,
                                    weights=types.SimpleNamespace(
                                        te=backend, backends={"te": backend, "dit": other}))
    residence = TeResidency(runtime)
    return residence, runtime, backend, other, mm, calls


class ResidenceTests(unittest.TestCase):
    def test_move_uses_native_registration_without_global_eviction(self):
        residence, runtime, backend, other, mm, calls = setup()
        stranger = Patcher()
        mm.current_loaded_models.append(mm.LoadedModel(stranger))
        old = other.model.weight.data_ptr()
        residence.move(backend, "cuda:0")
        residence.move(backend, "cuda:0")
        self.assertEqual(calls["native_load"], 1)
        self.assertEqual(calls["global_load"], 0)
        self.assertEqual(calls["free"], 0)
        self.assertEqual(len(mm.current_loaded_models), 2)
        self.assertEqual(other.model.weight.data_ptr(), old)
        residence.move(backend, "cpu")
        self.assertEqual(calls["unload"], 1)
        self.assertEqual(len(mm.current_loaded_models), 1)
        residence.unregister()
        self.assertIsNone(residence.bindings.registry.get("te"))

    def test_inference_only_checks_existing_booking(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        with residence.inference(backend) as proof:
            mm.load_models_gpu([backend.obj.patcher], memory_required=100000)
        self.assertTrue(proof["unchanged"])
        self.assertEqual(len(proof["storage_sha256"]), 64)
        self.assertEqual(calls["global_load"], 0)

    def test_unowned_model_cannot_be_evicted(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        mm.current_loaded_models.append(mm.LoadedModel(Patcher()))
        with residence.inference(backend):
            with self.assertRaises(ImplicitLoadRefused):
                mm.free_memory(100000, "cuda:0")
        self.assertEqual(calls["free"], 0)
        self.assertEqual(len(mm.current_loaded_models), 2)

    def test_helper_thread_cannot_load_even_an_unowned_model(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        errors = []
        def worker():
            try:
                mm.load_models_gpu([Patcher()])
            except ImplicitLoadRefused as error:
                errors.append(error)
        with residence.inference(backend):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(3)
        self.assertEqual(len(errors), 1)
        self.assertEqual(calls["global_load"], 0)

    def test_physical_mutation_detected_even_when_device_labels_stay_same(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        with self.assertRaises(OperationError) as error:
            with residence.inference(backend):
                other.model.weight = torch.nn.Parameter(other.model.weight.clone())
        self.assertEqual(error.exception.code, "WEIGHT_MUTATED")
        self.assertFalse(runtime.device_healthy)

    def test_audit_failure_is_not_swallowed(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        with self.assertRaises(RuntimeError):
            with residence.inference(backend):
                other.inventory = lambda: (_ for _ in ()).throw(RuntimeError("audit failed"))

    def test_missing_booking_cannot_be_repaired_by_inference(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        mm.current_loaded_models.clear()
        with self.assertRaises(OperationError) as error:
            with residence.inference(backend):
                self.fail("must not execute")
        self.assertEqual(error.exception.code, "MISSING_RESIDENCE")
        self.assertEqual(calls["native_load"], 1)

    def test_alias_and_replaced_wrapper_refuse_execution(self):
        residence, runtime, backend, other, mm, calls = setup()
        residence.move(backend, "cuda:0")
        module = types.ModuleType("unsafe_plugin_test")
        module.captured = residence.bindings._record.originals["load_models_gpu"]
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaises(OperationError) as error:
                residence.require(backend)
            self.assertEqual(error.exception.code, "UNSAFE_COMFY_ALIAS")
        mm.free_memory = lambda *args, **kwargs: None
        with self.assertRaises(OperationError) as error:
            residence.require(backend)
        self.assertEqual(error.exception.code, "BINDINGS_CONFLICT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
