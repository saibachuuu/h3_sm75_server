"""Real Comfy LoadedModel/ModelPatcher CPU bookkeeping contract, no checkpoint/GPU.

This verifies native registration/unload and resident loader interception against
an installed ComfyUI. It is NOT a real TE/GPU performance or numerical test.
"""
import argparse
import gc
import os
import sys
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.comfy_root)
    # Configure real Comfy imports explicitly for CPU; never select a GPU.
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    import torch
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher
    from dit_svc.comfy_bindings import ComfyBindings, PHASE_RESIDENT_INFER
    from dit_svc.weights import storage_inventory

    model = torch.nn.Linear(4, 4)
    patcher = ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    entry = mm.LoadedModel(patcher)
    entry.model_load()
    mm.current_loaded_models.insert(0, entry)
    assert entry.real_model() is model
    assert patcher.loaded_size() == storage_inventory(model).total_cpu
    bindings = ComfyBindings()
    bindings.install(mm)
    bindings.registry.register(patcher, "te", {"te.model": "cpu"})
    before = storage_inventory(model).to_summary()["storages"]
    try:
        with bindings.execution(PHASE_RESIDENT_INFER, "cpu-native-contract",
                                declared_units=("te.model",),
                                expected_devices={"te.model": "cpu"}):
            mm.load_models_gpu([patcher], memory_required=1024 ** 3)
        assert storage_inventory(model).to_summary()["storages"] == before
        assert bindings.state()["stats"]["resident_passes"] == 1
        entry.model_unload()
        mm.current_loaded_models.remove(entry)
        assert patcher.loaded_size() == 0
        bindings.registry.unregister("te")
    finally:
        bindings.uninstall()
    reference = weakref.ref(model)
    del model, patcher, entry
    gc.collect()
    assert reference() is None, "native bookkeeping leaked model"
    print("OK: real Comfy CPU native load/register/resident-check/unload/release")
    print("GPU TE inference remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
