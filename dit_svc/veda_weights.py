"""VEDA predictor bundle 显式权重后端：整组四动作。

bundle 是 ``veda.bundle_compat.load(path)`` 得到的 ``LoadedBundle``：
``loaded.bundle.proj_q`` / ``proj_k`` 各 50 层 bf16 tensor（bundle 内已反量化），
外加 budget 派生信息。首版 unit 是整组 ``veda.bundle.model``；后续若要按层
显存优化，再拆 ``veda.layers.N``，但必须先有真实 GPU 搬运与 engine 复用验证。
"""
import os
import types

from dit_svc import weights as W
from dit_svc.group_weights import GroupBackend

ADAPTER_VERSION = "veda-bundle-v0"


class VedaBackend(GroupBackend):
    VALID_KINDS = ("checkpoint",)
    ADAPTER_VERSION = ADAPTER_VERSION

    def __init__(self, runtime, registry):
        super().__init__(runtime, registry)
        self.MODEL_ID = "veda.bundle"
        self.PRIMARY_UNIT = "veda.bundle.model"

    def _tensors(self):
        if self.obj is None:
            return None
        inner = getattr(self.obj, "bundle", None)
        if inner is None:
            return None
        return inner

    def extra_state(self):
        info = {}
        inner = self._tensors()
        if inner is not None:
            info = {"num_layers": getattr(inner, "num_layers", None),
                    "num_heads": getattr(inner, "num_heads", None),
                    "head_dim": getattr(inner, "head_dim", None)}
            budget = getattr(self.obj, "budget_source", None)
            if budget:
                info["budget_source"] = budget
        info["path"] = (self.config or {}).get("path")
        return info

    def build_units(self, cfg):
        return [W.UnitInfo(
            id=self.PRIMARY_UNIT, checkpoint_keys=(),
            expected_bytes=os.path.getsize(cfg["path"]),
            groups=(self.MODEL_ID,), factory="veda-bundle-v1",
            capabilities={"attention": True})]

    def construct_cpu(self, operation=None):
        from dit_svc import veda_star7
        self._stage("读取 veda bundle 到 CPU", operation)
        shim = types.SimpleNamespace(veda_bundle=self.config["path"])
        self.obj = veda_star7.load_bundle(
            shim, log=lambda text: print(text, flush=True))

    def move_to(self, device):
        inner = self._tensors()
        if inner is None:
            raise RuntimeError("veda bundle 未加载")
        import torch
        target = torch.device(device)
        inner.proj_q = [t.to(target) for t in inner.proj_q]
        inner.proj_k = [t.to(target) for t in inner.proj_k]

    def release(self):
        refs = []
        if self.obj is not None:
            refs.append(self._release_ref(self.obj))
        self.obj = None
        return refs

    def inventory(self):
        out = {"units": {}, "uninspectable": [], "storage_digest": []}
        inner = self._tensors()
        if inner is None:
            return out
        tensors = {"proj_q": inner.proj_q, "proj_k": inner.proj_k}
        inv = W.storage_inventory(tensors, owner=self.PRIMARY_UNIT)
        devices = {str(t.device)
                   for t in list(inner.proj_q) + list(inner.proj_k)}
        if not devices:
            device = "cpu"
        elif all(d.startswith("cuda") for d in devices):
            device = "cuda:0"
        elif all(d == "cpu" for d in devices):
            device = "cpu"
        else:
            device = "mixed"
        out["units"][self.PRIMARY_UNIT] = {
            "device": device, "cpu": inv.total_cpu, "cuda": inv.total_cuda}
        out["uninspectable"] += list(inv.uninspectable)
        out["storage_digest"] = [e["storage_id"] for e in inv.entries]
        return out
