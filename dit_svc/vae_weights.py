"""VAE（vae.video / vae.audio）显式权重后端：整组四动作。

ComfyUI 的 VAE 由 ``comfy.sd.VAE(sd=..., metadata=..., device=...)`` 一次构造
整组；视频 VAE 还要按 checkpoint 的 decoder 深度选择 Light/官方 tile 几何
（复用 runtime._build_video_vae）。因此 unit 首版是整组一个
``<model>.model``，等真实结构拆分验证后再细化为 encoder/decoder/shared。

disk2ram 使用 runner 的 ``stream_load_safetensors_to_dict(path, device="cpu")``：
真实读取该 VAE 文件到 CPU，不构造 DiT/TE，也不上卡。
"""
import os

from dit_svc import weights as W
from dit_svc.group_weights import GroupBackend

ADAPTER_VERSION = "vae-comfy-v0"


class VaeBackend(GroupBackend):
    VALID_KINDS = ("checkpoint",)
    ADAPTER_VERSION = ADAPTER_VERSION

    def __init__(self, runtime, registry, model_id, video):
        super().__init__(runtime, registry)
        self.MODEL_ID = model_id
        self.video = bool(video)
        self.PRIMARY_UNIT = "%s.model" % model_id

    def extra_state(self):
        return {"video": self.video,
                "path": (self.config or {}).get("path")}

    def build_units(self, cfg):
        return [W.UnitInfo(
            id=self.PRIMARY_UNIT, checkpoint_keys=(),
            expected_bytes=os.path.getsize(cfg["path"]),
            groups=(self.MODEL_ID,), factory="vae-comfy-v1",
            capabilities={"encode": self.video, "decode": True,
                          "video": self.video})]

    def construct_cpu(self, operation=None):
        import torch
        import comfy.sd

        runtime = self.runtime
        loader = getattr(getattr(runtime, "rt", None),
                         "stream_load_safetensors_to_dict", None)
        if loader is None:
            raise RuntimeError("runner 缺少 stream_load_safetensors_to_dict")
        self._stage("读取 %s 到 CPU" % self.MODEL_ID, operation)
        state, metadata = loader(self.config["path"], device="cpu")
        try:
            if self.video:
                from dit_svc.runtime import _build_video_vae
                vae = _build_video_vae(comfy, state, metadata,
                                       torch.device("cpu"), torch)
            else:
                vae = comfy.sd.VAE(sd=state, metadata=metadata,
                                   device=torch.device("cpu"))
        finally:
            state.clear()
        vae.first_stage_model.requires_grad_(False)
        vae.patcher.offload_device = torch.device("cpu")
        self.obj = vae
        if self.video:
            self.runtime.vvae = vae
        else:
            self.runtime.avae = vae

    def move_to(self, device):
        if self.obj is None:
            raise RuntimeError("%s 未加载" % self.MODEL_ID)
        self.obj.patcher.model.to(device)

    def release(self):
        refs = []
        if self.obj is not None:
            try:
                refs.append(self._release_ref(self.obj.first_stage_model))
            except Exception:
                pass
            refs.append(self._release_ref(self.obj))
        if self.video:
            self.runtime.vvae = None
        else:
            self.runtime.avae = None
        self.obj = None
        return refs

    def inventory(self):
        out = {"units": {}, "uninspectable": [], "storage_digest": []}
        if self.obj is None:
            return out
        model = self.obj.patcher.model
        inv = W.storage_inventory(model, owner=self.PRIMARY_UNIT)
        out["units"][self.PRIMARY_UNIT] = {
            "device": self.dominant_device(model),
            "cpu": inv.total_cpu, "cuda": inv.total_cuda}
        out["uninspectable"] += list(inv.uninspectable)
        out["storage_digest"] = [e["storage_id"] for e in inv.entries]
        return out
