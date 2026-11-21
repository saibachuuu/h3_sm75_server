"""VAE（vae.video / vae.audio）显式权重后端：整组四动作。

ComfyUI 的 VAE 由 ``comfy.sd.VAE(sd=..., metadata=..., device=...)`` 一次构造
整组；视频 VAE 还要按 checkpoint 的 decoder 深度选择 Light/官方 tile 几何
（复用 runtime._build_video_vae）。因此 unit 首版是整组一个
``<model>.model``，等真实结构拆分验证后再细化为 encoder/decoder/shared。

重要：量化 VAE（int8_convrot）的量化 Linear 在 CPU 构造后仅靠 ``.to(cuda)``
不会同步内部设备选择，decode 时 attention 的 q/k/v 仍在 CPU（实测
``q,k,v must be on the same CUDA device``）。因此 ``ram2vram`` 对 VAE 采用
“在目标设备重建”的等价搬运：先释放 CPU 实例，再以 device=cuda 构造。
"""
import gc
import os

from dit_svc import weights as W
from dit_svc.group_weights import GroupBackend

ADAPTER_VERSION = "vae-comfy-v1"


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

    # ---------------------------------------------------------- 构造
    def construct_cpu(self, operation=None):
        import torch
        self._build(torch.device("cpu"), operation)

    def _build(self, device, operation=None):
        import torch
        import comfy.sd

        runtime = self.runtime
        loader = getattr(getattr(runtime, "rt", None),
                         "stream_load_safetensors_to_dict", None)
        if loader is None:
            raise RuntimeError("runner 缺少 stream_load_safetensors_to_dict")
        self._stage("读取 %s 到 %s" % (self.MODEL_ID, device), operation)
        # 关键：sd 本身必须先在目标设备。Comfy 的 VAE(device=...) 只设
        # patcher/load_device，不会把 CPU sd 搬上卡；旧 _load() 也是先用
        # device="cuda" 读盘再构造。
        read_device = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
        state, metadata = loader(self.config["path"], device=read_device)
        try:
            if self.video:
                from dit_svc.runtime import _build_video_vae
                vae = _build_video_vae(comfy, state, metadata, device, torch)
            else:
                vae = comfy.sd.VAE(sd=state, metadata=metadata, device=device)
        finally:
            state.clear()
        vae.first_stage_model.requires_grad_(False)
        vae.patcher.offload_device = torch.device("cpu")
        self._attach(vae)

    def _attach(self, vae):
        self.obj = vae
        if self.video:
            self.runtime.vvae = vae
        else:
            self.runtime.avae = vae

    # ---------------------------------------------------------- 搬运
    def move_to(self, device):
        if self.obj is None:
            raise RuntimeError("%s 未加载" % self.MODEL_ID)
        if device == "cpu":
            self.obj.patcher.model.to("cpu")
            return
        # 量化 VAE 的 CPU 实例不能靠 .to(cuda) 正确 decode；在目标设备重建。
        import torch
        self._detach()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        self._build(torch.device(device))

    def _detach(self):
        self.obj = None
        if self.video:
            self.runtime.vvae = None
        else:
            self.runtime.avae = None

    def release(self):
        refs = []
        if self.obj is not None:
            try:
                refs.append(self._release_ref(self.obj.first_stage_model))
            except Exception:
                pass
            refs.append(self._release_ref(self.obj))
        self._detach()
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
