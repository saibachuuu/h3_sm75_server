"""adapter 包：fake（自测语义）+ h3_comfy（真实 Comfy 绑定）。

保持 `from dit_svc.model_adapters import FakeAdapter, ...` 旧导入可用。
"""

from .fake import (AdapterError, FakeAdapter, FakeQuantTensor, FakeSkeleton,
                   FakeSource, ModelAdapter, NoStoragePlaceholder,
                   SkeletonNoStorage, fake_disk2ram, fake_move, fake_purge)
from .h3_comfy import (ComfyUnavailable, H3ComfyAdapter, ShapeProxySD,
                       WAIVED_KEYS)

__all__ = [
    "AdapterError", "ComfyUnavailable", "FakeAdapter", "FakeQuantTensor",
    "FakeSkeleton", "FakeSource", "H3ComfyAdapter", "ModelAdapter",
    "NoStoragePlaceholder", "ShapeProxySD", "SkeletonNoStorage", "WAIVED_KEYS",
    "fake_disk2ram", "fake_move", "fake_purge",
]
