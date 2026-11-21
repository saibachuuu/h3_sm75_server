"""P2 模型 adapter 底座：接口 + fake 实现（真选 key 物化 / 拓扑保持）。

真实 H3/Comfy 构造器尚未提供（见 docs/p0_weights.md），本文件只收敛
adapter 接口形状，并用 FakeAdapter 证明四动作 + partial purge/reload
闭环的正确语义。真实 H3 adapter 由主代理在拿到源码后实现，必须满足：

* 首次 disk2ram 只读所选 unit 的 checkpoint keys（FakeSource.accessed_keys
  即此断言的写法）；
* 未选 unit 无真实 storage（placeholder.storage is None）；
* partial purge 不删除拓扑槽位（FakeSkeleton.slots 保持索引稳定）；
* 补回只读缺失 unit 的 keys，既有驻留块 storage 不被替换。

本文件 torch 可选：仅 fake 物化路径需要 torch；接口定义本身标准库可用。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .weights import UnitCatalog, WeightsError


class AdapterError(WeightsError):
    code = "ADAPTER_ERROR"


class SkeletonNoStorage(WeightsError):
    """无存储骨架被误用为有权重。"""
    code = "SKELETON_NO_STORAGE"


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


class ModelAdapter:
    """模型特有 factory/bind/move/purge 接口（P0 收敛形状，P2 fake 验证）。"""

    family = "base"

    def build_skeleton(self, catalog: UnitCatalog) -> Any:
        raise NotImplementedError

    def materialize_unit(self, skeleton: Any, unit_id: str,
                         tensors: Mapping[str, Any]) -> Any:
        """把该 unit 的所选 key 张量绑定进骨架槽位。"""
        raise NotImplementedError

    def move_unit(self, skeleton: Any, unit_id: str, device: str) -> None:
        raise NotImplementedError

    def purge_unit(self, skeleton: Any, unit_id: str) -> None:
        raise NotImplementedError

    def unit_handle(self, skeleton: Any, unit_id: str) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# fake 构造：无存储占位 / 拓扑槽位 / 假量化包装
# ---------------------------------------------------------------------------


class NoStoragePlaceholder:
    """无存储占位：catalog/factory 信息可保留，但 storage 必须为 None。"""

    __slots__ = ("unit_id", "checkpoint_keys", "expected_bytes", "storage")

    def __init__(self, unit_id: str, checkpoint_keys: Sequence[str],
                 expected_bytes: int):
        self.unit_id = unit_id
        self.checkpoint_keys = tuple(checkpoint_keys)
        self.expected_bytes = int(expected_bytes)
        self.storage = None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "NoStoragePlaceholder(%s)" % (self.unit_id,)


class FakeQuantTensor:
    """假量化张量：外层无 storage，内部 _qdata/_scale 为真实 torch 张量。

    用于证明 storage_inventory 会展开量化内部张量而不是只看逻辑 dtype。
    """

    def __init__(self, qdata: Any, scale: Any):
        self._qdata = qdata
        self._scale = scale
        self.storage = None  # 外层不直接持有 storage

    @property
    def device(self) -> str:
        try:
            return str(self._qdata.device)
        except Exception:
            return "cpu"


class FakeSource:
    """假 checkpoint 源：记录被读取的 key，证明只读所选 keys。"""

    def __init__(self, tensors: Mapping[str, Any]):
        self._tensors = dict(tensors)
        self.accessed_keys: List[str] = []

    def header_keys(self) -> List[str]:
        return list(self._tensors.keys())

    def read_keys(self, keys: Sequence[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            if k not in self._tensors:
                raise AdapterError("fake source missing key %r" % (k,),
                                   details={"key": k})
            self.accessed_keys.append(k)
            out[k] = self._tensors[k]
        return out

    def reset_access_log(self) -> None:
        self.accessed_keys = []


class FakeSkeleton:
    """拓扑稳定的假骨架：slots 索引创建后永不删除/重排。

    slots: {slot_index: {"unit_id": str, "payload": ...}}；
    payload 为 NoStoragePlaceholder（absent）或 materialized dict。
    """

    def __init__(self, catalog: UnitCatalog):
        self.catalog = catalog
        self.slot_of_unit: Dict[str, int] = {}
        self.slots: Dict[int, Dict[str, Any]] = {}
        for i, uid in enumerate(catalog.unit_ids()):
            info = catalog.units[uid]
            self.slot_of_unit[uid] = i
            self.slots[i] = {
                "unit_id": uid,
                "payload": NoStoragePlaceholder(uid, info.checkpoint_keys,
                                                info.expected_bytes),
            }

    def num_slots(self) -> int:
        return len(self.slots)


class FakeAdapter(ModelAdapter):
    """真选 key 物化的 fake adapter（CPU torch 张量，无 GPU 语义）。"""

    family = "fake"

    def __init__(self, source: FakeSource,
                 quant_units: Sequence[str] = (),
                 view_units: Sequence[str] = ()):
        """
        quant_units: 这些 unit 的首个 key 用 FakeQuantTensor 包装
                    （行数一半 qdata int8 + scale fp32），模拟量化内部张量；
        view_units: 这些 unit 的第二个 key 存为首 key 张量的视图
                   （共享 storage，用于去重测试）。
        """
        self.source = source
        self.quant_units = set(quant_units)
        self.view_units = set(view_units)

    # -- skeleton --
    def build_skeleton(self, catalog: UnitCatalog) -> FakeSkeleton:
        return FakeSkeleton(catalog)

    # -- materialize: 只读该 unit 的 keys --
    def materialize_unit(self, skeleton: FakeSkeleton, unit_id: str,
                         tensors: Optional[Mapping[str, Any]] = None) -> Any:
        if unit_id not in skeleton.catalog.units:
            raise AdapterError("unknown unit %r" % (unit_id,))
        info = skeleton.catalog.units[unit_id]
        if tensors is None:
            tensors = self.source.read_keys(list(info.checkpoint_keys))
        else:
            # 调用方显式给张量时仍校验 key 集合精确一致（防多读/漏读）。
            if set(tensors.keys()) != set(info.checkpoint_keys):
                raise AdapterError(
                    "tensor keys mismatch for %r" % (unit_id,),
                    details={"want": sorted(info.checkpoint_keys),
                             "got": sorted(tensors.keys())})
        payload = self._wrap_payload(unit_id, dict(tensors))
        slot = skeleton.slot_of_unit[unit_id]
        skeleton.slots[slot]["payload"] = payload
        return payload

    def _wrap_payload(self, unit_id: str,
                      tensors: Dict[str, Any]) -> Dict[str, Any]:
        keys = sorted(tensors.keys())
        if unit_id in self.quant_units and keys:
            import torch  # fake 物化需要 torch；接口本身不依赖

            base = tensors[keys[0]]
            if not isinstance(base, torch.Tensor):
                raise AdapterError("quant fake needs torch tensor")
            flat = base.contiguous().view(-1)
            qdata = (flat[:max(1, flat.numel() // 2)].to(torch.int8)
                     .contiguous())
            scale = base.new_ones(4, dtype=torch.float32)
            tensors[keys[0]] = FakeQuantTensor(qdata, scale)
        if unit_id in self.view_units and len(keys) >= 2:
            base = tensors[keys[0]]
            inner = base._qdata if isinstance(base, FakeQuantTensor) else base
            try:
                tensors[keys[1]] = inner.view(-1)  # 共享 storage 的视图
            except Exception:
                tensors[keys[1]] = inner
        return {"unit_id": unit_id, "tensors": tensors,
                "logical_device": "cpu"}

    # -- move: fake 只改 logical_device（CPU 本机无 CUDA，物理搬运未验证） --
    def move_unit(self, skeleton: FakeSkeleton, unit_id: str,
                  device: str) -> None:
        handle = self.unit_handle(skeleton, unit_id)
        if isinstance(handle, NoStoragePlaceholder):
            raise SkeletonNoStorage("unit %r has no storage to move" % (unit_id,))
        if device not in ("cpu", "cuda:0"):
            raise AdapterError("fake adapter only knows cpu|cuda:0",
                               details={"device": device})
        handle["logical_device"] = device

    # -- purge: 换回占位，槽位保留；调用方负责清 registry 强引用 --
    def purge_unit(self, skeleton: FakeSkeleton, unit_id: str) -> None:
        if unit_id not in skeleton.slot_of_unit:
            raise AdapterError("unknown unit %r" % (unit_id,))
        info = skeleton.catalog.units[unit_id]
        slot = skeleton.slot_of_unit[unit_id]
        skeleton.slots[slot]["payload"] = NoStoragePlaceholder(
            unit_id, info.checkpoint_keys, info.expected_bytes)

    def unit_handle(self, skeleton: FakeSkeleton, unit_id: str) -> Any:
        if unit_id not in skeleton.slot_of_unit:
            raise AdapterError("unknown unit %r" % (unit_id,))
        return skeleton.slots[skeleton.slot_of_unit[unit_id]]["payload"]

    def is_resident(self, skeleton: FakeSkeleton, unit_id: str) -> bool:
        return not isinstance(self.unit_handle(skeleton, unit_id),
                              NoStoragePlaceholder)


# ---------------------------------------------------------------------------
# P2 fake 生命周期编排（registry + adapter 联动，供自测与主代理复用）
# ---------------------------------------------------------------------------

def fake_disk2ram(registry: Any, adapter: FakeAdapter,
                  skeleton: FakeSkeleton, model_id: str,
                  unit_ids: Sequence[str]) -> Dict[str, Any]:
    """所选 unit 真选 key 物化；未选 unit 不触碰（access log 可证）。"""
    from .weights import storage_inventory  # 延迟导入，保持模块 stdlib 可读性
    completed, failed = [], []
    for uid in unit_ids:
        try:
            payload = adapter.materialize_unit(skeleton, uid)
            inv = storage_inventory(payload["tensors"], owner=uid).to_summary()
            registry.set_unit_loaded(
                model_id, uid, "cpu", resident_bytes_cpu=inv["total_cpu"],
                resident_bytes_cuda=0, storage_digest=inv["storages"])
            completed.append(uid)
        except Exception as e:  # noqa: BLE001 - 失败记 error 状态后继续上报
            registry.mark_error(model_id, uid, "%s: %s" % (type(e).__name__, e),
                                recovery=["检查 fake source 是否含该 unit keys",
                                          "用 GET 状态确认 catalog key 集合"])
            failed.append(uid)
    return registry.summarize_operation(
        "disk2ram", model_id, list(unit_ids), completed=completed,
        failed=failed, not_attempted=[u for u in unit_ids
                                      if u not in completed and u not in failed])


def fake_move(registry: Any, adapter: FakeAdapter, skeleton: FakeSkeleton,
              model_id: str, unit_ids: Sequence[str], action: str,
              device: str) -> Dict[str, Any]:
    completed, failed = [], []
    for uid in unit_ids:
        try:
            adapter.move_unit(skeleton, uid, device)
            registry.set_unit_loaded(
                model_id, uid, device,
                resident_bytes_cpu=registry.unit_record(
                    model_id, uid)["resident_bytes_cpu"],
                resident_bytes_cuda=registry.unit_record(
                    model_id, uid)["resident_bytes_cuda"])
            completed.append(uid)
        except Exception as e:  # noqa: BLE001
            registry.mark_error(model_id, uid, "%s: %s" % (type(e).__name__, e),
                                recovery=["用 recover 修复 error/mixed 后重试"])
            failed.append(uid)
    return registry.summarize_operation(
        action, model_id, list(unit_ids), completed=completed, failed=failed,
        not_attempted=[])


def fake_purge(registry: Any, adapter: FakeAdapter, skeleton: FakeSkeleton,
               model_id: str, unit_ids: Sequence[str]) -> Dict[str, Any]:
    completed, failed = [], []
    for uid in unit_ids:
        try:
            adapter.purge_unit(skeleton, uid)
            registry.set_unit_absent(model_id, uid)
            completed.append(uid)
        except Exception as e:  # noqa: BLE001
            registry.mark_error(model_id, uid, "%s: %s" % (type(e).__name__, e),
                                recovery=["确认该 unit 无 CUDA 残留后重试"])
            failed.append(uid)
    return registry.summarize_operation(
        "purgefromram", model_id, list(unit_ids), completed=completed,
        failed=failed, not_attempted=[])
