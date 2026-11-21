"""服务所有权感知的 ComfyUI 绑定层（P0：可测试的最小内核）。

设计边界
========

本模块只做三件事，**不**实现权重读写、不实现构造器重载：

1. **记录原始函数与版本签名**。安装时保存 `load_models_gpu` / `free_memory` /
   `load_model_gpu` 的原函数对象、`inspect.signature` 的参数名与函数体摘要；
   与已核实版本（remote ComfyUI 5f415e31a）的签名比对，不匹配直接拒绝安装
   （除非显式 `allow_unknown_signature=True`）。
2. **所有权表**。`OwnedModelRegistry` 记录 patcher/model/clone identity →
   `model_id + generation + unit→device`。clone（含 SigmaShift 派生包装）通过
   `parent` 链或共享 `model` 对象解析为同一 owner，不重复登记。
3. **执行上下文**。`ExecutionContext` 声明 phase / operation / 可用 unit /
   期望设备；包装函数只在上下文内改变行为，上下文外**原样委托**，不影响
   ComfyUI 的非服务用途。

包装语义（`load_models_gpu`）
============================

| phase            | 已驻留且在册 | 已登记但未驻留        | 未登记/unrelated |
|------------------|--------------|-----------------------|------------------|
| resident_infer   | 只校验+置 used，绝不调用原函数 | `MissingResidence` | `UnownedModelError` |
| attention_apply  | 同上         | `MissingResidence`    | `UnownedModelError` |
| construct_cpu    | —            | 仅当目标 load_device 是 CPU 才委托 | 需 `allow_unregistered`，否则拒绝 |
| explicit_move    | 委托原函数    | 委托原函数            | 需 `allow_unrelated`，否则拒绝 |
| recovery         | 委托原函数    | 委托原函数            | 同上 |

- **不做** `return None` 了事：resident 路径必须真的在
  `current_loaded_models` 里找到对应 LoadedModel（device 匹配），否则报错；
  找不到也不伪造条目。
- 委托路径调用原函数后**回读簿记**：请求的 owned 模型必须出现
  LoadedModel 且 device 正确；其他 owned 模型设备快照不得变化，否则
  `UnexpectedWeightMutation`。
- `free_memory` 在 resident_infer/attention_apply 下若会驱逐 owned 模型则拒绝
  （隐式 offload）。

`load_model_gpu` 不必单独包装：真实 ComfyUI 的实现体是
`return load_models_gpu([model])`，模块全局查找，天然经过本包装。为防第三方
提前 `from ... import load_models_gpu` 捕获裸函数，`audit_aliases()` 会扫描
`sys.modules` 报出这样的模块（只报告，不改别人）。

本模块不 import torch / comfy：可在无 GPU 的隔离环境导入与测试。
"""
from __future__ import annotations

import contextlib
import collections
import contextvars
import hashlib
import inspect
import itertools
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "PHASE_CONSTRUCT_CPU",
    "PHASE_EXPLICIT_MOVE",
    "PHASE_RESIDENT_INFER",
    "PHASE_ATTENTION_APPLY",
    "PHASE_RECOVERY",
    "PHASES",
    "BindingsError",
    "UnsupportedComfyVersion",
    "OwnershipError",
    "UnownedModelError",
    "ImplicitLoadRefused",
    "MissingUnits",
    "MissingResidence",
    "BookkeepingError",
    "UnexpectedWeightMutation",
    "ExecutionContext",
    "OwnedModelRegistry",
    "OwnedModel",
    "FunctionSignature",
    "ComfyBindings",
]

PHASE_CONSTRUCT_CPU = "construct_cpu"
PHASE_EXPLICIT_MOVE = "explicit_move"
PHASE_RESIDENT_INFER = "resident_infer"
PHASE_ATTENTION_APPLY = "attention_apply"
PHASE_RECOVERY = "recovery"
PHASES = (
    PHASE_CONSTRUCT_CPU,
    PHASE_EXPLICIT_MOVE,
    PHASE_RESIDENT_INFER,
    PHASE_ATTENTION_APPLY,
    PHASE_RECOVERY,
)

#: 允许“调用原函数把权重搬到目标设备”的 phase。
_MOVING_PHASES = (PHASE_EXPLICIT_MOVE, PHASE_RECOVERY)
#: 只允许使用已完整驻留资源的 phase（任何隐式 load/offload 都是错误）。
_STRICT_PHASES = (PHASE_RESIDENT_INFER, PHASE_ATTENTION_APPLY)


# ---------------------------------------------------------------------------
# 异常：全部带可读原因，供 operation 结果分类（不吞掉成 500）
# ---------------------------------------------------------------------------


class BindingsError(RuntimeError):
    """绑定层基类。

    跨模块（runtime/queue/entry）集成时不应把一切异常都变成 500：
    每个子类带 `code` / `http_status`，`details` 是**纯 JSON dict 的数据属性**
    （与 WeightsError/ArtifactError 协议一致，`OperationError.from_exception`
    直接读 `.details`，不能是 bound method），可用 `as_operation_error()`
    直接塞进 operation 回执。
    """

    code = "bindings_error"
    http_status = 500

    #: 子类只实现这个 hook；`details` 负责 JSON 化与"无 Tensor/method"保证。
    def _detail_fields(self) -> dict[str, Any]:
        return {}

    @property
    def details(self) -> dict[str, Any]:
        return _details_json(self._detail_fields())

    def as_operation_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "http_status": self.http_status,
            "message": str(self),
            "details": self.details,
        }


class UnsupportedComfyVersion(BindingsError):
    """真实函数签名与已核实版本不一致（部署/版本问题，不是调用方错误）。"""

    code = "unsupported_comfy_version"
    http_status = 500


class OwnershipError(BindingsError):
    """所有权表使用错误（重复登记、未知 model_id、generation 冲突）。"""

    code = "ownership_error"
    http_status = 500


class UnownedModelError(BindingsError):
    """严格 phase 下出现不属于本服务的模型。"""

    code = "unowned_model"
    http_status = 409


class ImplicitLoadRefused(BindingsError):
    """严格 phase 下请求了会搬动权重的操作。"""

    code = "implicit_load_refused"
    http_status = 409


class MissingUnits(ImplicitLoadRefused):
    """请求的模型缺少上下文声明的 unit，或声明了无法核对的 unit。"""

    code = "missing_units"

    def __init__(self, message: str, *, missing: Any = (),
                 unknown: Any = ()) -> None:
        super().__init__(message)
        self.missing = sorted(str(x) for x in missing)
        self.unknown = sorted(str(x) for x in unknown)

    def _detail_fields(self) -> dict[str, Any]:
        return {"missing": self.missing, "unknown": self.unknown}


class MissingResidence(ImplicitLoadRefused):
    """模型没有在 ComfyUI 台账中登记为已驻留（不会伪造条目）。"""

    code = "missing_residence"
    http_status = 503


class BookkeepingError(BindingsError):
    """委托原函数后，ComfyUI 台账没有出现应有的登记。"""

    code = "bookkeeping_error"
    http_status = 500


class UnexpectedWeightMutation(BindingsError):
    """执行期间本不该移动的 owned 模型设备发生了变化。"""

    code = "unexpected_weight_mutation"
    http_status = 500

    def __init__(self, message: str, *, moved: Any = ()) -> None:
        super().__init__(message)
        # moved 既可能是 {"model": {unit: {...}}} 也可能是 [unit, ...]；
        # 原样保留，details 负责 JSON 化。
        self.moved = moved

    def _detail_fields(self) -> dict[str, Any]:
        return {"moved": self.moved}


# ---------------------------------------------------------------------------
# 版本签名
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionSignature:
    """一个被包装函数的可审计指纹。"""

    name: str
    qualname: str
    module: str
    params: tuple[str, ...]
    code_sha1: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "module": self.module,
            "params": list(self.params),
            "code_sha1": self.code_sha1,
        }

    def describe(self) -> str:
        return "%s(%s)" % (self.qualname, ", ".join(self.params))


def capture_signature(func: Callable[..., Any]) -> FunctionSignature:
    """抓取函数签名（vectorcall 的内建函数会缺 owner 信息，仍尽量记录）。"""
    try:
        params = tuple(str(p) for p in inspect.signature(func).parameters)
    except (TypeError, ValueError):  # pragma: no cover - 真实 ComfyUI 是 py 函数
        params = ()
    code = getattr(func, "__code__", None)
    digest = ""
    if code is not None:
        digest = hashlib.sha1(code.co_code).hexdigest()
    return FunctionSignature(
        name=getattr(func, "__name__", "?"),
        qualname=getattr(func, "__qualname__", getattr(func, "__name__", "?")),
        module=getattr(func, "__module__", "?"),
        params=params,
        code_sha1=digest,
    )


#: 已核实的 remote ComfyUI 5f415e31a / comfy 源码（sha256 见 docs/p0_bindings.md）。
#: 键：mm 模块上的函数名；值：必须完全一致的参数名序列。
EXPECTED_SIGNATURES: dict[str, tuple[str, ...]] = {
    "load_models_gpu": (
        "models",
        "memory_required",
        "force_patch_weights",
        "minimum_memory_required",
        "force_full_load",
    ),
    "load_model_gpu": ("model",),
    "free_memory": (
        "memory_required",
        "device",
        "keep_loaded",
        "for_dynamic",
        "pins_required",
        "ram_required",
    ),
}


# ---------------------------------------------------------------------------
# 所有权表
# ---------------------------------------------------------------------------


@dataclass
class OwnedModel:
    """一个 model_id 的代际与单元设备快照。

    units: unit_id -> 实际设备字符串（"cpu" / "cuda:0" / "mixed"）。
    probe: 可选回调，返回最新 unit→device；由 P2 weights 层在执行后刷新。
    """

    model_id: str
    generation: int
    model: Any
    units: dict[str, str]
    load_device: str
    offload_device: str
    patcher_ids: set[int] = field(default_factory=set)
    probe: Callable[[], Mapping[str, str]] | None = None

    def devices(self) -> dict[str, str]:
        if self.probe is not None:
            return {str(k): _device_str(v) for k, v in self.probe().items()}
        return dict(self.units)

    def add_patcher(self, patcher: Any) -> None:
        self.patcher_ids.add(id(patcher))


def _device_str(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


class OwnedModelRegistry:
    """patcher/model/clone identity → owner。

    解析顺序：patcher 身份 → `parent` 链（clone）→ 共享 `model` 对象。
    这样 SigmaShift 之类的派生包装不会另立 owner，也不会漏检。
    """

    def __init__(self) -> None:
        self._by_patcher: dict[int, OwnedModel] = {}
        self._by_model: dict[int, OwnedModel] = {}
        self._by_id: dict[str, OwnedModel] = {}
        self._generations: dict[str, int] = {}
        self._lock = threading.RLock()

    # -- 登记 ---------------------------------------------------------------

    def register(
        self,
        patcher: Any,
        model_id: str,
        units: Mapping[str, Any],
        *,
        generation: int | None = None,
        probe: Callable[[], Mapping[str, str]] | None = None,
    ) -> OwnedModel:
        if not model_id:
            raise OwnershipError("model_id 不能为空")
        with self._lock:
            existing = self._by_id.get(model_id)
            if generation is None:
                generation = self._generations.get(model_id, 0) + 1
            if existing is not None:
                if existing.generation != generation:
                    raise OwnershipError(
                        "model_id %r 已登记 generation=%d，不能换成 %d；"
                        "请先 unregister（全量 purge 后）"
                        % (model_id, existing.generation, generation))
                existing.add_patcher(patcher)
                self._by_patcher[id(patcher)] = existing
                return existing
            entry = OwnedModel(
                model_id=model_id,
                generation=generation,
                model=getattr(patcher, "model", patcher),
                units={str(k): _device_str(v) for k, v in units.items()},
                load_device=_device_str(getattr(
                    patcher, "load_device", "unknown")),
                offload_device=_device_str(getattr(
                    patcher, "offload_device", "unknown")),
                probe=probe,
            )
            entry.add_patcher(patcher)
            self._by_patcher[id(patcher)] = entry
            self._by_model[id(entry.model)] = entry
            self._by_id[model_id] = entry
            self._generations[model_id] = generation
            return entry

    def unregister(self, model_id: str) -> OwnedModel:
        with self._lock:
            entry = self._by_id.pop(model_id, None)
            if entry is None:
                raise OwnershipError("未登记的 model_id %r" % (model_id,))
            for pid in entry.patcher_ids:
                self._by_patcher.pop(pid, None)
            self._by_model.pop(id(entry.model), None)
            return entry

    # -- 查询 ---------------------------------------------------------------

    def resolve(self, obj: Any) -> OwnedModel | None:
        """obj 可以是 patcher、它的 clone、或裸 model 对象。"""
        if obj is None:
            return None
        with self._lock:
            entry = self._by_patcher.get(id(obj))
            if entry is not None:
                return entry
            entry = self._by_model.get(id(obj))
            if entry is not None:
                return entry
            # clone 链：真 ModelPatcher.clone 会把 parent 指向源 patcher
            seen: set[int] = set()
            cur = getattr(obj, "parent", None)
            while cur is not None and id(cur) not in seen:
                seen.add(id(cur))
                entry = self._by_patcher.get(id(cur))
                if entry is not None:
                    self._by_patcher[id(obj)] = entry
                    entry.add_patcher(obj)
                    return entry
                cur = getattr(cur, "parent", None)
            # 共享同一 model 对象（clone_has_same_weights 的弱化版）
            inner = getattr(obj, "model", None)
            if inner is not None:
                entry = self._by_model.get(id(inner))
                if entry is not None:
                    self._by_patcher[id(obj)] = entry
                    entry.add_patcher(obj)
                    return entry
            return None

    def get(self, model_id: str) -> OwnedModel | None:
        return self._by_id.get(model_id)

    def owner_of_unit(self, unit: str) -> OwnedModel | None:
        for entry in self._by_id.values():
            if unit in entry.units:
                return entry
        return None

    def entries(self) -> list[OwnedModel]:
        return list(self._by_id.values())

    def snapshot(self, model_id: str) -> dict[str, str]:
        entry = self._by_id.get(model_id)
        if entry is None:
            raise OwnershipError("未登记的 model_id %r" % (model_id,))
        return entry.devices()

    def snapshot_all(self) -> dict[str, dict[str, str]]:
        return {e.model_id: e.devices() for e in self._by_id.values()}

    def clear(self) -> None:
        with self._lock:
            self._by_patcher.clear()
            self._by_model.clear()
            self._by_id.clear()


# ---------------------------------------------------------------------------
# 执行上下文
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionContext:
    """一次 operation 的资源授权范围。

    declared_units: 本次允许使用的 unit（None = 不额外限制）。
    expected_devices: unit → 期望设备；空表示按 registry 的 load_device 校验。
    allowed_models: 允许被本操作移动的 model_id（仅 explicit_move/recovery）。
    allow_unrelated: 是否允许出现非本服务模型（默认 False）。
    allow_unregistered: construct_cpu 是否允许尚未登记的构造对象。
    """

    operation_id: str
    phase: str
    declared_units: frozenset[str] | None = None
    expected_devices: Mapping[str, str] = field(default_factory=dict)
    allowed_models: frozenset[str] = frozenset()
    allow_unrelated: bool = False
    allow_unregistered: bool = False

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise BindingsError("未知 phase %r（允许 %s）"
                                % (self.phase, ", ".join(PHASES)))
        if not self.operation_id:
            raise BindingsError("ExecutionContext 需要 operation_id")

    def unit_declared(self, unit: str) -> bool:
        return self.declared_units is None or unit in self.declared_units

    def expected_device(self, unit: str) -> str | None:
        return self.expected_devices.get(unit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "phase": self.phase,
            "declared_units": (None if self.declared_units is None
                               else sorted(self.declared_units)),
            "expected_devices": dict(self.expected_devices),
            "allowed_models": sorted(self.allowed_models),
            "allow_unrelated": self.allow_unrelated,
            "allow_unregistered": self.allow_unregistered,
        }


_CURRENT_CONTEXT: contextvars.ContextVar[ExecutionContext | None] = (
    contextvars.ContextVar("dit_comfy_bindings_ctx", default=None))


# ---------------------------------------------------------------------------
# 绑定安装
# ---------------------------------------------------------------------------


@dataclass
class _InstallRecord:
    module: types.ModuleType
    originals: dict[str, Callable[..., Any]]
    wrappers: dict[str, Callable[..., Any]]
    signatures: dict[str, FunctionSignature]
    allow_unknown_signature: bool


class ComfyBindings:
    """安装/还原 ComfyUI 所有权包装，并对外暴露状态。"""

    WRAPPED = ("load_models_gpu", "free_memory")

    def __init__(
        self,
        registry: OwnedModelRegistry | None = None,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.registry = registry if registry is not None else OwnedModelRegistry()
        self.log = log or (lambda _msg: None)
        self._record: _InstallRecord | None = None
        self._active_context: ExecutionContext | None = None
        self._execution_lock = threading.Lock()
        self._stats = {
            "delegated": 0,
            "resident_passes": 0,
            "refused": 0,
            "bookkeeping_asserts": 0,
        }
        self._refusals: collections.deque = collections.deque(maxlen=64)

    # -- 生命周期 -----------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._record is not None

    def install(
        self,
        model_management: types.ModuleType,
        *,
        allow_unknown_signature: bool = False,
    ) -> dict[str, Any]:
        """保存原函数并安装包装；重复调用是 no-op。"""
        if self._record is not None:
            return self.state()
        originals: dict[str, Callable[..., Any]] = {}
        signatures: dict[str, FunctionSignature] = {}
        for name in self.WRAPPED:
            func = getattr(model_management, name, None)
            if not callable(func):
                raise UnsupportedComfyVersion(
                    "comfy.model_management 缺少可调用函数 %r" % (name,))
            sig = capture_signature(func)
            expected = EXPECTED_SIGNATURES.get(name)
            if expected is not None and sig.params != expected:
                message = (
                    "签名为 %s，已核实版本要求 %s(%s)；拒绝安装（可用 "
                    "allow_unknown_signature=True 强制，但必须重新记录 P0 证据）"
                    % (sig.describe(), name, ", ".join(expected)))
                if not allow_unknown_signature:
                    raise UnsupportedComfyVersion(message)
                self.log("[comfy-bindings] WARNING: %s" % message)
            originals[name] = func
            signatures[name] = sig

        wrappers = {
            "load_models_gpu": self._make_load_models_gpu(originals),
            "free_memory": self._make_free_memory(originals),
        }
        for name, wrapper in wrappers.items():
            setattr(model_management, name, wrapper)
        self._record = _InstallRecord(
            module=model_management,
            originals=originals,
            wrappers=wrappers,
            signatures=signatures,
            allow_unknown_signature=allow_unknown_signature,
        )
        self.log("[comfy-bindings] installed load_models_gpu/free_memory "
                 "signatures=%s"
                 % {k: v.describe() for k, v in signatures.items()})
        return self.state()

    def uninstall(self, *, force: bool = False) -> dict[str, Any]:
        """还原原函数。正在执行的上下文内拒绝（除非 force）。"""
        record = self._record
        if record is None:
            return self.state()
        if _CURRENT_CONTEXT.get() is not None and not force:
            raise BindingsError(
                "执行上下文 %r 仍在活动，不能还原 ComfyUI 绑定"
                % (_CURRENT_CONTEXT.get().operation_id,))
        conflicts = []
        for name, wrapper in record.wrappers.items():
            current = getattr(record.module, name, None)
            if current is not wrapper:
                conflicts.append(name)
                continue
            setattr(record.module, name, record.originals[name])
        self._record = None
        if conflicts:
            raise BindingsError(
                "以下函数已被第三方替换，未覆盖回原函数: %s（需人工确认）"
                % (", ".join(conflicts),))
        return self.state()

    @contextlib.contextmanager
    def execution(
        self,
        phase: str,
        operation_id: str,
        *,
        declared_units: Iterable[str] | None = None,
        expected_devices: Mapping[str, Any] | None = None,
        allowed_models: Iterable[str] = (),
        allow_unrelated: bool = False,
        allow_unregistered: bool = False,
    ):
        """进入执行上下文；不支持嵌套（单 worker 语义）。"""
        if _CURRENT_CONTEXT.get() is not None:
            raise BindingsError(
                "执行上下文不支持嵌套：当前 %r"
                % (_CURRENT_CONTEXT.get().operation_id,))
        ctx = ExecutionContext(
            operation_id=operation_id,
            phase=phase,
            declared_units=(None if declared_units is None
                            else frozenset(declared_units)),
            expected_devices={str(k): _device_str(v)
                              for k, v in (expected_devices or {}).items()},
            allowed_models=frozenset(allowed_models),
            allow_unrelated=allow_unrelated,
            allow_unregistered=allow_unregistered,
        )
        if not self._execution_lock.acquire(blocking=False):
            raise ImplicitLoadRefused("另一个 Comfy 执行上下文仍在运行")
        try:
            before = self.registry.snapshot_all()
            self._active_context = ctx
            token = _CURRENT_CONTEXT.set(ctx)
            try:
                yield ctx
            finally:
                _CURRENT_CONTEXT.reset(token)
                error = self._audit_exit(ctx, before)
                if error is not None:
                    raise error
        finally:
            self._active_context = None
            self._execution_lock.release()

    def _audit_exit(self, ctx: ExecutionContext, before) -> BindingsError | None:
        if ctx.phase not in _STRICT_PHASES:
            return None
        try:
            after = self.registry.snapshot_all()
        except Exception as error:  # noqa: BLE001 - 审核不能掩盖主流程
            raise BookkeepingError("resident exit audit 无法完成: %s" % error) from error
        moved = {}
        for model_id, new in after.items():
            old = before.get(model_id)
            if old is None:
                continue
            changed = [u for u in new
                       if old.get(u) != new.get(u)
                       and not ctx.unit_declared(u)]
            if changed:
                moved[model_id] = {u: {"before": old.get(u), "after": new[u]}
                                   for u in changed}
        if not moved:
            return None
        self._stats["refused"] += 1
        message = ("%s 期间未声明 unit 被搬动: %s"
                   % (ctx.phase, {k: sorted(v) for k, v in moved.items()}))
        self._refusals.append({
            "operation_id": ctx.operation_id,
            "phase": ctx.phase,
            "kind": "undeclared_unit_mutation",
            "message": message,
        })
        self.log("[comfy-bindings] EXIT AUDIT: %s" % message)
        return UnexpectedWeightMutation(message, moved=_jsonable(moved))

    @property
    def context(self) -> ExecutionContext | None:
        return _CURRENT_CONTEXT.get()

    # -- 包装实现 -----------------------------------------------------------

    def _make_load_models_gpu(self, originals):
        original = originals["load_models_gpu"]
        bindings = self

        def load_models_gpu(models, memory_required=0,
                            force_patch_weights=False,
                            minimum_memory_required=None,
                            force_full_load=False):
            ctx = bindings.context
            requested = list(models)
            if ctx is None:
                if bindings._active_context is not None:
                    raise ImplicitLoadRefused("helper 线程没有当前操作的权重授权")
                # ContextVar 不传播到第三方 helper 线程 —— 那里 ctx 为 None。
                # 没有上下文 ≠ 有授权：请求里只要有 owned 模型就必须拒绝，
                # 否则 owned 模型可以被线程池/预取线程轻易隐式搬上卡。
                bindings._guard_no_context(requested, "load_models_gpu")
                return original(
                    requested, memory_required=memory_required,
                    force_patch_weights=force_patch_weights,
                    minimum_memory_required=minimum_memory_required,
                    force_full_load=force_full_load)
            classified = bindings._classify(ctx, requested)
            if ctx.phase in _STRICT_PHASES:
                bindings._verify_resident(ctx, classified)
                bindings._stats["resident_passes"] += 1
                return None
            if ctx.phase == PHASE_CONSTRUCT_CPU:
                bindings._assert_cpu_only(ctx, classified)
                return bindings._delegate(
                    ctx, original, requested, classified,
                    dict(memory_required=memory_required,
                         force_patch_weights=force_patch_weights,
                         minimum_memory_required=minimum_memory_required,
                         force_full_load=force_full_load))
            # explicit_move / recovery
            bindings._assert_move_allowed(ctx, classified)
            return bindings._delegate(
                ctx, original, requested, classified,
                dict(memory_required=memory_required,
                     force_patch_weights=force_patch_weights,
                     minimum_memory_required=minimum_memory_required,
                     force_full_load=force_full_load))

        load_models_gpu.__name__ = "load_models_gpu"
        load_models_gpu.__qualname__ = "ComfyBindings.load_models_gpu"
        load_models_gpu._dit_comfy_bindings = True  # type: ignore[attr-defined]
        load_models_gpu._dit_original = original  # type: ignore[attr-defined]
        return load_models_gpu

    def _make_free_memory(self, originals):
        original = originals["free_memory"]
        bindings = self

        def free_memory(memory_required, device, keep_loaded=[],
                        for_dynamic=False, pins_required=0, ram_required=0):
            ctx = bindings.context
            if ctx is not None:
                bindings._refuse_eviction(ctx, keep_loaded)
            elif ctx is None:
                if bindings._active_context is not None:
                    raise ImplicitLoadRefused("helper 线程不得为当前操作驱逐模型")
                # 无上下文同样不能卸 owned：free_memory 会驱逐 current_loaded_models
                # 里除 keep_loaded 之外的模型，包括本服务已驻留的权重。
                bindings._guard_no_context_eviction(keep_loaded, device)
            before = bindings._snapshot_safe()
            try:
                return original(
                    memory_required, device, keep_loaded=keep_loaded,
                    for_dynamic=for_dynamic, pins_required=pins_required,
                    ram_required=ram_required)
            except BaseException as error:
                # 驱逐中途失败同样要如实记录（部分模型可能已被卸下）
                if ctx is not None:
                    bindings._report_partial(before, bindings._snapshot_safe(),
                                             error, ctx)
                raise

        free_memory.__name__ = "free_memory"
        free_memory.__qualname__ = "ComfyBindings.free_memory"
        free_memory._dit_comfy_bindings = True  # type: ignore[attr-defined]
        free_memory._dit_original = original  # type: ignore[attr-defined]
        return free_memory

    # -- 无执行上下文时的保护 -----------------------------------------------

    def _guard_no_context(self, requested, api: str) -> None:
        """ctx 为 None 时：unowned 原样委托，owned 一律拒绝。"""
        owned = []
        for patcher in requested:
            owner = self.registry.resolve(patcher)
            if owner is not None:
                owned.append(owner.model_id)
        if not owned:
            return
        self._stats["refused"] += 1
        message = ("%s 在没有执行上下文（可能是第三方 helper 线程）时请求了 "
                   "owned 模型 %s：隐含 load 未被授权"
                   % (api, sorted(set(owned))))
        self._refusals.append({
            "operation_id": None,
            "phase": None,
            "kind": "no_context_owned_load",
            "message": message,
        })
        self.log("[comfy-bindings] REFUSE no-context/%s: %s" % (api, message))
        raise ImplicitLoadRefused(message)

    def _guard_no_context_eviction(self, keep_loaded, device: Any) -> None:
        """ctx 为 None 时：free_memory 不得驱逐 owned 模型。"""
        doomed = self._doomed_owned(keep_loaded)
        if not doomed:
            return
        self._stats["refused"] += 1
        message = ("free_memory(device=%s) 在没有执行上下文时会驱逐 owned 模型 "
                   "%s：拒绝隐式 offload" % (_device_str(device), sorted(doomed)))
        self._refusals.append({
            "operation_id": None,
            "phase": None,
            "kind": "no_context_owned_eviction",
            "message": message,
        })
        self.log("[comfy-bindings] REFUSE no-context/free_memory: %s" % message)
        raise ImplicitLoadRefused(message)

    def _doomed_owned(self, keep_loaded) -> list[str]:
        """current_loaded_models 中会被驱逐的 owned model_id。"""
        mm = self._record.module
        keep_ids = {id(x) for x in (keep_loaded or ())}
        doomed: list[str] = []
        for loaded in list(getattr(mm, "current_loaded_models", [])):
            if id(loaded) in keep_ids:
                continue
            patcher = getattr(loaded, "model", None)
            owner = (self.registry.resolve(patcher)
                     if patcher is not None else None)
            if owner is not None:
                doomed.append(owner.model_id)
        return sorted(set(doomed))

    # -- 分类与校验 ---------------------------------------------------------

    def _classify(self, ctx: ExecutionContext, requested: Sequence[Any]):
        """-> list[(patcher, owner|None)]。"""
        return [(patcher, self.registry.resolve(patcher))
                for patcher in requested]

    def _verify_resident(self, ctx: ExecutionContext, classified) -> None:
        """严格 phase：只校验**声明与 owner unit 的交集**。

        之前要求 owner 的每个 unit 都被声明，导致"DiT.refiner 还在 CPU、
        而这次 sample 只用 blocks/heads"这种合法声明被整体拒绝。现在：

        * required = declared_units ∩ owner.units —— 只有这些必须已驻留；
        * declared 里有、owner 里没有的 unit 记入 `unknown` 并拒绝
          （声明了一个不存在的 unit 说明调用方对模型结构的假设是错的）；
        * required 为空 → 该模型完全没被本次 operation 覆盖，拒绝；
        * 未声明的 unit 不做设备断言，但会在 `execution()` 退出时审核
          "它们没有被偷偷改动"（见 `_audit_exit`）。
        """
        mm = self._record.module
        for patcher, owner in classified:
            if owner is None:
                self._refuse(
                    ctx, "unowned_model",
                    "模型 %r 不属于本服务，%s 阶段不允许其参与"
                    % (_label(patcher), ctx.phase))
                raise UnownedModelError(
                    "strict phase %s: unowned model %s"
                    % (ctx.phase, _label(patcher)))
            units = owner.devices()
            declared = ctx.declared_units
            if declared is None:
                required = set(units)
            else:
                required = set(declared) & set(units)
                unknown = set(declared) - set(units)
                if unknown:
                    self._refuse(
                        ctx, "unknown_units",
                        "%s 声明了不存在的 unit: %s（实际有 %d 个）"
                        % (owner.model_id, sorted(unknown), len(units)))
                    raise MissingUnits(
                        "%s 声明了 owner 上不存在的 unit" % (owner.model_id,),
                        unknown=sorted(unknown))
                if not required:
                    self._refuse(
                        ctx, "missing_units",
                        "%s 的 unit 与本次 operation 声明无交集（声明 %s）"
                        % (owner.model_id, sorted(declared)))
                    raise MissingUnits(
                        "%s: 本次 operation 未声明该模型任何 unit"
                        % (owner.model_id,), missing=sorted(units))
            booking = _find_booking(mm, patcher, owner)
            if booking is None:
                self._refuse(
                    ctx, "missing_residence",
                    "%s 未在 current_loaded_models 登记，拒绝隐式加载"
                    % (owner.model_id,))
                raise MissingResidence(
                    "%s 没有 LoadedModel 登记（不会伪造簿记）；"
                    "请先显式 ram2vram" % (owner.model_id,))
            subset = {u: units[u] for u in units if u in required}
            self._assert_expected_devices(ctx, owner, subset)
            booking_device = _device_str(getattr(booking, "device", "unknown"))
            if booking_device != owner.load_device:
                self._refuse(
                    ctx, "residence_device",
                    "%s 台账设备 %s 与声明 %s 不符"
                    % (owner.model_id, booking_device, owner.load_device))
                raise MissingResidence(
                    "%s 台账设备 %s != %s"
                    % (owner.model_id, booking_device, owner.load_device))
            try:
                booking.currently_used = True
            except AttributeError:  # pragma: no cover - 真 LoadedModel 可写
                pass
            self._stats["bookkeeping_asserts"] += 1

    def _assert_expected_devices(self, ctx: ExecutionContext,
                                 owner: OwnedModel,
                                 units: Mapping[str, str]) -> None:
        for unit, actual in units.items():
            expected = ctx.expected_device(unit)
            if expected is None:
                expected = owner.load_device
            if _device_str(actual) != _device_str(expected):
                self._refuse(
                    ctx, "unexpected_weight_mutation",
                    "%s/%s 设备 %s != 期望 %s"
                    % (owner.model_id, unit, actual, expected))
                raise UnexpectedWeightMutation(
                    "%s/%s device %s, expected %s"
                    % (owner.model_id, unit, actual, expected))

    def _assert_cpu_only(self, ctx: ExecutionContext, classified) -> None:
        for patcher, owner in classified:
            label = owner.model_id if owner is not None else _label(patcher)
            if owner is None and not ctx.allow_unregistered:
                self._refuse(
                    ctx, "unregistered_construct",
                    "construct_cpu 出现未登记对象 %r，且未声明 "
                    "allow_unregistered" % (_label(patcher),))
                raise UnownedModelError(
                    "construct_cpu: unregistered model %s"
                    % (_label(patcher),))
            # 无论是否登记：构造阶段只允许 CPU 物化，登记本身不是上卡许可。
            target = _device_str(getattr(
                patcher, "load_device",
                owner.load_device if owner is not None else "unknown"))
            if not _is_cpu(target):
                self._refuse(
                    ctx, "implicit_load",
                    "%s 的 load_device=%s，construct_cpu 阶段拒绝上卡"
                    % (label, target))
                raise ImplicitLoadRefused(
                    "%s: construct_cpu 只能物化到 CPU，实际 load_device=%s"
                    % (label, target))

    def _assert_move_allowed(self, ctx: ExecutionContext, classified) -> None:
        """explicit_move/recovery 的前置授权。

        搬动集合由原函数决定，所以这里只校验"哪个模型被授权"，被搬的 unit
        在事后按 declared_units 复核（见 _delegate）。
        """
        for patcher, owner in classified:
            if owner is None:
                if not ctx.allow_unrelated:
                    self._refuse(
                        ctx, "unowned_model",
                        "%s 阶段出现非本服务模型 %r"
                        % (ctx.phase, _label(patcher)))
                    raise UnownedModelError(
                        "%s: unowned model %s"
                        % (ctx.phase, _label(patcher)))
                continue
            if owner.model_id not in ctx.allowed_models:
                self._refuse(
                    ctx, "model_not_allowed",
                    "%s 阶段未授权移动 %s（allowed=%s）"
                    % (ctx.phase, owner.model_id,
                       sorted(ctx.allowed_models)))
                raise ImplicitLoadRefused(
                    "%s 未列入 allowed_models" % (owner.model_id,))

    def _refuse_eviction(self, ctx: ExecutionContext, keep_loaded) -> None:
        mm = self._record.module
        keep_ids = {id(x) for x in (keep_loaded or ())}
        doomed = []
        for loaded in list(getattr(mm, "current_loaded_models", [])):
            if id(loaded) in keep_ids:
                continue
            patcher = getattr(loaded, "model", None)
            owner = self.registry.resolve(patcher) if patcher is not None else None
            doomed.append(owner.model_id if owner is not None else _label(patcher))
        if doomed:
            self._refuse(
                ctx, "implicit_offload",
                "%s 阶段 free_memory 会驱逐模型: %s"
                % (ctx.phase, sorted(set(doomed))))
            raise ImplicitLoadRefused(
                "%s: free_memory 会隐式卸载 %s"
                % (ctx.phase, sorted(set(doomed))))

    # -- 委托 + 事后校验 ----------------------------------------------------

    def _delegate(self, ctx, original, requested, classified, kwargs):
        """委托原函数 + 事后回读。

        原函数抛错（典型：CUDA OOM）时**也要回读**：部分加载可能已经发生，
        状态已经不是调用前的样子。这里不做"默默回滚"（绑定层不持有权重，
        回滚会掩盖真实资源状态），而是把实际 devices 写进异常 details，
        再原样抛出原异常（OOM 的语义与重试逻辑必须由调用方保留）。
        """
        mm = self._record.module
        before = self.registry.snapshot_all()
        self._stats["delegated"] += 1
        try:
            result = original(requested, **kwargs)
        except BaseException as error:
            after = self._snapshot_safe()
            self._report_partial(before, after, error, ctx)
            raise
        after = self.registry.snapshot_all()

        moved = {mid: (before[mid], after[mid])
                 for mid in before
                 if mid in after and before[mid] != after[mid]}
        allowed = set(ctx.allowed_models)
        # 本操作显式请求的 owned 模型允许变化；其余必须不变。
        permitted = set()
        for patcher, owner in classified:
            if owner is not None:
                permitted.add(owner.model_id)
        for model_id, (old, new) in moved.items():
            if model_id not in permitted:
                self._refuse(
                    ctx, "unexpected_weight_mutation",
                    "%s 在 %s 期间设备变化 %s -> %s"
                    % (model_id, ctx.phase, old, new))
                raise UnexpectedWeightMutation(
                    "%s: %s -> %s（未在本次 operation 授权）"
                    % (model_id, old, new),
                    moved=_jsonable({model_id: {"before": old, "after": new}}))
            if ctx.phase == PHASE_CONSTRUCT_CPU:
                off = [u for u, dev in new.items() if not _is_cpu(dev)]
                if off:
                    self._refuse(
                        ctx, "implicit_load",
                        "construct_cpu 后 %s 的 unit 上卡: %s"
                        % (model_id, sorted(off)))
                    raise ImplicitLoadRefused(
                        "%s construct_cpu 阶段被上卡: %s"
                        % (model_id, sorted(off)))
                continue
            if model_id not in allowed:
                self._refuse(
                    ctx, "model_not_allowed",
                    "%s 未列入 allowed_models 却被搬动" % (model_id,))
                raise ImplicitLoadRefused(
                    "%s 未列入 allowed_models" % (model_id,))
            changed = [u for u in new
                       if old.get(u) != new.get(u)
                       and not ctx.unit_declared(u)]
            if changed:
                self._refuse(
                    ctx, "missing_units",
                    "%s 搬动了未声明的 unit: %s"
                    % (model_id, sorted(changed)))
                raise MissingUnits(
                    "%s 搬动未声明 unit %s" % (model_id, sorted(changed)),
                    missing=changed)

        # 台账回读：请求的 owned 模型必须真的登记为 load_device。
        for patcher, owner in classified:
            if owner is None:
                continue
            booking = _find_booking(mm, patcher, owner)
            if booking is None:
                self._refuse(
                    ctx, "bookkeeping_missing",
                    "%s 委托原函数后仍无 LoadedModel" % (owner.model_id,))
                raise BookkeepingError(
                    "%s 委托后台账无 LoadedModel（原函数可能早退）"
                    % (owner.model_id,))
            booking_device = _device_str(getattr(booking, "device", "unknown"))
            target = _device_str(getattr(patcher, "load_device",
                                         owner.load_device))
            if booking_device != target:
                self._refuse(
                    ctx, "bookkeeping_device",
                    "%s 台账 device=%s 与 load_device=%s 不符"
                    % (owner.model_id, booking_device, target))
                raise BookkeepingError(
                    "%s 台账 device=%s != %s"
                    % (owner.model_id, booking_device, target))
            self._stats["bookkeeping_asserts"] += 1
        return result

    def _snapshot_safe(self) -> dict[str, dict[str, str]]:
        try:
            return self.registry.snapshot_all()
        except Exception as error:  # noqa: BLE001 - probe 抛错不能掩盖原异常
            self.log("[comfy-bindings] 事后快照失败: %s" % (error,))
            return {}

    def _report_partial(self, before, after, error: BaseException,
                        ctx: ExecutionContext) -> None:
        """原函数抛错后如实记录实际状态（不装作什么都没发生）。"""
        moved = {}
        for model_id, new in (after or {}).items():
            old = (before or {}).get(model_id)
            if old is None or old == new:
                continue
            moved[model_id] = {
                "before": _jsonable(old), "after": _jsonable(new)}
        entry = {
            "operation_id": ctx.operation_id,
            "phase": ctx.phase,
            "kind": "delegate_failed",
            "error": "%s: %s" % (type(error).__name__, error),
            "partial_moves": moved,
        }
        self._refusals.append(entry)
        self._stats.setdefault("partial_failures", 0)
        self._stats["partial_failures"] += 1
        self.log("[comfy-bindings] 原函数抛错 %s；实际状态变化=%s"
                 % (entry["error"], moved or "无"))
        # 把实际状态挂到原异常上，供上层回执/诊断读取（不改异常类型）
        try:
            setattr(error, "dit_comfy_partial_state", moved)
        except Exception:  # noqa: BLE001 - 某些内建异常不允许设属性
            pass

    def _refuse(self, ctx: ExecutionContext, kind: str, message: str) -> None:
        self._stats["refused"] += 1
        self._refusals.append({
            "operation_id": ctx.operation_id,
            "phase": ctx.phase,
            "kind": kind,
            "message": message,
        })
        self.log("[comfy-bindings] REFUSE %s/%s: %s"
                 % (ctx.operation_id, kind, message))

    # -- 审计与状态 ---------------------------------------------------------

    def audit_aliases(self) -> list[dict[str, str]]:
        """找出提前捕获了裸原函数的模块（只报告，不修改他人）。"""
        if self._record is None:
            return []
        originals = {id(fn): name for name, fn in self._record.originals.items()}
        found: list[dict[str, str]] = []
        for mod_name, module in list(sys.modules.items()):
            if module is None or not hasattr(module, "__dict__"):
                continue
            for attr, value in list(vars(module).items()):
                name = originals.get(id(value))
                if name is None:
                    continue
                found.append({"module": mod_name, "attribute": attr,
                              "function": name})
        return found

    def state(self) -> dict[str, Any]:
        record = self._record
        ctx = self.context
        return {
            "installed": record is not None,
            "wrapped": ([] if record is None
                        else sorted(record.wrappers)),
            "signatures": ({} if record is None else
                           {k: v.as_dict()
                            for k, v in record.signatures.items()}),
            "allow_unknown_signature": (None if record is None
                                       else record.allow_unknown_signature),
            "context": None if ctx is None else ctx.as_dict(),
            "owned_models": {e.model_id: {"generation": e.generation,
                                          "units": e.devices()}
                             for e in self.registry.entries()},
            "stats": dict(self._stats),
            "refusals": list(self._refusals)[-16:],
        }

    def status_report(self) -> dict[str, Any]:
        """/api/status 用的稳定小结（不含 Tensor/对象引用）。"""
        st = self.state()
        return {
            "installed": st["installed"],
            "signatures": st["signatures"],
            "context_phase": (None if st["context"] is None
                              else st["context"]["phase"]),
            "owned": sorted(st["owned_models"]),
            "stats": st["stats"],
            "alias_count": len(self.audit_aliases()),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _label(obj: Any) -> str:
    inner = getattr(obj, "model", None)
    name = type(inner if inner is not None else obj).__name__
    return "%s@%x" % (name, id(obj))


def _is_cpu(device: str) -> bool:
    return str(device).split(":", 1)[0].strip().lower() == "cpu"


def _jsonable(value: Any) -> Any:
    """把嵌套结构里的非 JSON 值转成字符串/原生类型。

    `details` 会直接进 operation 回执，所以这里必须保证：
    不返回 Tensor、不返回 bound method / function 对象本身。
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        return "<callable %s>" % (getattr(value, "__qualname__",
                                          type(value).__name__),)
    return str(value)


def _details_json(value: dict[str, Any]) -> dict[str, Any]:
    """`details` 数据属性的唯一出口：纯 JSON dict，绝不含 Tensor/method。

    `OperationError.from_exception` 会直接读属性，不做二次转换；所以这里
    必须保证（a）返回值是 dict；（b）所有叶子是 JSON 原生类型。
    """
    return {str(k): _jsonable(v) for k, v in (value or {}).items()}


def _find_booking(mm: types.ModuleType, patcher: Any, owner: OwnedModel):
    """在 current_loaded_models 里找该 patcher 的 LoadedModel（不新建）。"""
    entries = getattr(mm, "current_loaded_models", None)
    if entries is None:
        raise BindingsError("model_management 缺少 current_loaded_models")
    inner = getattr(patcher, "model", None)
    for loaded in list(entries):
        model = getattr(loaded, "model", None)
        if model is patcher:
            return loaded
        if inner is not None and model is not None:
            if getattr(model, "model", None) is inner:
                return loaded
    return None
