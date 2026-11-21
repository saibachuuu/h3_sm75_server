"""P0 自检：dit_svc.comfy_bindings 的所有权/簿记语义（纯 CPU，无 comfy 依赖）。

跑法（任意 torch 3.10/2.8 CPU 即可，不需要 ComfyUI）：

  python tests/_selftest_comfy_bindings.py

覆盖：签名记录与版本拒绝、install/uninstall 幂等、上下文外委托、resident
不调用原函数、未驻留不伪造簿记、unowned/missing units/设备不符拒绝、
free_memory 隐式驱逐拒绝、construct_cpu 上卡拒绝、explicit_move 授权与
事后设备漂移检测、clone(parent) 归并、alias 审计。

注意：这只是 fake ComfyUI 的行为契约测试；真实 GPU 上"不搬权重"仍待主代理
在远端跑 real probe，本脚本不宣称 GPU 已通过。
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc import comfy_bindings as cb  # noqa: E402


# ---------------------------------------------------------------------------
# fake comfy.model_management
# ---------------------------------------------------------------------------


class FakeLoaded:
    def __init__(self, patcher):
        self.model = patcher
        self.device = patcher.load_device
        self.currently_used = True

    def __repr__(self):
        return "<FakeLoaded %s>" % (self.device,)


class FakeModel:
    pass


class FakePatcher:
    def __init__(self, load_device="cuda:0", offload_device="cpu",
                 units=("blocks.0", "blocks.1"), parent=None, model=None):
        self.model = model if model is not None else FakeModel()
        self.load_device = load_device
        self.offload_device = offload_device
        self.parent = parent
        self.units = {u: "cpu" for u in units}
        self.fail_booking = False

    def clone(self):
        # 真 ModelPatcher.clone：共享同一 model 对象，parent 指向源
        child = FakePatcher(self.load_device, self.offload_device,
                            tuple(self.units), parent=self, model=self.model)
        child.units = dict(self.units)
        return child


def make_fake_mm():
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []
    calls = {"load": 0, "free": 0, "load_requests": []}

    def load_models_gpu(models, memory_required=0, force_patch_weights=False,
                        minimum_memory_required=None, force_full_load=False):
        calls["load"] += 1
        calls["load_requests"].append([id(m) for m in models])
        for patcher in models:
            target = patcher.get("load_device_override") \
                if isinstance(patcher, dict) else patcher.load_device
            patcher.units = {u: target for u in patcher.units}
            if patcher.fail_booking:
                continue
            if not any(x.model is patcher for x in mm.current_loaded_models):
                mm.current_loaded_models.insert(0, FakeLoaded(patcher))
            side_effect = getattr(patcher, "after_load", None)
            if side_effect is not None:
                side_effect()

    def load_model_gpu(model):
        return load_models_gpu([model])

    def free_memory(memory_required, device, keep_loaded=[], for_dynamic=False,
                    pins_required=0, ram_required=0):
        calls["free"] += 1
        keep = {id(x) for x in keep_loaded}
        mm.current_loaded_models[:] = [
            x for x in mm.current_loaded_models if id(x) in keep]

    mm.load_models_gpu = load_models_gpu
    mm.load_model_gpu = load_model_gpu
    mm.free_memory = free_memory
    return mm, calls


def fresh():
    mm, calls = make_fake_mm()
    registry = cb.OwnedModelRegistry()
    bindings = cb.ComfyBindings(registry)
    bindings.install(mm, allow_unknown_signature=True)
    return mm, calls, registry, bindings


def expect(exc_type, fn, label):
    try:
        fn()
    except exc_type as e:
        print("   OK %s: %s" % (exc_type.__name__, str(e)[:96]))
        return e
    except Exception as e:  # noqa: BLE001
        print("   FAIL %s: 抛了 %s: %s" % (label, type(e).__name__, e))
        raise SystemExit(1)
    print("   FAIL %s: 没抛错" % (label,))
    raise SystemExit(1)


def own(registry, patcher, model_id="dit", generation=None):
    return registry.register(
        patcher, model_id, patcher.units, generation=generation,
        probe=lambda p=patcher: p.units)


# ---------------------------------------------------------------------------
print("== 1. 安装：记录原函数与签名，重复安装 no-op ==")
mm, calls, registry, bindings = fresh()
state = bindings.state()
assert state["installed"] and state["wrapped"] == ["free_memory", "load_models_gpu"]
sig = state["signatures"]["load_models_gpu"]
assert sig["params"] == ["models", "memory_required", "force_patch_weights",
                         "minimum_memory_required", "force_full_load"], sig
assert sig["code_sha1"], "应有字节码摘要"
assert mm.load_models_gpu._dit_comfy_bindings is True
assert mm.load_models_gpu._dit_original is not None
state2 = bindings.install(mm, allow_unknown_signature=True)
assert state2["wrapped"] == state["wrapped"]
print("   OK params=%s sha1=%s" % (sig["params"], sig["code_sha1"][:12]))

print("== 2. 真实签名不匹配必须拒绝安装 ==")
mm2, _ = make_fake_mm()


def real_sig(mm_):
    def load_models_gpu(models, memory_required=0, extra_flag=None):
        return None

    def free_memory(memory_required, device):
        return None

    mm_.load_models_gpu = load_models_gpu
    mm_.free_memory = free_memory
    return mm_


expect(cb.UnsupportedComfyVersion,
       lambda: cb.ComfyBindings().install(real_sig(mm2)),
       "signature mismatch")
strict = cb.ComfyBindings()
strict.install(real_sig(mm2), allow_unknown_signature=True)
assert strict.state()["allow_unknown_signature"] is True
print("   OK allow_unknown_signature 可显式强装（并记录）")

print("== 3. 上下文外委托原函数（不影响非服务调用） ==")
mm, calls, registry, bindings = fresh()
outside = FakePatcher(units=("blocks.0",))
before = calls["load"]
mm.load_models_gpu([outside])
assert calls["load"] == before + 1, "上下文外必须委托原函数"
print("   OK 无上下文 -> 原函数解析=%s" % outside.units)

print("== 4. resident_infer：已驻留只校验、绝不调用原函数 ==")
mm.load_models_gpu([outside])          # 先让它登记为已驻留（cuda:0）
own(registry, outside, "dit")
before = calls["load"]
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-resident",
                        declared_units=("blocks.0",),
                        expected_devices={"blocks.0": "cuda:0"}):
    mm.load_models_gpu([outside])
assert calls["load"] == before, "resident 路径不得调用原函数"
assert mm.current_loaded_models[0].currently_used is True
assert bindings.state()["stats"]["bookkeeping_asserts"] >= 1
print("   OK 原函数调用次数 %d（未增加），台账命中" % calls["load"])

print("== 5. resident_infer：未驻留不得伪造簿记 ==")
idle = FakePatcher(units=("blocks.5",))
own(registry, idle, "dit_alt")
before = calls["load"]
try:
    with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-missing",
                            declared_units=("blocks.5",)):
        expect(cb.MissingResidence,
               lambda: mm.load_models_gpu([idle]), "missing residence")
finally:
    pass
assert calls["load"] == before
assert not any(getattr(x, "model", None) is idle
               for x in mm.current_loaded_models), "不得插入假条目"
print("   OK 台账无新条目，原函数未调用")

print("== 6. resident_infer：unowned / 缺声明 / 设备不符 ==")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-unowned"):
    expect(cb.UnownedModelError,
           lambda: mm.load_models_gpu([FakePatcher()]), "unowned")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-units",
                        declared_units=("blocks.1",)):
    expect(cb.MissingUnits,
           lambda: mm.load_models_gpu([outside]), "undeclared unit")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-dev",
                        declared_units=("blocks.0",),
                        expected_devices={"blocks.0": "cpu"}):
    expect(cb.UnexpectedWeightMutation,
           lambda: mm.load_models_gpu([outside]), "device mismatch")

print("== 7. free_memory 在严格阶段拒绝隐式驱逐 ==")
other = FakePatcher(units=("blocks.9",))
mm.load_models_gpu([other])
own(registry, other, "te")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-evict"):
    expect(cb.ImplicitLoadRefused,
           lambda: mm.free_memory(1 << 30, "cuda:0"), "implicit eviction")
before = calls["free"]
owned_ids = {id(outside), id(other)}
keep = [x for x in mm.current_loaded_models
        if id(getattr(x, "model", None)) in owned_ids]
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-keep"):
    mm.free_memory(1 << 30, "cuda:0", keep_loaded=keep)
assert calls["free"] == before + 1
print("   OK 被驱逐的 owned 模型会拒绝；keep_loaded 全覆盖才允许")

print("== 8. construct_cpu：load_device 非 CPU 一律拒绝 ==")
construct = FakePatcher(load_device="cuda:0", units=("blocks.2",))
with bindings.execution(cb.PHASE_CONSTRUCT_CPU, "op-construct"):
    expect(cb.UnownedModelError,
           lambda: mm.load_models_gpu([construct]),
           "unregistered construct without allow_unregistered")
with bindings.execution(cb.PHASE_CONSTRUCT_CPU, "op-construct2",
                        allow_unregistered=True):
    expect(cb.ImplicitLoadRefused,
           lambda: mm.load_models_gpu([construct]), "implicit gpu construct")
cpu_construct = FakePatcher(load_device="cpu", units=("blocks.3",))
before = calls["load"]
with bindings.execution(cb.PHASE_CONSTRUCT_CPU, "op-construct3",
                        allow_unregistered=True):
    mm.load_models_gpu([cpu_construct])
assert calls["load"] == before + 1
print("   OK CPU 构造委托原函数；上卡构造拒绝")

print("== 9. explicit_move：授权、事后设备复核、越权搬动检测 ==")
move = FakePatcher(load_device="cuda:0", units=("blocks.4", "blocks.5"))
mm.load_models_gpu([move])            # 先登记（此时 units 全 cuda）
move.units = {u: "cpu" for u in move.units}
mm.current_loaded_models[:] = [x for x in mm.current_loaded_models
                               if getattr(x, "model", None) is not move]
own(registry, move, "dit_move")
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-move-unallowed"):
    expect(cb.ImplicitLoadRefused,
           lambda: mm.load_models_gpu([move]), "not in allowed_models")
before = calls["load"]
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-move",
                        allowed_models=("dit_move",),
                        declared_units=("blocks.4", "blocks.5")):
    mm.load_models_gpu([move])
assert calls["load"] == before + 1
assert move.units["blocks.4"] == "cuda:0"
print("   OK 授权后委托原函数并复核台账")

print("== 10. explicit_move：未声明的 unit 被搬 -> MissingUnits ==")
move2 = FakePatcher(load_device="cuda:0", units=("blocks.6",))
own(registry, move2, "dit_move2")
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-move-units",
                        allowed_models=("dit_move2",),
                        declared_units=()):
    expect(cb.MissingUnits,
           lambda: mm.load_models_gpu([move2]), "undeclared moved unit")

print("== 11. explicit_move：原函数早退不登记 -> BookkeepingError ==")
ghost = FakePatcher(load_device="cuda:0", units=("blocks.7",))
ghost.fail_booking = True
own(registry, ghost, "dit_ghost")
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-ghost",
                        allowed_models=("dit_ghost",),
                        declared_units=("blocks.7",)):
    expect(cb.BookkeepingError,
           lambda: mm.load_models_gpu([ghost]), "no booking after delegate")
assert not any(getattr(x, "model", None) is ghost
               for x in mm.current_loaded_models)

print("== 12. explicit_move：偷搬另一个 owned 模型 -> UnexpectedWeightMutation ==")
victim = FakePatcher(load_device="cuda:0", units=("blocks.8",))
mm.load_models_gpu([victim])
own(registry, victim, "vae")


def _steal():
    """模拟原函数顺手把别家的模型下卡。"""
    victim.units = {u: "cpu" for u in victim.units}


move.after_load = _steal
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-steal",
                        allowed_models=("dit_move",),
                        declared_units=("blocks.4", "blocks.5")):
    expect(cb.UnexpectedWeightMutation,
           lambda: mm.load_models_gpu([move]),
           "other owned model mutated")
del move.after_load

print("== 13. clone(parent 链/共享 model)归并到同一 owner ==")
base = FakePatcher(units=("blocks.0", "blocks.1"))
own(registry, base, "dit_clone")
shifted = base.clone()                # SigmaShift 形态：共享 model，parent 指向 base
assert registry.resolve(shifted) is registry.get("dit_clone")
assert id(shifted) in registry.get("dit_clone").patcher_ids
shared = FakePatcher(units=("blocks.0",), model=base.model)
assert registry.resolve(shared).model_id == "dit_clone"
print("   OK clone/shared-model 均解析为 dit_clone")

print("== 14. 执行上下文：嵌套拒绝、活动中不可还原 ==")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-nest"):
    expect(cb.BindingsError,
           lambda: bindings.execution(cb.PHASE_RESIDENT_INFER,
                                      "op-inner").__enter__(),
           "nested context")
    expect(cb.BindingsError, lambda: bindings.uninstall(), "uninstall active")
assert bindings.state()["installed"]

print("== 15. alias 审计：提前捕获裸函数的模块会被列出 ==")
alias_mod = types.ModuleType("fake_third_party")
alias_mod.load_models_gpu = bindings._record.originals["load_models_gpu"]
sys.modules["fake_third_party"] = alias_mod
try:
    hits = [h for h in bindings.audit_aliases()
            if h["module"] == "fake_third_party"]
    assert hits and hits[0]["function"] == "load_models_gpu", hits
    print("   OK 捕获到 %s" % hits[0])
finally:
    sys.modules.pop("fake_third_party", None)

print("== 16. uninstall 还原原函数，并检测第三方替换 ==")
mm, calls, registry, bindings = fresh()
orig = bindings._record.originals["load_models_gpu"]
bindings.uninstall()
assert mm.load_models_gpu is orig
assert bindings.state()["installed"] is False
bindings.install(mm, allow_unknown_signature=True)
mm.load_models_gpu = lambda *a, **k: None      # 第三方覆盖
expect(cb.BindingsError, lambda: bindings.uninstall(), "third-party override")
assert bindings._record is None, "冲突时也要放下安装记录，避免半挂状态"

print("== 17. 状态报告可 JSON 化，且不含对象引用 ==")
mm, calls, registry, bindings = fresh()
own(registry, FakePatcher(units=("blocks.0",)), "dit")
import json
json.dumps(bindings.status_report(), sort_keys=True)
json.dumps(bindings.state(), sort_keys=True)
print("   OK status_report:\n   %s"
      % json.dumps(bindings.status_report(), ensure_ascii=False, sort_keys=True))

print("== 18. 无执行上下文（helper 线程）：owned 拒绝、unowned 原样委托 ==")
import threading  # noqa: E402

mm, calls, registry, bindings = fresh()
owned_ctx = FakePatcher(units=("blocks.0",))
mm.load_models_gpu([owned_ctx])                  # 先驻留
own(registry, owned_ctx, "dit_ctx")
stranger = FakePatcher(units=("blocks.7",))
before = calls["load"]
expect(cb.ImplicitLoadRefused,
       lambda: mm.load_models_gpu([owned_ctx]), "no-context owned load")
assert calls["load"] == before, "拒绝时不得调用原函数"
mm.load_models_gpu([stranger])
assert calls["load"] == before + 1, "unowned 在无上下文时必须正常委托"
print("   OK owned 被拒（原函数未调用）；unowned 正常委托")

# 真实新线程：ContextVar 不继承 → 同一 owned 模型也会被拒
result = {}


def _worker():
    # 屏障只用来确认"主线程 context 存活时子线程已经在跑"，
    # 但 Barrier 需要双方都到齐；子线程不等待主线程，避免自锁。
    try:
        mm.load_models_gpu([owned_ctx])
        result["outcome"] = "delegated"
    except cb.ImplicitLoadRefused as error:
        result["outcome"] = "refused"
        result["message"] = str(error)


barrier = threading.Barrier(2)
t = threading.Thread(target=_worker)
before = calls["load"]
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-thread",
                        declared_units=("blocks.0",)):
    assert bindings.context is not None, "主线程应持有 context"
    t.start()
    t.join(timeout=10)           # 主线程仍持有 context，子线程没有
    assert bindings.context is not None, "join 期间主线程 context 仍在"
assert result.get("outcome") == "refused", result
assert calls["load"] == before, "子线程拒绝时原函数不得被调用"
print("   OK 新线程绕过：主线程有 context 时子线程仍被拒 (%s)"
      % result.get("outcome"))

print("== 19. 无上下文的 free_memory 也不能驱逐 owned ==")
keep = [x for x in mm.current_loaded_models
        if getattr(x, "model", None) is owned_ctx]
before = calls["free"]
expect(cb.ImplicitLoadRefused,
       lambda: mm.free_memory(1 << 30, "cuda:0"), "no-context eviction")
assert calls["free"] == before
mm.free_memory(1 << 30, "cuda:0", keep_loaded=keep)
print("   OK 无上下文驱逐 owned 被拒；keep_loaded 覆盖后允许")

print("== 20. 粒度：只声明部分 unit 合法（refiner 留 CPU） ==")
mm, calls, registry, bindings = fresh()
split = FakePatcher(units=("blocks.0", "refiner"))
mm.load_models_gpu([split])
own(registry, split, "dit_split")
with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-partial",
                        declared_units=("blocks.0",)):
    mm.load_models_gpu([split])       # refiner 未声明，但不再整单拒绝
print("   OK 声明 blocks.0 即可通过（不再要求 owner 全部 unit）")

with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-unknown-unit",
                        declared_units=("blocks.0", "not_a_unit")):
    expect(cb.MissingUnits,
           lambda: mm.load_models_gpu([split]), "declared unknown unit")

print("== 21. 未声明 unit 在 operation 期间被搬动 -> 退出审核报警 ==")
assert split.units["refiner"] == "cuda:0", split.units
try:
    with bindings.execution(cb.PHASE_RESIDENT_INFER, "op-exit-audit",
                            declared_units=("blocks.0",)):
        split.units["refiner"] = "cpu"        # 偷偷搬未声明 unit
    print("   FAIL 未声明 unit 被搬动却没报错")
    raise SystemExit(1)
except cb.UnexpectedWeightMutation as error:
    assert "refiner" in error.details["moved"].get("dit_split", {}), error.details
    print("   OK 退出审核抛出 UnexpectedWeightMutation（refiner 未声明）")
split.units["refiner"] = "cuda:0"

print("== 22. 原函数抛错（OOM）：回读实际状态并保留原异常 ==")
mm, calls, registry, bindings = fresh()
half = FakePatcher(units=("blocks.0", "blocks.1"))
own(registry, half, "dit_oom")
half.load_device = "cuda:0"


def _partial_then_boom(models, **kwargs):
    models[0].units["blocks.0"] = "cuda:0"      # 部分上卡
    raise RuntimeError("CUDA out of memory")


mm.load_models_gpu = _partial_then_boom
bindings._record.wrappers["load_models_gpu"] = bindings._make_load_models_gpu(
    {"load_models_gpu": _partial_then_boom,
     "free_memory": bindings._record.originals["free_memory"]})
setattr(mm, "load_models_gpu", bindings._record.wrappers["load_models_gpu"])
with bindings.execution(cb.PHASE_EXPLICIT_MOVE, "op-oom",
                        allowed_models=("dit_oom",),
                        declared_units=("blocks.0", "blocks.1")):
    try:
        mm.load_models_gpu([half])
    except RuntimeError as error:
        assert "out of memory" in str(error)
        partial = getattr(error, "dit_comfy_partial_state", {})
        assert "dit_oom" in partial, partial
        assert partial["dit_oom"]["after"]["blocks.0"] == "cuda:0", partial
        print("   OK 原异常保留，且 details 记录 blocks.0 已上卡: %s" % partial)
    else:
        print("   FAIL 没抛错")
        raise SystemExit(1)

print("== 23. 异常带 code/http_status，details 是数据属性（不是 bound method） ==")
import inspect as _inspect  # noqa: E402
import json  # noqa: E402
_instances = (cb.ImplicitLoadRefused("x"), cb.MissingResidence("y"),
              cb.MissingUnits("z", missing=["a"], unknown=["b"]),
              cb.UnownedModelError("u"), cb.UnexpectedWeightMutation(
                  "m", moved={"dit": {"blocks.0": {"before": "cpu",
                                                   "after": "cuda:0"}}}),
              cb.UnsupportedComfyVersion("v"), cb.OwnershipError("o"),
              cb.BookkeepingError("b"), cb.BindingsError("base"))
for exc in _instances:
    # 模拟 OperationError.from_exception：直接读属性，不调用
    assert isinstance(exc.details, dict), (type(exc).__name__, exc.details)
    assert not _inspect.ismethod(exc.details), type(exc).__name__
    assert not callable(exc.details), type(exc).__name__
    # 属性重复读取必须稳定且独立（每次新的 dict，可被回执层安全保留）
    assert exc.details == exc.details
    assert exc.details is not exc.details
    payload = exc.as_operation_error()
    assert payload["details"] == exc.details, type(exc).__name__
    assert payload["http_status"] in (409, 500, 503), payload
    json.dumps(payload, sort_keys=True)
assert cb.MissingUnits("z", missing=["a"]).details["missing"] == ["a"]
assert cb.UnexpectedWeightMutation(
    "m", moved=[("u", 1)]).details["moved"] == [["u", 1]]
assert cb.ImplicitLoadRefused("x").http_status == 409
assert cb.MissingResidence("y").http_status == 503
assert cb.UnsupportedComfyVersion("v").http_status == 500
print("   OK details 是 property(dict)；409/503/500 分类明确，JSON 可序列化")

print("== 24. details 里绝不出现 Tensor / method 对象 ==")


class _FakeTensorish:
    def __repr__(self):
        return "<Tensor>"


class _Holder:
    def method(self):
        return None


_holder = _Holder()
_tricky = cb.UnexpectedWeightMutation(
    "m", moved={"dit": {"blocks.0": {"before": _FakeTensorish(),
                                     "after": _holder.method}}})
_details = _tricky.details
json.dumps(_details, sort_keys=True)          # 必须能直接序列化
_flat = json.dumps(_details)
assert "<Tensor>" in _flat and "callable" in _flat, _flat
assert not any(callable(v) for v in json.loads(_flat).values())
print("   OK 非 JSON 叶子被转字符串，bound method 标记为 <callable ...>")

print("== 25. status()/state() 递归无 callable 或非 JSON 叶子 ==")


def _walk(value, path="", out=None):
    if out is None:
        out = []
    if callable(value):
        out.append("CALLABLE @ %s" % (path or "/"))
    elif isinstance(value, dict):
        for k, v in value.items():
            _walk(v, "%s/%s" % (path, k), out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk(v, "%s[%d]" % (path, i), out)
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        out.append("NON-JSON %s @ %s" % (type(value).__name__, path or "/"))
    return out


mm, calls, registry, bindings = fresh()
own(registry, FakePatcher(units=("blocks.0",)), "dit")
for _label, _payload in (("state", bindings.state()),
                         ("status_report", bindings.status_report())):
    json.dumps(_payload, sort_keys=True)
    _bad = _walk(_payload)
    assert not _bad, (_label, _bad)
print("   OK state / status_report 可直接 json.dumps，无 Tensor/method")

print()
print("ALL OK（fake ComfyUI 契约层；真实 GPU 不搬权重仍需远端 probe）")
