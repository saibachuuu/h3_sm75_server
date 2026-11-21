"""自检 P2：fake adapter 完整四动作 + partial purge/reload 闭环。

证明（fake 域内）：
1. disk2ram 只读所选 unit 的 keys（ FakeSource.accessed_keys 作证）；
2. 未选 unit 保持无存储占位（storage is None）；
3. partial purge 保留拓扑槽位编号；
4. 补回只读缺失 unit keys，既有驻留块 storage 身份不变；
5. 全量 purge 后骨架槽位仍在、registry 全 absent。

需要 torch（CPU）。CUDA 物理搬运本机不可验证：move 只改 logical_device，
见 docs/p0_weights.md。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

fails = []


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("returned False")
        print("  OK   %-46s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-46s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


import torch  # noqa: E402

from dit_svc import weights as W  # noqa: E402
from dit_svc.model_adapters import (FakeAdapter, FakeSkeleton,  # noqa: E402
                                    FakeSource, NoStoragePlaceholder,
                                    fake_disk2ram, fake_move, fake_purge)


def _env():
    torch.manual_seed(0)
    header = {}
    tensors = {}
    for i in range(4):
        for leaf in ("attn.qkv_proj.weight", "mlp.fc1.weight"):
            k = "blocks.%d.%s" % (i, leaf)
            header[k] = {"dtype": "F32", "shape": [16, 16]}
            tensors[k] = torch.randn(16, 16)
    for k, shape in (("token_refiner.blocks.0.attn.qkv_proj.weight", [8, 16]),
                     ("condition_proj.weight", [8, 16])):
        header[k] = {"dtype": "F32", "shape": shape}
        tensors[k] = torch.randn(*shape)
    cat = W.build_catalog_from_header("dit", header, "dit")
    reg = W.WeightRegistry()
    rec = W.recipe_from_header_dict("dit", "dit", "fake://h", header)
    reg.register_recipe(rec)
    reg.attach_catalog(cat)
    src = FakeSource(tensors)
    adapter = FakeAdapter(src, quant_units=["dit.blocks.0"],
                          view_units=["dit.blocks.2"])
    skel = adapter.build_skeleton(cat)
    return reg, adapter, skel, src, cat


def _ptr_of(skel, adapter, uid):
    h = adapter.unit_handle(skel, uid)
    assert not isinstance(h, NoStoragePlaceholder), uid
    t = h["tensors"]
    first = t[sorted(t.keys())[0]]
    inner = getattr(first, "_qdata", first)
    return int(inner.untyped_storage().data_ptr())


def t_selective_first_load():
    reg, adapter, skel, src, cat = _env()
    units = W.resolve_units(cat, "units", ["dit.blocks.1-2"])
    reg.check_transition("disk2ram", "dit", units)
    op = fake_disk2ram(reg, adapter, skel, "dit", units)
    assert op["completed"] == units, op
    selected = set(cat.keys_of(units))
    assert set(src.accessed_keys) == selected, (
        sorted(set(src.accessed_keys) - selected)[:3])
    for uid in cat.unit_ids():
        resident = adapter.is_resident(skel, uid)
        assert resident == (uid in units), uid
        if not resident:
            assert isinstance(adapter.unit_handle(skel, uid),
                              NoStoragePlaceholder)
    # 未选块无真实 storage：inventory 为空
    inv = W.storage_inventory(
        {"h": adapter.unit_handle(skel, "dit.blocks.0")}, owner="x")
    assert inv.total_cpu == 0, inv.to_summary()
    return "only %d keys read" % len(src.accessed_keys)


def t_partial_purge_reload_preserves():
    reg, adapter, skel, src, cat = _env()
    units = W.resolve_units(cat, "units", ["dit.blocks.0-3"])
    reg.check_transition("disk2ram", "dit", units)
    fake_disk2ram(reg, adapter, skel, "dit", units)
    n_slots = skel.num_slots()
    ptr2_before = _ptr_of(skel, adapter, "dit.blocks.2")
    # partial purge blocks.1（topology 槽位保留）
    reg.check_transition("purgefromram", "dit", ["dit.blocks.1"])
    fake_purge(reg, adapter, skel, "dit", ["dit.blocks.1"])
    assert skel.num_slots() == n_slots
    assert not adapter.is_resident(skel, "dit.blocks.1")
    assert adapter.is_resident(skel, "dit.blocks.2")
    st = reg.model_state("dit")
    assert st["state"] == "partial", st["state"]
    # 补回：只读缺失 unit keys；既有块 storage 身份不变
    src.reset_access_log()
    reg.check_transition("disk2ram", "dit", ["dit.blocks.1"])
    fake_disk2ram(reg, adapter, skel, "dit", ["dit.blocks.1"])
    assert set(src.accessed_keys) == set(
        cat.units["dit.blocks.1"].checkpoint_keys)
    assert _ptr_of(skel, adapter, "dit.blocks.2") == ptr2_before
    assert adapter.is_resident(skel, "dit.blocks.1")
    return "purge/reload closed, ptr stable"


def t_full_cycle_move_and_purge():
    reg, adapter, skel, src, cat = _env()
    units = W.resolve_units(cat, "units", ["dit.blocks.0-1"])
    reg.check_transition("disk2ram", "dit", units)
    fake_disk2ram(reg, adapter, skel, "dit", units)
    reg.check_transition("ram2vram", "dit", units)
    op = fake_move(reg, adapter, skel, "dit", units, "ram2vram", "cuda:0")
    assert op["completed"] == units, op
    reg.check_transition("vram2ram", "dit", ["dit.blocks.0"])
    fake_move(reg, adapter, skel, "dit", ["dit.blocks.0"], "vram2ram", "cpu")
    st = reg.model_state("dit")
    dev = {u["id"]: u["device"] for u in st["units"]}
    assert dev["dit.blocks.0"] == "cpu" and dev["dit.blocks.1"] == "cuda:0", dev
    try:
        reg.check_transition("purgefromram", "dit", ["dit.blocks.1"])
    except W.TransitionError:
        pass
    else:
        raise AssertionError("cuda purge not blocked")
    reg.check_transition("vram2ram", "dit", ["dit.blocks.1"])
    fake_move(reg, adapter, skel, "dit", ["dit.blocks.1"], "vram2ram", "cpu")
    reg.check_transition("purgefromram", "dit", units)
    fake_purge(reg, adapter, skel, "dit", units)
    st = reg.model_state("dit")
    assert st["state"] == "absent" and skel.num_slots() == len(cat.unit_ids())
    return "move/purge cycle ok"


def t_no_eager_full_load():
    """反例：全量 units 展开也逐 unit 物化，不存在一次性读全量 header 的路径。"""
    reg, adapter, skel, src, cat = _env()
    all_units = W.resolve_units(cat, "all")
    assert all_units == cat.unit_ids()
    src.reset_access_log()
    reg.check_transition("disk2ram", "dit", all_units[:1])
    fake_disk2ram(reg, adapter, skel, "dit", all_units[:1])
    assert set(src.accessed_keys) == set(
        cat.keys_of(all_units[:1])), src.accessed_keys
    return "no eager path: 1 unit -> %d keys" % len(src.accessed_keys)


print("== _selftest_partial_lifecycle ==")
check("selective first load, rest storageless", t_selective_first_load)
check("partial purge/reload keeps siblings", t_partial_purge_reload_preserves)
check("full move cycle + cuda purge block", t_full_cycle_move_and_purge)
check("no eager full-load path", t_no_eager_full_load)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")