"""自检 P0：recipe/header catalog 构造 + 稳定选择器 + 四动作状态机。

只依赖标准库（不断言 torch 可用；另用源码扫描保证 weights.py 顶层
无 torch 导入，做到 stdlib 可 import）。
真实 H3 checkpoint 不可用：header 全部为手工 fake key（形状仿 H3 风格但
明确标注 fake），只验证底座逻辑，不验证任何真实模型结论。
"""
import json
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


from dit_svc import weights as W  # noqa: E402


def fake_header():
    # key 命名按真实 H3 布局（Comfy 5f415e31a model_detection.py:401-433），
    # shape 为手工 fake，仅验证底座逻辑。
    h = {}
    for i in range(4):
        h["blocks.%d.attn.qkv_proj.weight" % i] = {"dtype": "BF16",
                                                   "shape": [3 * 256, 256]}
        h["blocks.%d.mlp.fc1.weight" % i] = {"dtype": "BF16",
                                             "shape": [512, 256]}
    h["token_refiner.blocks.0.attn.qkv_proj.weight"] = {"dtype": "F32",
                                                        "shape": [768, 256]}
    h["condition_proj.weight"] = {"dtype": "F32", "shape": [256, 256]}
    h["video_patch_proj.weight"] = {"dtype": "F32", "shape": [256, 128]}
    h["final_layer.video_out.weight"] = {"dtype": "F32", "shape": [64, 256]}
    h["__weird_tmp"] = {"dtype": "F32", "shape": [8]}
    return h


def t_no_top_level_torch_import():
    with open(os.path.join(os.path.dirname(HERE), "dit_svc", "weights.py"),
              encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            # 仅第 0 列顶层导入算数；函数内延迟 import torch 是允许的
            if line.startswith("import torch") or line.startswith("from torch"):
                raise AssertionError("top-level torch import at line %d" % i)
    return "weights.py stdlib-clean"


def t_build_catalog_blocks_stable():
    cat = W.build_catalog_from_header("dit", fake_header(), "dit",
                                      ignored_exact=["__weird_tmp"])
    ids = cat.unit_ids()
    assert ids == sorted(ids), ids
    assert "dit.blocks.0" in ids and "dit.blocks.3" in ids, ids
    assert "dit.refiner" in ids, ids
    assert "dit.modules.video_patch_proj" in ids, ids
    assert "dit.modules.final_layer" in ids, ids
    assert "__weird_tmp" not in cat.key_to_unit
    assert cat.key_to_unit["token_refiner.blocks.0.attn.qkv_proj.weight"] == \
        "dit.refiner"
    assert cat.key_to_unit["condition_proj.weight"] == "dit.refiner"
    assert cat.ignored_keys == ("__weird_tmp",), cat.ignored_keys
    # 穩定性：key 顺序打乱重建，unit id 与 key 归属不变
    h2 = dict(reversed(list(fake_header().items())))
    cat2 = W.build_catalog_from_header("dit", h2, "dit",
                                       ignored_exact=["__weird_tmp"])
    assert cat2.unit_ids() == ids
    assert cat2.key_to_unit == cat.key_to_unit
    return "units=%d keys=%d" % (len(ids), len(cat.key_to_unit))


def t_double_underscore_key_is_real_key():
    cat = W.build_catalog_from_header("dit", fake_header(), "dit")
    assert "__weird_tmp" in cat.key_to_unit, cat.ignored_keys
    assert cat.ignored_keys == (), cat.ignored_keys


def t_cover_rejects_leftover():
    cat = W.build_catalog_from_header("dit", fake_header(), "dit",
                                      ignored_exact=["__weird_tmp"])
    try:
        cat.check_cover(list(fake_header().keys()) + ["mystery.weight"])
    except W.UnmappedKeyError as e:
        return "UnmappedKeyError ok: %s" % str(e)[:40]
    raise AssertionError("no raise")


def t_selector_exact_group_range():
    cat = W.build_catalog_from_header("dit", fake_header(), "dit",
                                      ignored_exact=["__weird_tmp"])
    assert W.resolve_units(cat, "all") == cat.unit_ids()
    one = W.resolve_units(cat, "units", ["dit.blocks.1"])
    assert one == ["dit.blocks.1"], one
    grp = W.resolve_units(cat, "units", ["dit.blocks"])
    assert grp == ["dit.blocks.0", "dit.blocks.1", "dit.blocks.2",
                   "dit.blocks.3"], grp
    rng = W.resolve_units(cat, "units", ["dit.blocks.1-2"])
    assert rng == ["dit.blocks.1", "dit.blocks.2"], rng
    both = W.resolve_units(cat, "units", ["dit.blocks.2-3", "dit.blocks.1",
                                          "dit.blocks.2-3"])
    assert both == ["dit.blocks.1", "dit.blocks.2", "dit.blocks.3"], both
    try:
        W.resolve_units(cat, "units", ["dit.blocks.*"])
    except W.SelectorError:
        pass
    else:
        raise AssertionError("glob accepted")
    try:
        W.resolve_units(cat, "units", [])
    except W.SelectorError:
        pass
    else:
        raise AssertionError("empty accepted")
    return "selector strict ok"


def t_four_action_matrix():
    reg = W.WeightRegistry()
    h = fake_header()
    rec = W.recipe_from_header_dict("dit", "dit", "fake://h", h)
    reg.register_recipe(rec)
    reg.attach_catalog(W.build_catalog_from_header("dit", h, "dit",
                                                     ignored_exact=["__weird_tmp"]))
    units = ["dit.blocks.0", "dit.blocks.1"]
    reg.check_transition("disk2ram", "dit", units)
    for u in units:
        reg.set_unit_loaded("dit", u, "cpu", resident_bytes_cpu=100,
                            resident_bytes_cuda=0)
    # 同 recipe 已驻留 = no-op 成功（返回 True，不抛）
    assert reg.check_transition("disk2ram", "dit", units) is True
    reg.check_transition("ram2vram", "dit", units)
    for u in units:
        reg.set_unit_loaded("dit", u, "cuda:0", resident_bytes_cpu=0,
                            resident_bytes_cuda=100)
    try:
        reg.check_transition("purgefromram", "dit", units)
    except W.TransitionError as e:
        assert "cuda" in str(e).lower() or "CUDA" in str(e)
    else:
        raise AssertionError("purge with cuda not blocked")
    try:
        reg.check_transition("ram2vram", "dit", ["dit.blocks.2"])
    except W.TransitionError:
        pass
    else:
        raise AssertionError("ram2vram absent not rejected")
    snap = reg.snapshot()
    json.dumps(snap)  # 必须可 JSON
    assert snap["models"]["dit"]["absent_units"] == 5, snap["models"]["dit"]
    return "matrix ok; absent=%d" % snap["models"]["dit"]["absent_units"]


def t_recipe_change_rejected():
    reg = W.WeightRegistry()
    h = fake_header()
    rec = W.recipe_from_header_dict("dit", "dit", "fake://h", h)
    reg.register_recipe(rec)
    reg.attach_catalog(W.build_catalog_from_header("dit", h, "dit",
                                                     ignored_exact=["__weird_tmp"]))
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=10,
                        resident_bytes_cuda=0)
    other = W.recipe_from_header_dict("dit", "dit", "fake://h2", h)
    try:
        reg.register_recipe(other)
    except W.RecipeError:
        return "recipe change blocked at register ok"
    raise AssertionError("recipe change allowed")


print("== _selftest_weights_logic ==")
check("stdlib import (no top-level torch)", t_no_top_level_torch_import)
check("catalog build + stable ids", t_build_catalog_blocks_stable)
check("double-underscore key is a real key", t_double_underscore_key_is_real_key)
check("cover rejects leftover key", t_cover_rejects_leftover)
check("selector exact/group/range", t_selector_exact_group_range)
check("four-action transition matrix", t_four_action_matrix)
check("recipe change rejected", t_recipe_change_rejected)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
