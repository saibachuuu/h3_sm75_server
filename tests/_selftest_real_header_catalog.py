"""真实 H3 布局 catalog 测试（本机可跑，stdlib only）。

按远端 header 报告（remote_environment.json，主代理实测）复刻的
布局保真合成 header：50-block 结构缩为 3 block，但保留全部真实
key 家族——双量化布局（int8_tensorwise / asym_w4a8）、adaln 三件套、
rope、refiner、condition_proj、patch/final double 键。
断言 catalog 全覆盖、unit 划分、字节账、选择器与 waived 语义。

真实 1294-key header 由主代理在远端 probe_10 复核；此处只锁规则，
不冒充真实文件。真实数字（58 units / 13.998GB）见 docs/p0_weights.md。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc import weights as W
from dit_svc.model_adapters import WAIVED_KEYS

fails = []


def check(name, fn):
    try:
        msg = fn()
    except Exception as e:  # noqa: BLE001
        fails.append(name)
        print("FAIL %-44s %r" % (name, e))
    else:
        print("ok   %-44s %s" % (name, msg))


# 真实布局的 key 家族（后缀/伴生键按远端报告；形状随意但 dtype 真实）
def real_like_header(n_blocks=3):
    h = {}

    def put(k, dtype="F32", shape=(8,)):
        h[k] = {"dtype": dtype, "shape": list(shape)}

    for i in range(n_blocks):
        p = "blocks.%d" % i
        # int8_tensorwise 线性：I8 qdata + F32 per-row scale + U8 comfy_quant[82]
        put("%s.attn.qkv_proj.weight" % p, "I8", (24, 8))
        put("%s.attn.qkv_proj.weight_scale" % p, "F32", (24,))
        put("%s.attn.qkv_proj.comfy_quant" % p, "U8", (82,))
        # asym_w4a8 线性：F8 主体 + codebook + s_channel + s_rel
        put("%s.mlp.fc1.weight" % p, "F8_E4M3", (16, 8))
        put("%s.mlp.fc1.weight_codebook" % p, "F32", (16,))
        put("%s.mlp.fc1.weight_s_channel" % p, "F32", (8,))
        put("%s.mlp.fc1.weight_s_rel" % p, "F32", (16,))
        # norm 与纯 F32 线性
        put("%s.attn.q_norm.weight" % p, "F32", (8,))
        put("%s.attn.attn_out.weight" % p, "F32", (8, 8))
        put("%s.mlp.fc2.weight" % p, "F32", (8, 16))
    # refiner + condition
    put("token_refiner.blocks.0.attn.qkv_proj.weight", "F32", (24, 8))
    put("condition_proj.weight", "F32", (8, 32))
    # 双 patch 与 double head
    put("video_patch_proj.weight", "F32", (8, 8))
    put("audio_patch_proj.weight", "F32", (8, 8))
    put("final_layer.video_out.weight", "F32", (4, 8))
    put("final_layer.audio_out.weight", "F32", (4, 8))
    # rope + adaln 三件套（basis/mean 无前向消费，t_table 有）
    put("rope.inv_freq", "F32", (16,))
    put("adaln_t_table", "F32", (8, 32))
    put("adaln_basis", "F32", (8, 32))
    put("adaln_mean", "F32", (8, 32))
    return h


def t_cover_and_units():
    h = real_like_header()
    waived = [k for k in h if k.split(".")[-1] in WAIVED_KEYS
              or k in ("adaln_basis", "adaln_mean")]
    assert set(waived) == {"adaln_basis", "adaln_mean"}, waived
    cat = W.build_catalog_from_header("dit", h, "dit", ignored_exact=waived)
    cat.check_cover(list(h.keys()))
    ids = cat.unit_ids()
    assert "dit.refiner" in ids, ids
    for i in range(3):
        assert "dit.blocks.%d" % i in ids, ids
    assert "dit.modules.adaln_t_table" in ids, ids
    assert "dit.modules.rope" in ids, ids
    assert "dit.modules.final_layer" in ids, ids
    assert "dit.modules.video_patch_proj" in ids, ids
    # waived 不得进账
    for k in waived:
        assert k not in cat.key_to_unit, k
    # 字节账 = 各 key 真实 dtype 字节之和
    want = sum(W.tensor_key_bytes(v["dtype"], v["shape"])
               for k, v in h.items() if k not in waived)
    got = sum(cat.units[u].expected_bytes for u in ids)
    assert got == want, (got, want)
    # 选择器：整机 / 组 / 区间全可用
    assert W.resolve_units(cat, "all") == ids
    assert W.resolve_units(cat, "units", ["dit.blocks"]) == \
        ["dit.blocks.0", "dit.blocks.1", "dit.blocks.2"]
    assert W.resolve_units(cat, "units", ["dit.blocks.0-1"]) == \
        ["dit.blocks.0", "dit.blocks.1"]
    return "units=%d keys=%d bytes=%d" % (len(ids), len(cat.key_to_unit), got)


def t_descriptor_keys_present():
    h = real_like_header()
    missing = [k for k in W.H3_DESCRIPTOR_KEYS if k not in h
               and not k.startswith("time_embedder")]
    # time_embedder 二选一：本合成 header 用 adaln_t_table 路线
    assert "adaln_t_table" in h
    assert missing == [], missing
    return "descriptors ok (%d keys)" % len(W.H3_DESCRIPTOR_KEYS)


def t_quant_companions_stay_in_block():
    h = real_like_header()
    cat = W.build_catalog_from_header("dit", h, "dit",
                                      ignored_exact=["adaln_basis",
                                                     "adaln_mean"])
    for k in h:
        if k.startswith("blocks.1."):
            assert cat.key_to_unit[k] == "dit.blocks.1", (k, cat.key_to_unit[k])
    return "companions co-located with qdata"


print("== _selftest_real_header_catalog ==")
check("real-like cover + units + bytes", t_cover_and_units)
check("H3 descriptor keys present", t_descriptor_keys_present)
check("quant companions stay in block", t_quant_companions_stay_in_block)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
