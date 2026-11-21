"""自检 H3 真实 adapter（本机可跑）：stub 注入的 Comfy 语义。

stub 只模拟已核实的真实行为：device 透传构造、meta 零 storage、
strict 绑定缺 key 抛错。量化 ops（mixed_precision）不在本机验证，
probe 20/21 在远端用真 ops 跑。Comfy 以来模块缺失时必须 ComfyUnavailable。
"""
import json
import os
import sys
import threading

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
from torch import nn  # noqa: E402

from dit_svc import weights as W  # noqa: E402
from dit_svc.model_adapters import (ComfyUnavailable, H3ComfyAdapter,  # noqa: E402
                                    ShapeProxySD, WAIVED_KEYS)


# -- stub comfy（语义对标真实快照，不冒充量化） --
class StubOps:
    class Linear(nn.Linear):
        pass

    RMSNorm = nn.RMSNorm
    disable_weight_init = None  # 下方回填

    @staticmethod
    def mixed_precision_ops(quant_config, dtype):
        raise AssertionError("quant ops not under local test (see probe 20/21)")


StubOps.disable_weight_init = StubOps


class StubQuantLinear(nn.Linear):
    """模拟 MixedPrecision Linear：factory_kwargs 决定重建 device。

    构造期 device 固定（如 meta）；_load_from_state_dict 按
    self.factory_kwargs["device"] 落权重（对标真实
    _load_quantized_module:1263）。adapter 若不同步 fk，权重回落 meta，
    审计 META_REMAINS 必炸——这正是远端 probe 20 的失败形态。
    """

    def __init__(self, in_f, out_f, bias=False, device=None, dtype=None):
        super().__init__(in_f, out_f, bias=bias, device=device, dtype=dtype)
        self.factory_kwargs = {"device": device, "dtype": dtype}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        wkey = prefix + "weight"
        if wkey not in state_dict:
            if strict:
                missing_keys.append(wkey)
            return
        w = state_dict.pop(wkey)
        dev = (self.factory_kwargs or {}).get("device", "cpu")
        self.weight = torch.nn.Parameter(w.to(dev))
        skey = prefix + "weight_scale"
        if skey in state_dict:
            state_dict.pop(skey)


class StubBlock(nn.Module):
    def __init__(self, hidden, ffn, device=None, operations=None, dtype=None):
        super().__init__()
        op = operations
        self.norm1 = op.RMSNorm(hidden, device=device, dtype=dtype)
        self.norm2 = op.RMSNorm(hidden, device=device, dtype=dtype)
        self.attn_qkv = op.Linear(hidden, 3 * hidden, bias=False,
                                   device=device, dtype=dtype)
        self.attn_out = op.Linear(hidden, hidden, bias=False,
                                   device=device, dtype=dtype)
        self.mlp_fc1 = StubQuantLinear(hidden, ffn, bias=False,
                                         device=device, dtype=dtype)
        self.mlp_fc2 = op.Linear(ffn, hidden, bias=False, device=device, dtype=dtype)


class StubRefinerBlock(nn.Module):
    def __init__(self, hidden, device=None, operations=None, dtype=None):
        super().__init__()
        self.attn_qkv = operations.Linear(hidden, 3 * hidden, bias=False,
                                            device=device, dtype=dtype)


class StubH3(nn.Module):
    def __init__(self, num_layers=4, hidden_size=64, ffn_hidden_size=128,
                 token_refiner_num_layers=1, rope_inv_freq_len=16,
                 time_embed_dim=32, image_model=None, device=None,
                 operations=None, dtype=None, **kwargs):
        super().__init__()
        op = operations
        self.blocks = nn.ModuleList(
            [StubBlock(hidden_size, ffn_hidden_size, device=device,
                       operations=op, dtype=dtype) for _ in range(num_layers)])
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList(
            [StubRefinerBlock(hidden_size, device=device, operations=op, dtype=dtype)
             for _ in range(token_refiner_num_layers)])
        self.condition_proj = op.Linear(32, hidden_size, bias=False,
                                          device=device, dtype=dtype)
        self.video_patch_proj = op.Linear(8, hidden_size, bias=False,
                                          device=device, dtype=dtype)
        # 故意不用 device：模拟真实 minimax/model.py:503/:508
        # （torch.empty(dtype=float32) 无 device），骨架必须靠 device 上下文兜底
        self.register_buffer("adaln_t_table",
                             torch.empty(8, time_embed_dim))
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq",
                                  torch.empty(rope_inv_freq_len))


class StubDetection:
    @staticmethod
    def detect_unet_config(sd, prefix, metadata=None):
        assert "blocks.0.attn_qkv.weight" in sd
        assert hasattr(sd["blocks.0.attn_qkv.weight"], "shape")
        return {"image_model": "minimax_h3", "num_layers": 4,
                "token_refiner_num_layers": 1, "hidden_size": 64,
                "ffn_hidden_size": 128, "rope_inv_freq_len": 16,
                "time_embed_dim": 32}


def _adapter():
    return H3ComfyAdapter(torch_mod=torch, ops_mod=StubOps,
                          model_cls=StubH3, detection_mod=StubDetection)


def _shapes():
    s = {}
    for i in range(4):
        s["blocks.%d.attn_qkv.weight" % i] = (192, 64)
        s["blocks.%d.attn_out.weight" % i] = (64, 64)
        s["blocks.%d.mlp_fc1.weight" % i] = (128, 64)
        s["blocks.%d.mlp_fc2.weight" % i] = (64, 128)
        s["blocks.%d.norm1.weight" % i] = (64,)
        s["blocks.%d.norm2.weight" % i] = (64,)
    s["token_refiner.blocks.0.attn_qkv.weight"] = (192, 64)
    s["condition_proj.weight"] = (64, 32)
    s["video_patch_proj.weight"] = (64, 8)
    s["adaln_t_table"] = (8, 32)
    s["rope.inv_freq"] = (16,)
    return s


def t_shape_proxy():
    p = ShapeProxySD({"a": (2, 3)})
    assert "a" in p and "b" not in p
    assert p["a"].shape == (2, 3)
    assert sorted(p.keys()) == ["a"]
    return "mapping ok"


def t_quant_config_gate():
    a = _adapter()
    qc, rep = a.quant_config_from_metadata({})
    assert qc == {} and rep["quantized"] is False
    good = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0",
         "layers": {"blocks.0.attn": {"format": "int8_tensorwise"}}})}
    qc, rep = a.quant_config_from_metadata(good)
    assert rep["formats"] == ["int8_tensorwise"], rep
    bad = {"_quantization_metadata": json.dumps({"format_version": "2.0",
                                                 "layers": {}})}
    try:
        a.quant_config_from_metadata(bad)
    except Exception as e:
        assert "format_version" in str(e), e
        return "gate ok (refuses 2.0)"
    raise AssertionError("bad version accepted")


def t_detect_and_skeleton_meta():
    a = _adapter()
    shapes = _shapes()
    cfg, rep = a.detect_unet_config(shapes, None)
    assert rep["num_layers"] == 4, rep
    header = {k: {"dtype": "F32", "shape": list(v)} for k, v in shapes.items()}
    cat = W.build_catalog_from_header("dit", header, "dit")
    model = a.build_skeleton(cat, cfg, None, torch.bfloat16, device="meta")
    assert len(model.blocks) == 4
    inv = W.storage_inventory(model, owner="dit").to_summary()
    assert inv["total_cpu"] == 0 and inv["total_cuda"] == 0, inv
    assert inv["meta_tensors"] > 0, inv
    # 无 device 分配必须也被兜到 meta（stub 故意复现真实 bug）
    assert model.get_buffer("adaln_t_table").is_meta, "adaln_t_table leaked to CPU"
    assert model.rope.get_buffer("inv_freq").is_meta, "inv_freq leaked to CPU"
    return "meta skeleton, %d meta tensors" % inv["meta_tensors"]


def t_holders_and_waived():
    a = _adapter()
    assert a.unit_holders("dit.blocks.2", "dit") == [("blocks.2", "module")]
    assert a.unit_holders("dit.refiner", "dit") == [("token_refiner", "module"),
                                                   ("condition_proj", "module")]
    assert a.unit_holders("dit.modules.adaln_t_table", "dit") == [("", "buffer:adaln_t_table")]
    assert "adaln_basis" in WAIVED_KEYS and "adaln_mean" in WAIVED_KEYS
    return "holders ok"


def _env():
    shapes = _shapes()
    header = {k: {"dtype": "F32", "shape": list(v)} for k, v in shapes.items()}
    cat = W.build_catalog_from_header("dit", header, "dit")
    a = _adapter()
    cfg, _ = a.detect_unet_config(shapes, None)
    model = a.build_skeleton(cat, cfg, None, torch.float32, device="meta")
    return a, cat, model


def t_strict_bind_and_audit():
    a, cat, model = _env()
    uid = "dit.blocks.0"
    torch.manual_seed(1)
    sub = {}
    for k in cat.units[uid].checkpoint_keys:
        if k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            sub[k] = torch.ones(64)
        elif "mlp_fc1" in k:
            sub[k] = torch.randn(128, 64)
        elif "mlp_fc2" in k:
            sub[k] = torch.randn(64, 128)
        elif "attn_out" in k:
            sub[k] = torch.randn(64, 64)
        else:
            sub[k] = torch.randn(192, 64)
    rep = a.materialize_unit(model, uid, sub, device="cpu")
    audit = a.audit_after_bind(cat, model, uid, rep, expect_loaded=True)
    assert audit["verdict"] == "ok", audit
    # 缺 key 必须抛（strict 关门），不留随机参数
    bad = dict(sub)
    bad.pop(sorted(bad.keys())[0])
    try:
        a.materialize_unit(model, uid, bad, device="cpu")
    except Exception as e:
        assert "strict" in str(e).lower() or "missing" in str(e).lower(), e
    else:
        raise AssertionError("missing key accepted")
    # 跨 unit 多余 key 必须拒（不得静默忽略）
    other_key = next(iter(cat.units["dit.blocks.1"].checkpoint_keys))
    mixed = dict(sub)
    mixed[other_key] = torch.randn(4)
    try:
        a.materialize_unit(model, uid, mixed, device="cpu")
    except Exception as e:
        assert "unconsumed" in str(e).lower(), e
    else:
        raise AssertionError("cross-unit extra key accepted")
    # expected_keys 精确钉死：多/少都拒
    try:
        a.materialize_unit(model, uid, sub, device="cpu",
                           expected_keys=list(sub.keys()) + ["dit.blocks.0.EXTRA"])
    except Exception as e:
        assert "mismatch" in str(e).lower(), e
    else:
        raise AssertionError("expected_keys extra accepted")
    return "bind ok, strict refuses missing"


def t_purge_keeps_slots():
    a, cat, model = _env()
    uid = "dit.blocks.1"
    torch.manual_seed(2)
    sub = {k: torch.randn(64, 64) if k.endswith("attn_out.weight")
           else torch.randn(192, 64) if "attn_qkv" in k
           else torch.randn(128, 64) if "mlp_fc1" in k
           else torch.randn(64, 128) if "mlp_fc2" in k
           else torch.ones(64)
           for k in cat.units[uid].checkpoint_keys}
    a.materialize_unit(model, uid, sub, device="cpu")
    n_before = len(model.blocks)
    mod_before = model.blocks[1]
    a.purge_unit(model, uid)
    assert len(model.blocks) == n_before and model.blocks[1] is mod_before
    audit = a.audit_after_bind(cat, model, uid,
                               {"bound_keys": [], "waived": []},
                               expect_loaded=False)
    assert audit["verdict"] == "ok", audit
    return "purge ok, slots kept"


def t_registry_lock_smoke():
    reg = W.WeightRegistry()
    h = {"a.weight": {"dtype": "F32", "shape": [4]},
         "b.weight": {"dtype": "F32", "shape": [4]}}
    reg.register_recipe(W.recipe_from_header_dict("m", "dit", "fake://x", h))
    reg.attach_catalog(W.build_catalog_from_header("m", h, "dit"))
    errs = []

    def writer(n):
        try:
            for _ in range(50):
                reg.set_unit_loaded("m", "m.modules.a", "cpu",
                                    resident_bytes_cpu=16)
                reg.snapshot()
                reg.set_unit_absent("m", "m.modules.a")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    # header 只有顶层 keys -> dit.modules.a/b；用真实 unit id
    assert not errs
    ths = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert not errs, errs[:1]
    snap = reg.snapshot()
    json.dumps(snap)
    return "4 threads x 50 cycles clean"


def t_no_comfy_gives_clear_error():
    a = H3ComfyAdapter()
    try:
        a.build_skeleton({}, {}, None)
    except Exception as e:
        assert "injected" in str(e), e
        return "ComfyUnavailable ok"
    raise AssertionError("no error without comfy")


def t_quant_factory_kwargs():
    # fixture：stub 量化层的 factory_kwargs 在骨架期为 meta（真实现象），
    # materialize 必须同步到 cpu，否则权重回落 meta。
    a = _adapter()
    shapes = _shapes()
    header = {k: {"dtype": "F32", "shape": list(v)} for k, v in shapes.items()}
    # 量化伴生键（hook 消费，不进 state_dict  strict 残留）
    header["blocks.0.mlp_fc1.weight_scale"] = {"dtype": "F32", "shape": [128]}
    header["blocks.1.mlp_fc1.weight_scale"] = {"dtype": "F32", "shape": [128]}
    cat = W.build_catalog_from_header("dit", header, "dit")
    cfg, _ = a.detect_unet_config(shapes, None)
    model = a.build_skeleton(cat, cfg, None, torch.float32, device="meta")
    q = model.blocks[0].mlp_fc1
    assert isinstance(q, StubQuantLinear), type(q)
    assert q.factory_kwargs["device"] == "meta", q.factory_kwargs
    uid = "dit.blocks.0"
    torch.manual_seed(7)
    sub = {}
    for k in cat.units[uid].checkpoint_keys:
        if k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            sub[k] = torch.ones(64)
        elif "mlp_fc1" in k and k.endswith("weight_scale"):
            sub[k] = torch.ones(128)
        elif "mlp_fc1" in k:
            sub[k] = torch.randn(128, 64)
        elif "mlp_fc2" in k:
            sub[k] = torch.randn(64, 128)
        elif "attn_out" in k:
            sub[k] = torch.randn(64, 64)
        else:
            sub[k] = torch.randn(192, 64)
    rep = a.materialize_unit(model, uid, sub, device="cpu",
                             expected_keys=cat.units[uid].checkpoint_keys)
    assert rep["factory_kwargs_synced"] >= 1, rep
    assert rep["meta_repair"] == [], rep
    assert q.factory_kwargs["device"] == "cpu", q.factory_kwargs
    assert not q.weight.is_meta, "quant weight fell back to meta"
    audit = a.audit_after_bind(cat, model, uid, rep, expect_loaded=True)
    assert audit["verdict"] == "ok", audit
    # _sync_factory_kwargs 原语：陈旧 fk 被纠正并计数（materialize 内同款调用）
    q.factory_kwargs["device"] = "meta"
    n2 = a._sync_factory_kwargs(model.blocks[1], "cpu")
    assert n2 >= 1, n2
    assert model.blocks[1].mlp_fc1.factory_kwargs["device"] == "cpu"
    # purge 把 fk 置回 meta（与空壳状态一致），reload 经 materialize 回 cpu
    a.purge_unit(model, uid)
    assert q.factory_kwargs["device"] == "meta", q.factory_kwargs
    rep2 = a.materialize_unit(model, uid, sub, device="cpu",
                              expected_keys=cat.units[uid].checkpoint_keys)
    assert rep2["factory_kwargs_synced"] >= 1, rep2
    assert not q.weight.is_meta
    assert a.audit_after_bind(cat, model, uid, rep2,
                              expect_loaded=True)["verdict"] == "ok"
    return "fk sync + rebind + purge/reload ok"


def t_noncompute_short_circuit():
    a = _adapter()
    shapes = _shapes()
    header = {k: {"dtype": "F32", "shape": list(v)} for k, v in shapes.items()}
    header["adaln_basis"] = {"dtype": "F32", "shape": [8, 32]}
    header["adaln_mean"] = {"dtype": "F32", "shape": [8, 32]}
    cat = W.build_catalog_from_header("dit", header, "dit",
                                      noncompute_exact=["adaln_basis", "adaln_mean"],
                                      noncompute_reason="test")
    assert "adaln_basis" not in cat.key_to_unit
    assert cat.noncompute_keys == ("adaln_basis", "adaln_mean"), cat.noncompute_keys
    _, _, model = _env()
    n_params_before = sum(1 for _ in model.parameters())
    rep = a.materialize_unit(model, "dit.modules.adaln_basis",
                             {"adaln_basis": torch.zeros(8, 32)}, device="cpu")
    assert rep.get("noncompute") is True, rep
    n_params_after = sum(1 for _ in model.parameters())
    assert n_params_before == n_params_after, "noncompute touched model"
    audit = a.audit_after_bind(cat, model, "dit.blocks.0",
                               {"bound_keys": [], "waived": [],
                                "noncompute": True, "device": "cpu"},
                               expect_loaded=False)
    assert audit["verdict"] == "ok", audit
    return "waived-only never materialized"


print("== _selftest_h3_adapter ==")
check("shape proxy mapping", t_shape_proxy)
check("quant config version gate", t_quant_config_gate)
check("detect + meta skeleton zero storage", t_detect_and_skeleton_meta)
check("unit holders + waived", t_holders_and_waived)
check("strict bind + audit; missing refused", t_strict_bind_and_audit)
check("purge keeps slots, zero storage", t_purge_keeps_slots)
check("waived-only short-circuit", t_noncompute_short_circuit)
check("quant factory_kwargs sync", t_quant_factory_kwargs)
check("registry lock concurrency", t_registry_lock_smoke)
check("no-comfy clear error", t_no_comfy_gives_clear_error)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
