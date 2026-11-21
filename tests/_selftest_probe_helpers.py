"""probe 安全 helper 本地自测（CPU only）。

证明（远端同样生效；CUDA 断言本身只在远端跑）：
- parse_units 拒 all/空；
- enforce_probe_budget：resolve 后 unit 数 + 源端字节双上限，
  range 展开（0-49 -> 50 units）被拒；
- require_schema：不符 verdict=unverified 且抛 SystemExit（不许 ok）；
- release_staging_proof：del+gc 后无强引用残留；外部仍持有则检出 alive>0；
- rss_current/peak 返回正数（真 peak，非 delta）；
- env_info(include_cuda=False) 无 CUDA 字段（零 CUDA 接触）；
- prepare_comfy_import 缺 comfy_root 不炸（记录 missing）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "probes"))
sys.path.insert(0, os.path.dirname(HERE))

fails = []


def check(name, fn):
    try:
        msg = fn()
    except Exception as e:  # noqa: BLE001
        fails.append(name)
        print("FAIL %-44s %r" % (name, e))
    else:
        print("ok   %-44s %s" % (name, msg))


import weights_common as C  # noqa: E402
from dit_svc import weights as W  # noqa: E402


def t_parse_units():
    assert C.parse_units("dit.blocks.0") == ["dit.blocks.0"]
    for bad in ("all", "dit.blocks.0,all", "", "  "):
        try:
            C.parse_units(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("accepted %r" % bad)
    return "all/empty refused"


def _cat(nblocks=3):
    h = {}
    for i in range(nblocks):
        h["blocks.%d.attn.qkv.weight" % i] = {"dtype": "F32",
                                              "shape": [256, 128]}
    return W.build_catalog_from_header("m", h, "dit")


def t_budget():
    cat = _cat(3)
    ok = W.resolve_units(cat, "units", ["m.blocks.0-1"])
    rep = C.enforce_probe_budget(cat, ok)
    assert rep["resolved"] == ["m.blocks.0", "m.blocks.1"], rep
    assert rep["expected_source_bytes"] == 2 * 256 * 128 * 4, rep
    # range 展开成 50 units（用大 catalog 模拟）
    big = _cat(50)
    many = W.resolve_units(big, "units", ["m.blocks.0-49"])
    assert len(many) == 50, len(many)
    try:
        C.enforce_probe_budget(big, many)
    except SystemExit:
        pass
    else:
        raise AssertionError("range-expanded 50 units allowed")
    # 字节上限：单 unit 超 1GiB 拒
    huge = {"w": {"dtype": "F32", "shape": [(1 << 28) + 1]}}
    hcat = W.build_catalog_from_header("h", huge, "dit")
    hid = hcat.unit_ids()
    try:
        C.enforce_probe_budget(hcat, hid)
    except SystemExit:
        pass
    else:
        raise AssertionError("oversize source bytes allowed")
    return "count+bytes gates ok"


def t_schema_gate():
    env = {"torch": "x", "safetensors": "y", "comfy-kitchen": "z"}
    rep: dict = {}
    try:
        C.require_schema(env, rep)
    except SystemExit:
        assert rep["verdict"] == "unverified", rep
        assert rep["schema_check"]["mismatches"], rep
    else:
        raise AssertionError("mismatch not rejected")
    good = dict(C.EXPECTED_SCHEMA)
    rep2: dict = {}
    C.require_schema(good, rep2)
    assert rep2["schema_check"]["verdict"] == "ok", rep2
    return "mismatch -> unverified+exit; match -> ok"


def t_release_proof():
    import torch
    a = torch.randn(8)
    b = torch.randn(8)
    staged = [a, b]
    del a, b
    proof = C.release_staging_proof(staged)
    del staged
    assert proof == {"staged_tensors": 2, "alive_after_release": 0}, proof
    # 外部仍持有：必须检出
    c = torch.randn(8)
    staged2 = [c]
    proof2 = C.release_staging_proof(staged2)
    assert proof2["alive_after_release"] == 1, proof2
    del staged2, c
    return "release proof detects survivors"


def t_rss_and_env():
    cur = C.rss_current_bytes()
    peak = C.rss_peak_bytes()
    assert cur > 0 and peak >= cur, (cur, peak)
    env = C.env_info(include_cuda=False)
    assert "cuda_available" not in env and "cuda_free_bytes" not in env, env
    assert isinstance(env["torch"], str) and isinstance(env["safetensors"], str)
    note = C.prepare_comfy_import("", cpu=True)
    assert note["cli_args"].startswith("missing"), note
    return "rss cur=%d peak=%d; no cuda fields" % (cur, peak)


print("== _selftest_probe_helpers ==")
check("parse_units refuses all/empty", t_parse_units)
check("resolve budget gates", t_budget)
check("schema gate", t_schema_gate)
check("staging release proof", t_release_proof)
check("rss + cuda-free env", t_rss_and_env)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
