"""Regression tests for the P0 review fixes (m00424):

same_source covers kind/options/adapter_version; deep-copy JSON summaries;
attach/recipe replace semantics + monotonic generation; unknown unit ids are
rejected by check_transition; purge on mixed/error must go through recover;
recover_unit requires a measured adapter release_proof (no ghost absent);
header parser hardening (<=32MiB, dtype/shape/offsets/EOF/dup keys);
snapshot revision + counts/bytes.

stdlib only. Run: <venv python> tests/_selftest_registry_review.py
"""
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc import weights as W

fails = []


def check(name, fn):
    try:
        msg = fn()
    except Exception as e:  # noqa: BLE001
        fails.append(name)
        print("FAIL %-44s %r" % (name, e))
    else:
        print("ok   %-44s %s" % (name, msg))


def fake_header():
    h = {}
    for i in range(3):
        h["blocks.%d.attn.qkv.weight" % i] = {"dtype": "F32",
                                              "shape": [256, 128]}
    h["__weird_tmp"] = {"dtype": "F32", "shape": [8]}
    return h


def reg_with(h, kind="dit", options=None):
    reg = W.WeightRegistry()
    rec = W.recipe_from_header_dict("dit", kind, "fake://h", h,
                                    options=options)
    reg.register_recipe(rec)
    reg.attach_catalog(W.build_catalog_from_header("dit", h, "dit",
                                                   ignored_exact=["__weird_tmp"]))
    return reg


def t_same_source_kind_options_version():
    h = fake_header()
    r0 = W.recipe_from_header_dict("dit", "dit", "fake://h", h)
    assert r0.same_source(W.recipe_from_header_dict("dit", "dit", "fake://h", h))
    assert not r0.same_source(W.recipe_from_header_dict("dit", "te", "fake://h", h))
    assert not r0.same_source(
        W.recipe_from_header_dict("dit", "dit", "fake://h", h,
                                  options={"mlp_chunk": 1}))
    assert not r0.same_source(
        W.recipe_from_header_dict("dit", "dit", "fake://h", h,
                                  adapter_version="weights-v9"))
    # same file + same options registered while resident with old kind: conflict
    reg = reg_with(h)
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    try:
        reg.register_recipe(
            W.recipe_from_header_dict("dit", "te", "fake://h", h))
    except W.RecipeError:
        pass
    else:
        raise AssertionError("kind change while resident allowed")
    return "kind/options/version all gate same_source"


def t_recipe_switch_clears_catalog_monotonic():
    h = fake_header()
    reg = reg_with(h)
    g0 = reg.model_state("dit")["generation"]
    assert g0 >= 1, g0
    # no resident: switch kind allowed, old records cleared, generation grows
    reg.register_recipe(W.recipe_from_header_dict("dit", "te", "fake://h", h))
    # catalog 已清：model_state 此时无 catalog 可报（StateError），snapshot 为空
    try:
        reg.model_state("dit")
    except W.StateError:
        pass
    else:
        raise AssertionError("model_state served without catalog")
    assert reg.snapshot()["models"] == {}, reg.snapshot()["models"]
    try:
        reg.unit_record("dit", "dit.blocks.0")
    except W.StateError:
        pass
    else:
        raise AssertionError("stale unit record survived recipe switch")
    # re-attach new catalog: unit ids back, generation keeps rising
    reg.attach_catalog(W.build_catalog_from_header("dit", h, "dit",
                                                   ignored_exact=["__weird_tmp"]))
    assert reg.model_state("dit")["generation"] > g0
    assert reg.model_state("dit")["recipe"]["kind"] == "te"
    assert "dit.blocks.0" in [u["id"] for u in
                              reg.model_state("dit")["units"]]
    return "switch clears stale records; generation monotonic"


def t_attach_replace_drops_stale():
    h = fake_header()
    reg = reg_with(h)
    gen0 = reg.model_state("dit")["generation"]
    # narrower header, no resident: replace drops dit.blocks.2
    h2 = {k: v for k, v in h.items() if "blocks.2" not in k}
    reg.attach_catalog(W.build_catalog_from_header("dit", h2, "dit",
                                                   ignored_exact=["__weird_tmp"]))
    try:
        reg.unit_record("dit", "dit.blocks.2")
    except W.StateError:
        pass
    else:
        raise AssertionError("stale dit.blocks.2 kept after replace")
    assert reg.model_state("dit")["generation"] > gen0
    return "replace drops stale; generation grows"


def t_attach_changed_while_resident_refused():
    h = fake_header()
    reg = reg_with(h)
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    h2 = {k: v for k, v in h.items() if "blocks.2" not in k}
    try:
        reg.attach_catalog(W.build_catalog_from_header("dit", h2, "dit",
                                                       ignored_exact=["__weird_tmp"]))
    except W.CatalogError:
        return "resident catalog change refused ok"
    raise AssertionError("resident catalog change allowed")


def t_unknown_unit_ids_rejected():
    reg = reg_with(fake_header())
    for action in ("disk2ram", "ram2vram", "vram2ram", "purgefromram"):
        try:
            reg.check_transition(action, "dit", ["dit.nope"])
        except W.SelectorError:
            pass
        else:
            raise AssertionError("%s accepted unknown unit" % action)
    try:
        reg.unit_record("dit", "dit.nope")
    except W.StateError:
        pass
    else:
        raise AssertionError("unit_record accepted unknown unit")
    return "unknown unit never passes as absent"


def t_purge_mixed_error_needs_recover():
    reg = reg_with(fake_header())
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    reg.mark_error("dit", "dit.blocks.0", "boom")
    for action in ("purgefromram", "disk2ram", "ram2vram", "vram2ram"):
        try:
            reg.check_transition(action, "dit", ["dit.blocks.0"])
        except W.TransitionError:
            pass
        else:
            raise AssertionError("%s bypassed error state" % action)
    return "error blocks all four actions"


def t_recover_needs_measured_proof():
    reg = reg_with(fake_header())
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    reg.mark_error("dit", "dit.blocks.0", "boom")
    try:
        reg.recover_unit("dit", "dit.blocks.0")
    except W.TransitionError:
        pass
    else:
        raise AssertionError("recover without proof allowed")
    try:
        reg.recover_unit("dit", "dit.blocks.0",
                         release_proof={"resident_bytes_cpu": 4,
                                        "resident_bytes_cuda": 0,
                                        "storage_digest": []})
    except W.TransitionError:
        pass
    else:
        raise AssertionError("recover with live storage allowed (ghost)")
    out = reg.recover_unit("dit", "dit.blocks.0",
                           release_proof={"resident_bytes_cpu": 0,
                                          "resident_bytes_cuda": 0,
                                          "storage_digest": []})
    assert out["state"] == "absent", out
    assert reg.unit_record("dit", "dit.blocks.0")["state"] == "absent"
    return "proof-gated recover ok"


def t_deepcopy_isolation():
    reg = reg_with(fake_header())
    rec = W.WeightRegistry()
    r = W.recipe_from_header_dict("dit", "dit", "fake://h", fake_header(),
                                  options={"nested": {"a": [1, 2]}})
    rec.register_recipe(r)
    rec.attach_catalog(W.build_catalog_from_header("dit", fake_header(), "dit",
                                                   ignored_exact=["__weird_tmp"]))
    rec.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    snap = rec.snapshot()
    snap["models"]["dit"]["units"][0]["recovery_actions"].append("X")
    snap["models"]["dit"]["recipe"]["options"]["nested"]["a"].append(9)
    snap["models"]["dit"]["counts"]["ram"] = 999
    st = rec.model_state("dit")
    assert st["units"][0]["recovery_actions"] == [], st["units"][0]
    assert st["recipe"]["options"] == {"nested": {"a": [1, 2]}}, st["recipe"]
    assert st["counts"]["ram"] == 1, st["counts"]
    ur = rec.unit_record("dit", "dit.blocks.0")
    ur["recovery_actions"].append("Y")
    ur["state"] = "vram"
    assert rec.unit_record("dit", "dit.blocks.0")["state"] == "ram"
    json.dumps(rec.snapshot())
    return "mutating outputs cannot pollute registry"


def t_snapshot_revision_counts_bytes():
    reg = reg_with(fake_header())
    s0 = reg.snapshot()
    assert isinstance(s0["revision"], int), s0.keys()
    ms = s0["models"]["dit"]
    assert set(ms["counts"]) == {"absent", "ram", "vram", "mixed", "error"}
    assert set(ms["bytes"]) == {"resident_cpu", "resident_cuda", "expected"}
    assert ms["counts"]["absent"] == len(ms["units"]), ms["counts"]
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=100,
                        resident_bytes_cuda=0)
    s1 = reg.snapshot()
    assert s1["revision"] > s0["revision"], (s0["revision"], s1["revision"])
    assert s1["models"]["dit"]["counts"]["ram"] == 1
    assert s1["models"]["dit"]["bytes"]["resident_cpu"] == 100
    json.dumps(s1)
    return "revision=%d counts/bytes ok" % s1["revision"]


def _write_safetensors(tmp, header_obj=None, raw_json=None, body_len=64):
    if raw_json is None:
        raw_json = json.dumps(header_obj).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw_json)))
        f.write(raw_json)
        f.write(b"\x00" * body_len)
    return tmp


def t_header_rejects_bad_inputs():
    with tempfile.TemporaryDirectory() as d:
        # bad dtype
        p = _write_safetensors(os.path.join(d, "a.sft"),
                               {"w": {"dtype": "XX", "shape": [2],
                                      "data_offsets": [0, 8]}})
        try:
            W.read_safetensors_header(p)
        except W.CatalogError:
            pass
        else:
            raise AssertionError("bad dtype accepted")
        # negative / float dim
        for bad_shape in ([-1], [2.5], "x"):
            p = _write_safetensors(os.path.join(d, "b.sft"),
                                   {"w": {"dtype": "F32", "shape": bad_shape,
                                          "data_offsets": [0, 8]}})
            try:
                W.read_safetensors_header(p)
            except W.CatalogError:
                pass
            else:
                raise AssertionError("bad shape accepted: %r" % (bad_shape,))
        # offsets: end < start, overrun EOF
        for bad_off in ([8, 4], [0, 10 ** 9]):
            p = _write_safetensors(os.path.join(d, "c.sft"),
                                   {"w": {"dtype": "F32", "shape": [2],
                                          "data_offsets": bad_off}},
                                   body_len=8)
            try:
                W.read_safetensors_header(p)
            except W.CatalogError:
                pass
            else:
                raise AssertionError("bad offsets accepted: %r" % (bad_off,))
        # duplicate JSON key
        dup = (b'{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},'
               b' "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}')
        p = _write_safetensors(os.path.join(d, "d.sft"), raw_json=dup)
        try:
            W.read_safetensors_header(p)
        except W.CatalogError:
            pass
        else:
            raise AssertionError("duplicate key accepted")
        # header overruns EOF
        p = os.path.join(d, "e.sft")
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", 100) + b'{"w": 1}')
        try:
            W.read_safetensors_header(p)
        except W.CatalogError:
            pass
        else:
            raise AssertionError("EOF overrun accepted")
        # tiny file
        p = os.path.join(d, "f.sft")
        with open(p, "wb") as f:
            f.write(b"123")
        try:
            W.read_safetensors_header(p)
        except W.CatalogError:
            pass
        else:
            raise AssertionError("tiny file accepted")
        # 32MiB bound: header_len field huge, body absent
        p = os.path.join(d, "g.sft")
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", 64 * 1024 * 1024))
        try:
            W.read_safetensors_header(p)
        except W.CatalogError:
            pass
        else:
            raise AssertionError("oversize header accepted")
        # good file round-trips
        p = _write_safetensors(os.path.join(d, "ok.sft"),
                               {"w": {"dtype": "F32", "shape": [2],
                                      "data_offsets": [0, 8]},
                                "__metadata__": {"a": "b"}}, body_len=8)
        t, hl = W.read_safetensors_header(p)
        assert set(t) == {"w"}, t.keys()
        assert t["w"]["shape"] == [2] and t["w"]["data_offsets"] == [0, 8]
    return "header validation matrix ok"


def t_signature_covers_shape_dtype():
    h = fake_header()
    reg = reg_with(h)
    reg.set_unit_loaded("dit", "dit.blocks.0", "cpu", resident_bytes_cpu=4,
                        resident_bytes_cuda=0)
    # 同 key 集、某 key shape 变大：签名变，驻留时换 catalog 必须拒
    h2 = {k: dict(v) for k, v in h.items()}
    k0 = "blocks.0.attn.qkv.weight"
    h2[k0] = {"dtype": "F32", "shape": [512, 256]}
    cat2 = W.build_catalog_from_header("dit", h2, "dit",
                                       ignored_exact=["__weird_tmp"])
    assert reg._catalog_signature(cat2) != reg._catalog_signature(
        reg.catalogs["dit"])
    try:
        reg.attach_catalog(cat2)
    except W.CatalogError:
        pass
    else:
        raise AssertionError("same-keys-different-shape accepted while resident")
    # dtype 变但字节相同（F32[4] vs I8[16]）：签名也必须变
    h3 = {k: dict(v) for k, v in h.items()}
    h3[k0] = {"dtype": "I8", "shape": [16]}
    cat3 = W.build_catalog_from_header("dit", h3, "dit",
                                       ignored_exact=["__weird_tmp"])
    assert reg._catalog_signature(cat3) != reg._catalog_signature(
        reg.catalogs["dit"])
    return "signature pins dtype+shape"


def t_audit_rejects_meta_missing():
    h = fake_header()
    reg = reg_with({k: v for k, v in h.items() if k != "__weird_tmp"})
    uids = [u for u in reg.catalogs["dit"].unit_ids()
            if u.startswith("dit.blocks")]
    uid = uids[0]
    inv = W.StorageInventory()
    bound = list(reg.catalogs["dit"].units[uid].checkpoint_keys)
    audit = W.audit_unit(reg.catalogs["dit"], uid, bound, inv, expect_loaded=True)
    assert audit["verdict"] == "META_MISSING", audit
    # 零 expected 的退化 unit 不误伤：expected==0 且 resident==0 应 ok
    return "meta-missing refused: %s" % audit["verdict"]


def t_noncompute_catalog():
    h = dict(fake_header())
    h["adaln_basis"] = {"dtype": "F32", "shape": [4]}
    cat = W.build_catalog_from_header("dit", h, "dit",
                                      ignored_exact=["__weird_tmp"],
                                      noncompute_exact=["adaln_basis"],
                                      noncompute_reason="reason-x")
    cat.check_cover(list(h.keys()))
    assert "adaln_basis" not in cat.key_to_unit
    assert cat.noncompute_keys == ("adaln_basis",)
    inv = W.StorageInventory()
    audit = W.audit_unit(cat, "adaln_basis", [], inv, expect_loaded=False)
    assert audit["verdict"] == "ok" and audit.get("noncompute") is True, audit
    bad = W.audit_unit(cat, "adaln_basis", ["adaln_basis"], inv,
                       expect_loaded=False)
    assert bad["verdict"] == "NONCOMPUTE_LOADED", bad
    return "noncompute classified with reason"


print("== _selftest_registry_review ==")
check("same_source gates kind/options/version", t_same_source_kind_options_version)
check("recipe switch clears catalog, gen monotonic", t_recipe_switch_clears_catalog_monotonic)
check("attach replace drops stale records", t_attach_replace_drops_stale)
check("resident catalog change refused", t_attach_changed_while_resident_refused)
check("unknown unit ids rejected", t_unknown_unit_ids_rejected)
check("purge/mixed/error needs recover", t_purge_mixed_error_needs_recover)
check("recover needs measured proof", t_recover_needs_measured_proof)
check("deepcopy isolation", t_deepcopy_isolation)
check("snapshot revision/counts/bytes", t_snapshot_revision_counts_bytes)
check("header rejects bad inputs", t_header_rejects_bad_inputs)
check("signature pins dtype+shape", t_signature_covers_shape_dtype)
check("audit refuses meta-missing", t_audit_rejects_meta_missing)
check("noncompute catalog", t_noncompute_catalog)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
