"""probe 20：单 unit 选择性 CPU 物化 + audit + purge/reload（CPU only）。

默认 --units=dit.blocks.0（0.387GB 常驻，staging 峰值约 2x）。
预估 CPU 峰值 < 1.5GB（含 torch）。不触 GPU。禁止 all/超过 2 units。

流程：check_transition -> 仅读所选 keys（access log 作证）-> strict 绑定 ->
audit(ok) -> 未选零 storage -> purge（槽位保留）-> reload（兄弟 ptr 不变）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import weights_common as C  # noqa: E402


def main() -> int:
    # CPU 钳制必须在任何 torch/safe_open import 之前：CUDA_VISIBLE_DEVICES='-1'
    #（远端实测：PS 设 '' 无效，'-1' 才隐藏 GPU，主代理核实）。
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    ap = C.base_parser("weights probe 20: selective CPU materialize")
    ap.add_argument("--compute-dtype", default="bfloat16")
    args = ap.parse_args()
    t0 = C.rss_bytes()
    report: dict = {"probe": "weights_20_block0_cpu",
                    "units": C.parse_units(args.units),
                    "cuda_visible_devices": "-1"}
    rc = 0
    try:
        _probe20_body(args, t0, report)
        report["verdict"] = "ok"
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        if report.get("verdict") not in ("unverified", "refused"):
            report["verdict"] = "refused"
            report["error"] = "SystemExit: %s" % e
    except BaseException as e:  # noqa: BLE001
        rc = 1
        report["verdict"] = "failed"
        report["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        C.write_report(report, args.out, "weights_probe_20.json")
    return rc


def _probe20_body(args, t0, report) -> None:
    units = report["units"]
    if len(units) > 2:
        raise SystemExit("probe 20 allows at most 2 units (got %d)" % len(units))

    # 版本 schema 门：不符不许 ok（unverified + exit 2；外层 finally 写报告）
    env = C.env_info(include_cuda=False)
    C.require_schema(env, report)
    report["env"] = env

    from dit_svc import weights as W
    from dit_svc.model_adapters import H3ComfyAdapter

    tensors, _ = W.read_safetensors_header(args.header)
    shapes = {k: tuple(v["shape"]) for k, v in tensors.items()}
    cat = W.build_catalog_from_header(
        args.model_id, tensors, "dit",
        noncompute_exact=["adaln_basis", "adaln_mean"],
        noncompute_reason=(
            "checkpoint-only keys with no forward consumer in "
            "minimax/model.py; never materialized, never counted resident."))
    cat.check_cover(list(tensors.keys()))
    resolved = W.resolve_units(cat, "units", units)
    report["resolved"] = resolved
    # resolve 后预算（range 可能展开成全量）：unit 数 + 源端字节
    report["budget"] = C.enforce_probe_budget(cat, resolved)

    from safetensors import safe_open
    with safe_open(args.header, framework="pt") as fh:
        metadata = dict(fh.metadata() or {})

    import torch
    assert torch.cuda.is_initialized() is False, "cuda already initialized"
    report["comfy_import_guard"] = C.prepare_comfy_import(args.comfy_root,
                                                          cpu=True)
    import comfy.model_detection as detection
    import comfy.ops as ops
    assert torch.cuda.is_initialized() is False, "comfy import touched CUDA"
    report["cuda_initialized_after_import"] = False
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        sys.path.insert(0, os.path.join(args.comfy_root, "comfy", "ldm"))
        from minimax.model import MiniMaxH3Model  # type: ignore

    adapter = H3ComfyAdapter(torch_mod=torch, ops_mod=ops,
                             model_cls=MiniMaxH3Model,
                             detection_mod=detection)
    unet_config, _ = adapter.detect_unet_config(shapes, metadata)
    quant_config, quant_rep = adapter.quant_config_from_metadata(metadata)
    report["quant"] = quant_rep
    compute_dtype = {"bfloat16": torch.bfloat16,
                     "float32": torch.float32}[args.compute_dtype]
    model = adapter.build_skeleton(cat, unet_config, quant_config,
                                   compute_dtype, device="meta")

    reg = W.WeightRegistry()
    reg.register_recipe(W.make_recipe(args.model_id, "dit", args.header))
    reg.attach_catalog(cat)

    # 选择性读：只 open 所选 keys
    from safetensors import safe_open as _so
    access_log: list = []
    key_tensors: dict = {}
    with _so(args.header, framework="pt") as fh:
        for uid in resolved:
            for k in cat.units[uid].checkpoint_keys:
                key_tensors[k] = fh.get_tensor(k)
                access_log.append(k)
    report["keys_read"] = len(access_log)
    report["keys_read_sample"] = access_log[:5]
    # 未选大 key 零物化：access_log 必须恰为所选 keys
    want_read = set()
    for uid in resolved:
        want_read |= set(cat.units[uid].checkpoint_keys)
    assert set(access_log) == want_read, "selective read violated"

    # 严格绑定
    reg.check_transition("disk2ram", args.model_id, resolved)
    bind_reports = {}
    for uid in resolved:
        sub = {k: v for k, v in key_tensors.items()
               if k in cat.units[uid].checkpoint_keys}
        rep = adapter.materialize_unit(
            model, uid, sub, device="cpu",
            expected_keys=cat.units[uid].checkpoint_keys)
        audit = adapter.audit_after_bind(cat, model, uid, rep, expect_loaded=True)
        bind_reports[uid] = {"bound": len(rep["bound_keys"]),
                             "waived": rep["waived"],
                             "quant_formats": rep["quant_formats"],
                             "audit": audit}
        assert audit["verdict"] == "ok", audit
        inv = W.storage_inventory(adapter.resident_tensors(model, uid),
                                  owner=uid).to_summary()
        reg.set_unit_loaded(args.model_id, uid, "cpu",
                            resident_bytes_cpu=inv["total_cpu"])
    report["bind"] = bind_reports

    # 未选 unit 零 storage
    for uid in cat.unit_ids():
        if uid in resolved:
            continue
        inv = W.storage_inventory(adapter.resident_tensors(model, uid),
                                  owner=uid).to_summary()
        assert inv["total_cpu"] == 0 and inv["total_cuda"] == 0, (uid, inv)
    report["unselected_zero_storage"] = True

    # 兄弟 storage 身份（reload 前后比对用；仅多 unit 时存在兄弟）
    def ptrs(uid):
        return sorted(e["storage_id"]
                      for e in W.storage_inventory(
                          adapter.resident_tensors(model, uid),
                          owner=uid).to_summary()["entries"])
    sib = resolved[1] if len(resolved) > 1 else None
    ptr_before = ptrs(sib) if sib else None

    # purge 第一个 unit（purgefromram 门卫：cpu 可 purge）
    reg.check_transition("purgefromram", args.model_id, [resolved[0]])
    adapter.purge_unit(model, resolved[0])
    # staging 释放证明：只针对被 purge 的 unit。仍驻留兄弟（如 sib）的张量
    # 被模型持有是合法常驻，不得计入 proof（否则两 unit 即误报 leak）。
    purged_keys = set(cat.units[resolved[0]].checkpoint_keys)
    staged = ([v for k, v in key_tensors.items() if k in purged_keys]
              + [v for k, v in sub.items() if k in purged_keys])
    for k in [k for k in key_tensors.keys() if k in purged_keys]:
        del key_tensors[k]
    for k in [k for k in sub.keys() if k in purged_keys]:
        del sub[k]
    proof = C.release_staging_proof(staged)
    del staged
    report["staging_release"] = proof
    assert proof["alive_after_release"] == 0, proof
    audit_p = adapter.audit_after_bind(
        cat, model, resolved[0],
        {"bound_keys": [], "waived": []}, expect_loaded=False)
    assert audit_p["verdict"] == "ok", audit_p
    reg.set_unit_absent(args.model_id, resolved[0])
    assert len(model.blocks) == unet_config["num_layers"], "slot removed!"
    report["purge_audit"] = audit_p

    # reload：只读缺失 keys；兄弟不变
    with _so(args.header, framework="pt") as fh:
        sub = {}
        for k in cat.units[resolved[0]].checkpoint_keys:
            sub[k] = fh.get_tensor(k)
    reg.check_transition("disk2ram", args.model_id, [resolved[0]])
    rep = adapter.materialize_unit(
        model, resolved[0], sub, device="cpu",
        expected_keys=cat.units[resolved[0]].checkpoint_keys)
    audit_r = adapter.audit_after_bind(cat, model, resolved[0], rep,
                                       expect_loaded=True)
    assert audit_r["verdict"] == "ok", audit_r
    report["reload_audit"] = audit_r
    if sib is not None:
        assert ptrs(sib) == ptr_before, "sibling storage replaced by reload"
        report["sibling_ptr_stable"] = True
    else:
        report["sibling_ptr_stable"] = "n/a (single unit)"
    inv = W.storage_inventory(adapter.resident_tensors(model, resolved[0]),
                              owner=resolved[0]).to_summary()
    reg.set_unit_loaded(args.model_id, resolved[0], "cpu",
                        resident_bytes_cpu=inv["total_cpu"])

    report["registry"] = reg.model_state(args.model_id)
    report["rss_delta_bytes"] = C.rss_bytes() - t0
    report["cpu_peak_bytes"] = C.rss_peak_bytes()


if __name__ == "__main__":
    raise SystemExit(main())
