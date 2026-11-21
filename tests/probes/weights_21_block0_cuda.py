"""probe 21：单 unit 选择性 CUDA 物化 + 搬运闭环（需 GPU，主代理运行）。

预估 GPU 峰值：blocks.0 常驻 0.387GB + allocator reserve（< 1GB），总量 < 2GB；
CPU staging 峰值 < 1GB（含 torch）。2080Ti 22GB 安全。
全模型常驻加载不做：拒绝 all，默认只 1 个 unit（--allow-2-units 才放行第 2 个）。

流程：meta 骨架 -> 选读 keys -> cpu 严格绑定 -> ram2vram 上卡 ->
cuda inventory 核对 -> vram2ram 回迁 -> purge -> 报告 peak（memory_stats）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import weights_common as C  # noqa: E402


def main() -> int:
    # 21 是显式 GPU probe：不清 CUDA_VISIBLE_DEVICES，不设 cpu。
    # 但仍清 argv 防 comfy.cli_args 误读 probe 参数。
    ap = C.base_parser("weights probe 21: selective CUDA materialize")
    ap.add_argument("--allow-cuda", action="store_true",
                    help="不传则只打印预估峰值后退出，不碰 GPU")
    ap.add_argument("--allow-2-units", action="store_true")
    ap.add_argument("--compute-dtype", default="bfloat16")
    args = ap.parse_args()
    t0 = C.rss_bytes()
    report: dict = {"probe": "weights_21_block0_cuda",
                    "units": C.parse_units(args.units)}
    rc = 0
    try:
        _probe21_body(args, t0, report)
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
        if report.get("estimate") is not None or args.allow_cuda:
            C.write_report(report, args.out, "weights_probe_21.json")
    return rc


def _probe21_body(args, t0, report) -> None:
    units = report["units"]
    max_units = 2 if args.allow_2_units else 1
    if len(units) > max_units:
        raise SystemExit("probe 21 refuses >%d units" % max_units)

    from dit_svc import weights as W
    tensors, _ = W.read_safetensors_header(args.header)
    shapes = {k: tuple(v["shape"]) for k, v in tensors.items()}
    cat = W.build_catalog_from_header(
        args.model_id, tensors, "dit",
        noncompute_exact=["adaln_basis", "adaln_mean"],
        noncompute_reason=(
            "checkpoint-only keys with no forward consumer in "
            "minimax/model.py; never materialized, never counted resident."))
    resolved = W.resolve_units(cat, "units", units)
    # resolve 后预算（range 可能展开成全量）：unit 数 + 源端字节
    report["budget"] = C.enforce_probe_budget(cat, resolved)
    want = cat.expected_bytes_of(resolved)
    estimate = {"expected_resident_bytes": want,
                "expected_peak_gpu_bytes": want + (1 << 30),
                "note": "reserve headroom 1GB; no full-model load"}
    report["estimate"] = estimate
    if not args.allow_cuda:
        print("ESTIMATE (no GPU touched):")
        print(__import__("json").dumps(estimate, indent=2))
        return

    # 版本 schema 门：不符不许 ok（unverified + exit 2；外层 finally 写报告）
    env = C.env_info(include_cuda=False)
    C.require_schema(env, report)
    report["env"] = env
    from dit_svc.model_adapters import H3ComfyAdapter
    from safetensors import safe_open
    with safe_open(args.header, framework="pt") as fh:
        metadata = dict(fh.metadata() or {})

    sys.path.insert(0, args.comfy_root)
    prog = sys.argv[0] if sys.argv else "probe"
    sys.argv = [prog]  # 防 comfy.cli_args 误读 probe 参数（21 需 CUDA，不设 cpu）
    import torch
    assert torch.cuda.is_available(), "cuda not available"
    import comfy.model_detection as detection
    import comfy.ops as ops
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

    torch.cuda.reset_peak_memory_stats(0)
    with safe_open(args.header, framework="pt") as fh:
        sub = {}
        for uid in resolved:
            for k in cat.units[uid].checkpoint_keys:
                sub[k] = fh.get_tensor(k)
    reg.check_transition("disk2ram", args.model_id, resolved)
    for uid in resolved:
        one = {k: v for k, v in sub.items()
               if k in cat.units[uid].checkpoint_keys}
        rep = adapter.materialize_unit(
            model, uid, one, device="cpu",
            expected_keys=cat.units[uid].checkpoint_keys)
        assert adapter.audit_after_bind(cat, model, uid, rep,
                                        expect_loaded=True)["verdict"] == "ok"
        inv = W.storage_inventory(adapter.resident_tensors(model, uid),
                                  owner=uid).to_summary()
        reg.set_unit_loaded(args.model_id, uid, "cpu",
                            resident_bytes_cpu=inv["total_cpu"])

    # ram2vram -> cuda inventory 核对
    reg.check_transition("ram2vram", args.model_id, resolved)
    for uid in resolved:
        adapter.move_unit(model, uid, "cuda:0")
        inv = W.storage_inventory(adapter.resident_tensors(model, uid),
                                  owner=uid).to_summary()
        assert inv["total_cuda"] > 0 and inv["total_cpu"] == 0, inv
        reg.set_unit_loaded(args.model_id, uid, "cuda:0",
                            resident_bytes_cuda=inv["total_cuda"])
    torch.cuda.synchronize(0)
    report["cuda_peak"] = torch.cuda.memory_stats(0).get(
        "allocated_bytes.all.peak", -1)
    report["cuda_reserved_peak"] = torch.cuda.memory_stats(0).get(
        "reserved_bytes.all.peak", -1)

    # vram2ram 回迁 -> purge
    reg.check_transition("vram2ram", args.model_id, resolved)
    for uid in resolved:
        adapter.move_unit(model, uid, "cpu")
        inv = W.storage_inventory(adapter.resident_tensors(model, uid),
                                  owner=uid).to_summary()
        assert inv["total_cpu"] > 0 and inv["total_cuda"] == 0, inv
        reg.set_unit_loaded(args.model_id, uid, "cpu",
                            resident_bytes_cpu=inv["total_cpu"])
    reg.check_transition("purgefromram", args.model_id, resolved)
    for uid in resolved:
        adapter.purge_unit(model, uid)
        # 只针对被 purge 的 unit 取证；仍驻留兄弟合法常驻，不得计入
        purged_keys = set(cat.units[uid].checkpoint_keys)
        staged = [v for k, v in sub.items() if k in purged_keys]
        for k in [k for k in sub.keys() if k in purged_keys]:
            del sub[k]
        proof = C.release_staging_proof(staged)
        del staged
        report.setdefault("staging_release", {})[uid] = proof
        assert proof["alive_after_release"] == 0, proof
        assert adapter.audit_after_bind(
            cat, model, uid, {"bound_keys": [], "waived": []},
            expect_loaded=False)["verdict"] == "ok"
        reg.set_unit_absent(args.model_id, uid)
    torch.cuda.empty_cache()

    report["registry"] = reg.model_state(args.model_id)
    report["rss_delta_bytes"] = C.rss_bytes() - t0
    report["cpu_peak_bytes"] = C.rss_peak_bytes()


if __name__ == "__main__":
    raise SystemExit(main())
