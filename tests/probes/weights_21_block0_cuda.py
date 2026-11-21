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
    ap = C.base_parser("weights probe 21: selective CUDA materialize")
    ap.add_argument("--allow-cuda", action="store_true",
                    help="不传则只打印预估峰值后退出，不碰 GPU")
    ap.add_argument("--allow-2-units", action="store_true")
    ap.add_argument("--compute-dtype", default="bfloat16")
    args = ap.parse_args()
    units = C.parse_units(args.units)
    if len(units) > (2 if args.allow_2_units else 1):
        raise SystemExit("probe 21 refuses >%d units" % (2 if args.allow_2_units else 1))

    from dit_svc import weights as W
    tensors, _ = W.read_safetensors_header(args.header)
    shapes = {k: tuple(v["shape"]) for k, v in tensors.items()}
    cat = W.build_catalog_from_header(args.model_id, tensors, "dit")
    resolved = W.resolve_units(cat, "units", units)
    want = cat.expected_bytes_of(resolved)
    estimate = {"expected_resident_bytes": want,
                "expected_peak_gpu_bytes": want + (1 << 30),
                "note": "reserve headroom 1GB; no full-model load"}
    if not args.allow_cuda:
        print("ESTIMATE (no GPU touched):")
        print(__import__("json").dumps(estimate, indent=2))
        return 0

    t0 = C.rss_bytes()
    report: dict = {"probe": "weights_21_block0_cuda", "units": units,
                    "estimate": estimate}
    from dit_svc.model_adapters import H3ComfyAdapter
    from safetensors import safe_open
    with safe_open(args.header, framework="pt") as fh:
        metadata = dict(fh.metadata() or {})

    sys.path.insert(0, args.comfy_root)
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
                if k in ("adaln_basis", "adaln_mean"):
                    continue
                sub[k] = fh.get_tensor(k)
    reg.check_transition("disk2ram", args.model_id, resolved)
    for uid in resolved:
        one = {k: v for k, v in sub.items()
               if k in cat.units[uid].checkpoint_keys}
        rep = adapter.materialize_unit(model, uid, one, device="cpu")
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
        assert adapter.audit_after_bind(
            cat, model, uid, {"bound_keys": [], "waived": []},
            expect_loaded=False)["verdict"] == "ok"
        reg.set_unit_absent(args.model_id, uid)
    torch.cuda.empty_cache()

    report["registry"] = reg.model_state(args.model_id)
    report["rss_delta_bytes"] = C.rss_bytes() - t0
    report["verdict"] = "ok"
    C.write_report(report, args.out, "weights_probe_21.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
