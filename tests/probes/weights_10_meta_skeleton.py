"""probe 10：header-only detect + meta 骨架 + 零 storage 断言（CPU only）。

预估峰值：torch import RSS 约 0.5GB + header JSON（~1MB）+ meta 骨架
Python 对象（~0.1-0.2GB）；CPU 峰值 < 1GB；不触 GPU、不读任何权重体。

用法（远端，主代理运行）：
  python tests/probes/weights_10_meta_skeleton.py --header D:\\xxx.safetensors --comfy-root C:\\...\\ComfyUI
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import weights_common as C  # noqa: E402


def main() -> int:
    ap = C.base_parser("weights probe 10: meta skeleton")
    ap.add_argument("--compute-dtype", default="bfloat16")
    args = ap.parse_args()
    t0 = C.rss_bytes()
    report: dict = {"probe": "weights_10_meta_skeleton", "header": args.header}

    from dit_svc import weights as W
    from dit_svc.model_adapters import H3ComfyAdapter

    # 1. stdlib header（不读权重体）
    tensors, header_len = W.read_safetensors_header(args.header)
    shapes = {k: tuple(v["shape"]) for k, v in tensors.items()}
    dtypes = {k: v["dtype"] for k, v in tensors.items()}
    report["header_keys"] = len(tensors)
    report["header_len"] = header_len

    # 2. 真实 header metadata（safe_open 只取 metadata，不读 tensor）
    from safetensors import safe_open
    with safe_open(args.header, framework="pt") as fh:
        metadata = dict(fh.metadata() or {})
    report["metadata_keys"] = sorted(metadata.keys())

    # 3. 注入真实 comfy 模块
    if not args.comfy_root:
        raise SystemExit("--comfy-root required on remote")
    sys.path.insert(0, args.comfy_root)
    import torch
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
    unet_config, detect_rep = adapter.detect_unet_config(shapes, metadata)
    report["detect"] = detect_rep
    quant_config, quant_rep = adapter.quant_config_from_metadata(metadata)
    report["quant"] = quant_rep

    # 4. catalog（真实 header 全覆盖）
    cat = W.build_catalog_from_header(args.model_id, tensors, "dit")
    cat.check_cover(list(tensors.keys()))
    report["catalog"] = {"units": len(cat.units), "keys": len(cat.key_to_unit)}

    # 5. meta 骨架
    compute_dtype = {"bfloat16": torch.bfloat16,
                     "float32": torch.float32}[args.compute_dtype]
    model = adapter.build_skeleton(cat, unet_config, quant_config,
                                   compute_dtype, device="meta")
    report["topology"] = {
        "num_blocks": len(model.blocks),
        "refiner_blocks": len(model.token_refiner.blocks),
        "has_final_layer": hasattr(model, "final_layer"),
        "has_condition_proj": hasattr(model, "condition_proj"),
        "rope_meta": bool(getattr(model.rope.get_buffer("inv_freq"), "is_meta", False)),
    }

    # 6. 零 storage 断言
    inv = W.storage_inventory(model, owner=args.model_id).to_summary()
    report["inventory"] = {"total_cpu": inv["total_cpu"],
                           "total_cuda": inv["total_cuda"],
                           "meta_tensors": inv["meta_tensors"],
                           "num_tensors_seen": inv["num_tensors_seen"]}
    assert inv["total_cpu"] == 0 and inv["total_cuda"] == 0, inv

    # 7. registry 快照
    reg = W.WeightRegistry()
    rec = W.make_recipe(args.model_id, "dit", args.header)
    reg.register_recipe(rec)
    reg.attach_catalog(cat)
    report["registry"] = reg.model_state(args.model_id)
    report["rss_delta_bytes"] = C.rss_bytes() - t0
    report["verdict"] = "ok"
    C.write_report(report, args.out, "weights_probe_10.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
