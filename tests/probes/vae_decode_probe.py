"""VAE 设备/解码定位探针：CPU 构造 -> .to(cuda) -> decode。

用于复现 `q, k, and v must be on the same CUDA device`：打印构造后、上卡后
的参数/buffer 设备分布，再跑一个与 22 帧对应的最小 video latent 解码。

只读 VAE 文件，不启动服务，不改安装依赖。
用法：
  python tests/probes/vae_decode_probe.py \
      --vae C:\\Users\\sheepserver\\minimax_h3_lynnreal_light_vae_int8_convrot.safetensors \
      --out C:\\Users\\sheepserver\\Desktop\\_vae_probe.json
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def device_hist(module):
    counts = {}
    for name, parameter in module.named_parameters():
        key = "param:%s:%s" % (parameter.device, parameter.dtype)
        counts[key] = counts.get(key, 0) + 1
    for name, buffer in module.named_buffers():
        key = "buffer:%s:%s" % (buffer.device, buffer.dtype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--construct-device", choices=("cpu", "cuda"),
                        default="cpu")
    args = parser.parse_args()
    out_path = Path(args.out).resolve()
    report = {"probe": "vae_decode_probe", "vae": args.vae, "verdict": "failed"}
    try:
        import h3_dit_server as entry
        runner = entry.find_runner_dir(None)
        if runner:
            sys.path.insert(0, runner)
        sys.argv = ["vae_decode_probe"]
        import infer_h3_turing as rt  # noqa: F401  COMFY_DIR/sys.path 引导
        import torch
        import comfy.options
        comfy.options.enable_args_parsing(True)
        import comfy.cli_args
        comfy.cli_args.args.gpu_only = True
        import comfy
        import comfy.sd
        from dit_svc.runtime import _build_video_vae

        state, metadata = rt.stream_load_safetensors_to_dict(args.vae, device="cpu")
        report["checkpoint_tensors"] = len(state)
        build_device = (torch.device("cuda:0") if args.construct_device == "cuda"
                        else torch.device("cpu"))
        vae = _build_video_vae(comfy, state, metadata, build_device, torch)
        state.clear()
        first = vae.first_stage_model
        report["construct_device"] = str(build_device)
        report["after_construct"] = device_hist(first)
        report["decoder_attn_types"] = sorted({
            type(module).__name__ for name, module in first.named_modules()
            if "attn" in name and hasattr(module, "qkv")})

        if args.construct_device != "cuda":
            first.to("cuda:0")
        report["after_to_cuda"] = device_hist(first)

        first.to("cuda:0")
        report["after_to_cuda"] = device_hist(first)

        # 量化 VAE：wrapper 的 device 与内部 _qdata 的 device 可能不一致。
        quant_sample = []
        for name, module in first.named_modules():
            weight = getattr(module, "weight", None)
            if weight is not None and hasattr(weight, "_qdata"):
                quant_sample.append({
                    "module": name,
                    "weight_device": str(weight.device),
                    "qdata_device": str(weight._qdata.device),
                    "qdata_dtype": str(weight._qdata.dtype),
                    "layout": type(weight._layout_cls).__name__,
                })
                if len(quant_sample) >= 5:
                    break
        report["quant_sample"] = quant_sample
        report["quant_count"] = sum(
            1 for _, module in first.named_modules()
            if hasattr(getattr(module, "weight", None), "_qdata"))

        if args.check_only:
            report["verdict"] = "ok"
            return 0

        # 包一层 attention 入口，打印 q/k/v 的真实 device/dtype/类型。
        import comfy.quant_ops as quant_ops
        original_int8 = quant_ops.ck.int8_attention

        def _checked(q, k, v, *a, **kw):
            report.setdefault("attn_calls", []).append({
                "q": [str(q.device), str(q.dtype), type(q).__name__],
                "k": [str(k.device), str(k.dtype), type(k).__name__],
                "v": [str(v.device), str(v.dtype), type(v).__name__],
            })
            return original_int8(q, k, v, *a, **kw)

        quant_ops.ck.int8_attention = _checked

        # 直接检查 decoder 内 attention 的 qkv 权重设备
        qkv_devices = {}
        for name, module in first.named_modules():
            qkv = getattr(module, "qkv", None)
            if qkv is not None and hasattr(qkv, "weight"):
                qkv_devices[name] = str(qkv.weight.device)
        report["qkv_devices_sample"] = dict(list(qkv_devices.items())[:5])
        report["qkv_device_set"] = sorted(set(qkv_devices.values()))

        latent_frames = (args.frames - 1) // 8 + 1
        z = torch.randn(1, 24, latent_frames, 60, 40, device="cuda:0",
                        dtype=torch.bfloat16)
        report["latent_shape"] = list(z.shape)
        with torch.no_grad():
            decoded = vae.decode(z)
        report["decoded_shape"] = list(decoded.shape)
        report["decoded_dtype"] = str(decoded.dtype)
        report["decoded_finite"] = bool(torch.isfinite(decoded.float()).all())
        report["verdict"] = "ok"
    except Exception as error:
        report["error"] = "%s: %s" % (type(error).__name__, error)
        report["traceback"] = traceback.format_exc()[-4000:]
        traceback.print_exc()
    finally:
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(json.dumps({"verdict": report["verdict"],
                          "error": report.get("error"),
                          "out": str(out_path)}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
