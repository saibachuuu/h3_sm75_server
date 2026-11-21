"""入口 B（远程）：H3 DiT 生成服务 + WebUI（连续生成，模型常驻）。

用法（远程 python_embeded）：
  python h3_dit_server.py [--port 8190] [--defaults-file paths.json] [--out-dir outputs]

推荐的本机连接方式（一条命令同时开双向隧道；或直接跑本目录 tunnel.ps1）：
  ssh -N -p 56622 -L 8190:127.0.0.1:8190 -R 9310:127.0.0.1:8189 sheepserver@bwg.aki.pm
  浏览器打开 http://127.0.0.1:8190/ 即 WebUI；远程回调 127.0.0.1:9310 -> 本机 cond server 8189。

说明：
  - 依赖 h3-runner 的 infer_h3_turing.py / h3_mlp_chunk.py 作为库（--runner-dir 自动探测）；
  - **不含 warmup 逻辑**（DiT 部分 warmup 已按需求删除）；
  - 启动不加载权重、不主动安装 attention 优化；所有模型动作通过 API 执行。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common.protocol import DEFAULT_COND_URL, DEFAULT_DIT_PORT  # noqa: E402

DEFAULT_UNET = r"D:\10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"
DEFAULT_VIDEO_VAE = r"D:\minimax_h3_video_vae_int8_convrot.safetensors"
DEFAULT_VIDEO_VAE_LIGHT = r"D:\minimax_h3_lynnreal_light_vae_int8_convrot.safetensors"
DEFAULT_AUDIO_VAE = r"D:\minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_VEDA_BUNDLE = r"D:\minimax_h3_r2va_veda_preview_fp8.safetensors"

def find_runner_dir(explicit=None):
    """探测 infer_h3_turing.py 所在目录：显式参数 > ../h3-runner（本机布局） > ..（远程 Desktop 布局）。"""
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.join(os.path.dirname(HERE), "h3-runner"))
    cands.append(os.path.dirname(HERE))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "infer_h3_turing.py")):
            return os.path.abspath(c)
    return None


def parse_args(argv=None):
    """Process settings only; pin paths before runner changes cwd."""
    ap = argparse.ArgumentParser(description="H3 DiT 空壳服务（模型与 attention 通过 API 控制）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_DIT_PORT)
    ap.add_argument("--runner-dir", default=None, help="h3-runner 目录（默认自动探测）")
    ap.add_argument("--cond-url", default=DEFAULT_COND_URL,
                    help="本机 cond server（经 ssh -R 9310 -> 8189 的回调地址）")
    ap.add_argument("--defaults-file", help="JSON model recipes (no automatic loading)")
    ap.add_argument("--w4a8-backend", "--w4a8_backend", dest="w4a8_backend",
                    choices=["turingutils", "cublas"], default="turingutils")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args(argv)
    args.out_dir = os.path.abspath(args.out_dir)
    if args.runner_dir:
        args.runner_dir = os.path.abspath(args.runner_dir)
    args.model_recipes = {}
    if args.defaults_file:
        args.defaults_file = os.path.abspath(args.defaults_file)
        try:
            with open(args.defaults_file, encoding="utf-8-sig") as source:
                config = json.load(source)
            if not isinstance(config, dict) or set(config) != {"models"}:
                raise ValueError("defaults JSON must contain only a models object")
            if not isinstance(config["models"], dict):
                raise ValueError("models must be a JSON object")
            known = {"dit", "vae.video", "vae.audio", "te", "veda.bundle"}
            for model, recipe in config["models"].items():
                if model not in known or not isinstance(recipe, dict):
                    raise ValueError("invalid model recipe: %s" % model)
                for key in ("path", "mmproj"):
                    value = recipe.get(key)
                    if value is not None:
                        if not isinstance(value, str) or not value:
                            raise ValueError("%s.%s must be a nonempty path" % (model, key))
                        recipe[key] = os.path.abspath(value)
            args.model_recipes = config["models"]
        except (OSError, ValueError) as exc:
            ap.error("invalid --defaults-file: %s" % exc)
    args.attn = "none"
    args.mlp_chunk = 8192
    return args


def main():
    args = parse_args()
    # argv 清干净，避免 comfy.cli_args 读到外部参数（embed_h3 同款卫生）
    sys.argv = [sys.argv[0]]

    runner_dir = find_runner_dir(args.runner_dir)
    if not runner_dir:
        raise SystemExit("找不到 infer_h3_turing.py，请用 --runner-dir 指定 h3-runner 目录")
    sys.path.insert(0, runner_dir)
    import infer_h3_turing as rt  # noqa: F401  副作用：COMFY_DIR/os.chdir/sys.path 引导
    print("[entry] runner lib: %s" % os.path.join(runner_dir, "infer_h3_turing.py"), flush=True)

    from dit_svc.runtime import Runtime
    from dit_svc.job_queue import QueueManager
    from dit_svc.http_api import serve

    os.makedirs(args.out_dir, exist_ok=True)
    runtime = Runtime(args, rt)
    ctx = {
        "runtime": runtime,
        "cond_url": args.cond_url,
        "out_dir": os.path.abspath(args.out_dir),
        "entry_args": args,
    }
    queue = QueueManager(runtime, ctx)

    # Only initialize the environment. Never start the legacy model loader.
    try:
        runtime.ensure_env()
    except Exception:
        # Recorded on runtime; keep diagnostic APIs available.
        pass

    print("", flush=True)
    print("=" * 72, flush=True)
    print(" H3 DiT server | WebUI: http://%s:%d/" % (args.host, args.port), flush=True)
    print(" attention=none (factory; no override installed)  cond=%s" % args.cond_url, flush=True)
    print("[entry] 空壳启动：未加载任何权重；环境就绪=%s" % runtime.env_ready, flush=True)
    print(" 本机隧道（一条命令开双向）:", flush=True)
    print("   ssh -p <port> -L %d:localhost:%d -R 9310:localhost:8189 sheepserver@<host>"
          % (args.port, args.port), flush=True)
    print("=" * 72, flush=True)

    serve(runtime, queue, ctx, args.host, args.port)


if __name__ == "__main__":
    main()
