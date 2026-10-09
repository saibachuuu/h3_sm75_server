"""入口 B（远程）：H3 DiT 生成服务 + WebUI（连续生成，模型常驻）。

用法（远程 python_embeded）：
  python h3_dit_server.py [--port 8190] [--attn int8] [--out-dir outputs]

推荐的本机连接方式（一条命令同时开双向隧道）：
  ssh -p 56622 -L 8190:localhost:8190 -R 9310:localhost:8189 sheepserver@bwg.aki.pm
  浏览器打开 http://127.0.0.1:8190/ 即 WebUI；远程回调 127.0.0.1:9310 -> 本机 cond server 8189。

说明：
  - 依赖 h3-runner 的 infer_h3_turing.py / h3_mlp_chunk.py 作为库（--runner-dir 自动探测）；
  - **不含 warmup 逻辑**（DiT 部分 warmup 已按需求删除）；
  - attention 默认 int8；sage/w8a8 仅为调试档（sage 全尺寸已知 0/2 崩，见事故报告）。
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common.protocol import DEFAULT_COND_URL, DEFAULT_DIT_PORT  # noqa: E402

DEFAULT_UNET = r"D:\10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"
DEFAULT_VIDEO_VAE = r"D:\minimax_h3_video_vae_int8_convrot.safetensors"
DEFAULT_AUDIO_VAE = r"D:\minimax_h3_audio_vae_fp32.safetensors"


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


def main():
    ap = argparse.ArgumentParser(description="H3 DiT 生成服务（模型常驻 + 串行队列 + WebUI）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_DIT_PORT)
    ap.add_argument("--runner-dir", default=None, help="h3-runner 目录（默认自动探测）")
    ap.add_argument("--cond-url", default=DEFAULT_COND_URL,
                    help="本机 cond server（经 ssh -R 9310 -> 8189 的回调地址）")
    ap.add_argument("--unet", default=DEFAULT_UNET)
    ap.add_argument("--video_vae", default=DEFAULT_VIDEO_VAE)
    ap.add_argument("--audio_vae", default=DEFAULT_AUDIO_VAE)
    ap.add_argument("--attn", choices=["int8", "sage", "w8a8", "sdpa"], default="int8",
                    help="服务级 attention 后端（int8=已验证默认；sage/w8a8=调试档）")
    ap.add_argument("--w4a8_backend", choices=["turingutils", "cublas"], default="turingutils")
    ap.add_argument("--mlp_chunk", type=int, default=8192)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()

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

    # 模型加载放后台线程：HTTP 先起来（WebUI 显示加载进度），ready 前拒绝提交
    loader = threading.Thread(target=runtime.load, name="h3-loader", daemon=True)
    loader.start()

    print("", flush=True)
    print("=" * 72, flush=True)
    print(" H3 DiT server | WebUI: http://%s:%d/" % (args.host, args.port), flush=True)
    print(" attention=%s  w4a8=%s  mlp_chunk=%d  cond=%s"
          % (args.attn, args.w4a8_backend, args.mlp_chunk, args.cond_url), flush=True)
    print(" 本机隧道（一条命令开双向）:", flush=True)
    print("   ssh -p <port> -L %d:localhost:%d -R 9310:localhost:8189 sheepserver@<host>"
          % (args.port, args.port), flush=True)
    print("=" * 72, flush=True)

    serve(runtime, queue, ctx, args.host, args.port)


if __name__ == "__main__":
    main()
