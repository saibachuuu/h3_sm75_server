"""入口 A（本机）：H3 条件编码服务。

用法（必须用本地 ComfyUI venv 的 python 跑，embed_h3 同款环境）：
  C:\\Users\\88888888\\comfyui\\.venv\\Scripts\\python.exe h3_cond_server.py [--port 8189]

职责：
  - 启动时加载 Qwen3-VL-32B int4 文本编码器并**常驻**（一次 ~几十秒）；
  - POST /encode 同步编码 ref2va/fl2va 条件（首次 ~22s，内容哈希缓存命中即时）；
  - 输出 safetensors 契约 = infer_h3_turing.read_cond 所需（含 ref_image/first_frame）。

远程 dit server 方案 A 的回调路径：ssh -R 9310:127.0.0.1:8189 之后（或跑 tunnel.ps1），
远程访问 http://127.0.0.1:9310 即本服务。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.dirname(HERE)  # work-on-comfyui（embed_server.py 所在）
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if WORK_ROOT not in sys.path:
    sys.path.insert(0, WORK_ROOT)

from common.protocol import DEFAULT_COND_PORT  # noqa: E402

DEFAULT_MODEL = r"D:\qwen3vl_32b_minimax_h3_int4_convrot.safetensors"


def main():
    ap = argparse.ArgumentParser(description="H3 条件编码服务（常驻文本编码器）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_COND_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Qwen3-VL-32B int4 convrot 权重")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cond_cache"),
                    help="cond safetensors 内容哈希缓存目录")
    ap.add_argument("--no-int4-mma", action="store_true", help="关闭 int4 MMA patch（调试用）")
    args = ap.parse_args()

    # embed_h3 模式：先解析完自己的参数，再把 argv 清干净，避免 comfy.cli_args 读到外部参数
    sys.argv = [sys.argv[0]]

    import embed_server  # noqa: F401  触发 COMFY_DIR / sys.path 副作用
    from cond_svc.encoder_service import EncoderService
    from cond_svc.http_api import serve

    svc = EncoderService(args.model, args.cache_dir, use_int4_mma=not args.no_int4_mma)
    print("[cond] loading encoder in background: %s" % args.model, flush=True)
    # 先起 HTTP 再后台加载：加载期间 /health 报 loading、/encode 返回 503；
    # 否则本机端口未监听时，远程 WebUI 会把“加载中”误判成“隧道不通”。
    import threading
    threading.Thread(target=svc.load, name="cond-loader", daemon=True).start()
    serve(svc, args.host, args.port)


if __name__ == "__main__":
    main()
