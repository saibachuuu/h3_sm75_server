"""cond server 的 HTTP 层（纯标准库 ThreadingHTTPServer）。

端点：
  GET  /health          -> 编码器状态
  GET  /files/<name>    -> 下载缓存的 cond safetensors
  GET  /                -> 用法说明
  POST /encode          -> JSON {prompt, image_b64s[1-9]|image_b64(兼容), mode, width,
                           height, ref_image_size}（ref2va 1–9 图，fl2va 仅单图）
"""
import base64
import binascii
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.protocol import DEFAULTS, dumps

MAX_BODY = 96 * 1024 * 1024  # b64 图片可到几十 MB


def make_handler(service):
    class CondHandler(BaseHTTPRequestHandler):
        server_version = "h3CondServer/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            try:
                line = fmt % args
            except Exception:
                line = str(args)
            # 轮询静默：远程 dit server 每 2 秒查一次 /health（经 ssh -R 隧道）
            if line.startswith('"GET /health '):
                return
            print("[cond-http] %s - %s" % (self.address_string(), line), flush=True)

        # ---------- helpers ----------
        def _json(self, obj, code=200):
            body = dumps(obj)
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype="application/octet-stream"):
            if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                self._json({"ok": False, "error": "file not found"}, 404)
                return
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", "attachment; filename=%s" % os.path.basename(path))
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        # ---------- GET ----------
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (
                    "h3 cond server (本机文本编码服务)\n"
                    "  GET  /health\n"
                    "  GET  /files/<hash>.safetensors\n"
                    "  POST /encode   {prompt, image_b64, mode, width, height, ref_image_size}\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                self._json({"ok": True, "service": "h3-cond", **service.health()})
            elif path.startswith("/files/"):
                name = os.path.basename(path[len("/files/"):])
                if not name.endswith(".safetensors"):
                    self._json({"ok": False, "error": "bad name"}, 400)
                    return
                self._file(os.path.join(service.cache_dir, name))
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        # ---------- POST ----------
        def do_POST(self):
            if self.path.split("?", 1)[0] != "/encode":
                self._json({"ok": False, "error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY:
                self._json({"ok": False, "error": "bad Content-Length"}, 411)
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"}, 400)
                return

            try:
                prompt = payload.get("prompt")
                b64s = payload.get("image_b64s")
                if b64s is None:
                    b64 = payload.get("image_b64")
                    b64s = [b64] if isinstance(b64, str) and b64 else []
                if (not isinstance(b64s, list) or not (1 <= len(b64s) <= 9)
                        or not all(isinstance(s, str) and s for s in b64s)):
                    raise ValueError("参考图须为 1–9 张（image_b64s；旧单图 image_b64 仍兼容）")
                if payload.get("mode", DEFAULTS["mode"]) == "fl2va" and len(b64s) != 1:
                    raise ValueError("fl2va 仅支持单图，多图请用 ref2va")
                try:
                    image_list = [base64.b64decode(s, validate=True) for s in b64s]
                except (binascii.Error, ValueError):
                    raise ValueError("image_b64 不是合法 base64") from None
                result = service.encode(
                    prompt=prompt,
                    image_bytes=image_list[0],
                    mode=payload.get("mode", DEFAULTS["mode"]),
                    width=int(payload.get("width", DEFAULTS["width"])),
                    height=int(payload.get("height", DEFAULTS["height"])),
                    ref_image_size=payload.get("ref_image_size", DEFAULTS["ref_image_size"]),
                    image_list=image_list,
                )
                result["ok"] = True
                result["file"] = "/files/" + result["file"]
                self._json(result, 200)
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, 400)
            except RuntimeError as e:
                self._json({"ok": False, "error": str(e)}, 503)
            except Exception as e:
                traceback.print_exc()
                self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

    return CondHandler


def serve(service, host, port):
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print("[cond] listening on http://%s:%d  (cache: %s)"
          % (host, port, service.cache_dir), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[cond] bye", flush=True)
    finally:
        httpd.server_close()
