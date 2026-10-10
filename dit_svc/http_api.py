"""dit server 的 HTTP 层：WebUI 静态页 + 任务 API + 成果文件（含 Range 断点）。"""
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.protocol import DEFAULTS, SAMPLERS, SCHEDULERS, dumps

WEBUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui")
MAX_BODY = 96 * 1024 * 1024


def _veda_probe_snapshot():
    """--veda-debug 的运行期探针（准入判定统计 / 稀疏真跑次数）。"""
    try:
        from dit_svc import veda_attn
        return dict(veda_attn.VEDA_PROBE) or None
    except Exception:
        return None


def make_handler(runtime, queue, ctx):
    out_dir = ctx["out_dir"]

    class DitHandler(BaseHTTPRequestHandler):
        server_version = "h3DitServer/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            try:
                line = fmt % args
            except Exception:
                line = str(args)
            # 轮询静默：WebUI 每 2 秒 GET /api/status + /api/jobs，会淹没真实事件
            # （提交/阶段/出错）。其余请求（POST、单任务详情、outputs、静态页）照常记。
            if line.startswith('"GET /api/status ') or line.startswith('"GET /api/jobs '):
                return
            print("[http] %s - %s" % (self.address_string(), line), flush=True)

        # ---------------- helpers ----------------
        def _json(self, obj, code=200):
            body = dumps(obj)
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _raw(self, body, ctype, code=200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype=None, allow_range=True):
            if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                self._json({"ok": False, "error": "文件不存在"}, 404)
                return
            size = os.path.getsize(path)
            ctype = ctype or "application/octet-stream"
            start, end = 0, size - 1
            code = 200
            range_hdr = self.headers.get("Range") if allow_range else None
            if range_hdr and range_hdr.startswith("bytes="):
                try:
                    spec = range_hdr[len("bytes="):].split(",")[0].strip()
                    a, _, b = spec.partition("-")
                    start = int(a) if a else 0
                    end = int(b) if b else size - 1
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        raise ValueError
                    code = 206
                except ValueError:
                    start, end, code = 0, size - 1, 200
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _read_json_body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY:
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None

        def _status(self):
            from dit_svc import cond_client
            torch = getattr(runtime, "torch", None)
            vram = None
            if torch is not None:
                try:
                    free, total = torch.cuda.mem_get_info()
                    vram = {
                        "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                        "free_gb": round(free / 1e9, 2),
                        "total_gb": round(total / 1e9, 2),
                    }
                except Exception:
                    vram = None
            snap = queue.snapshot(limit=1)
            return {
                "ready": runtime.ready,
                "error": runtime.error,
                "stage": runtime.stage,
                "stage_elapsed": round(runtime.stage_elapsed(), 1),
                "attn": runtime.attn,
                "w4a8_backend": getattr(runtime.args, "w4a8_backend", "?"),
                "mlp_chunk": getattr(runtime.args, "mlp_chunk", 0),
                "ffmpeg": runtime.ffmpeg,
                "cond": cond_client.health(ctx["cond_url"]) or
                        {"ok": False, "reachable": False, "url": ctx["cond_url"]},
                "cond_url": ctx["cond_url"],
                "vram": vram,
                "counts": snap["counts"],
                "running": snap["running"],
                "veda_probe": _veda_probe_snapshot(),
                "server_uptime": round(__import__("time").time() - runtime.started_at, 1),
            }

        # ---------------- GET ----------------
        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                p = os.path.join(WEBUI_DIR, "index.html")
                if not os.path.isfile(p):
                    self._json({"ok": False, "error": "index.html missing"}, 500)
                    return
                with open(p, "rb") as f:
                    self._raw(f.read(), "text/html; charset=utf-8")
            elif path == "/api/status":
                try:
                    self._json(self._status())
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 500)
            elif path == "/api/defaults":
                self._json({"defaults": DEFAULTS, "sampler": SAMPLERS, "scheduler": SCHEDULERS})
            elif path == "/api/jobs":
                self._json(queue.snapshot())
            elif path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                job = queue.get(job_id)
                if job is None:
                    self._json({"ok": False, "error": "任务不存在"}, 404)
                else:
                    self._json(job.public(with_log=True))
            elif path.startswith("/outputs/"):
                name = os.path.basename(path[len("/outputs/"):])
                if "/" in name or "\\" in name or ".." in name:
                    self._json({"ok": False, "error": "bad name"}, 400)
                    return
                p = os.path.join(out_dir, name)
                ctype = ("video/mp4" if name.endswith(".mp4")
                         else "application/json" if name.endswith(".json")
                         else "image/png" if name.endswith(".png") else None)
                self._file(p, ctype)
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        # ---------------- POST ----------------
        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/jobs":
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    from dit_svc.job_queue import BusyError
                    job = queue.submit(payload)
                    self._json({"ok": True, "job_id": job.id}, 201)
                except BusyError as e:
                    self._json({"ok": False, "error": str(e)}, 409)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)
            elif path == "/api/jobs/cond":
                # 直接喂 cond：{cond_b64 | cond_file, image_b64s? 0–9, mode, width,
                # height, length, steps, sampler, scheduler, seed, refine_steps,
                # refine_denoise, ref_image_size, sigmas?, fps?, prompt?}
                # cond_file 相对路径按 out_dir 解析；任务内不回调 cond server。
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    from dit_svc.job_queue import BusyError
                    job = queue.submit_cond(payload)
                    self._json({"ok": True, "job_id": job.id}, 201)
                except BusyError as e:
                    self._json({"ok": False, "error": str(e)}, 409)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)
            elif path == "/api/vae_probe":
                # 只跑视频 VAE 编解码（不进 DiT、不采样）：
                # {image_b64?, decode?:bool, latent?:noise|zeros, write_mp4?, write_png?,
                #  mode?, ref_image_size?, width, height, length, fps?, seed?}
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    from dit_svc.job_queue import BusyError
                    job = queue.submit_vae_probe(payload)
                    self._json({"ok": True, "job_id": job.id}, 201)
                except BusyError as e:
                    self._json({"ok": False, "error": str(e)}, 409)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)
            elif path == "/api/jobs/drop":
                payload = self._read_json_body() or {}
                job_id = payload.get("id")
                job, err = queue.drop(job_id)
                if err:
                    self._json({"ok": False, "error": err}, 400)
                else:
                    self._json({"ok": True, "job": job})
            elif path == "/api/jobs/clear":
                n = queue.clear_queued()
                self._json({"ok": True, "dropped": n})
            elif path == "/api/warmup":
                # {tokens: int 16–40000, repeat?: 1–50} 或 {level, repeat?}；
                # 进串行队列，结果轮询 /api/jobs/<id>（与视频任务同口径）。
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    from dit_svc.job_queue import BusyError
                    from dit_svc.veda_attn import WARMUP_TIERS
                    tokens = payload.get("tokens")
                    if tokens is None and payload.get("level"):
                        tier = WARMUP_TIERS.get(payload.get("level"))
                        if tier is None:
                            raise ValueError("level 须为 probe|tier1|tier2|tier3")
                        tokens = tier[0]
                    if not isinstance(tokens, int) or not (16 <= tokens <= 40000):
                        raise ValueError("tokens 须为 16–40000 的整数（或 level 挡位）")
                    repeat = payload.get("repeat", 1)
                    if not isinstance(repeat, int) or not (1 <= repeat <= 50):
                        raise ValueError("repeat 须为 1–50 的整数")
                    compare = payload.get("compare")
                    if compare not in (None, "", "int8"):
                        raise ValueError("compare 只支持 \"int8\"（同一对 Q/K/V 上跟稠密 int8 对比）")
                    job = queue.submit_warmup(tokens, repeat=repeat, compare=compare or None,
                                              profile=bool(payload.get("profile")))
                    self._json({"ok": True, "job_id": job.id, "tokens": tokens,
                                "repeat": repeat, "compare": compare or None,
                                "profile": bool(payload.get("profile"))}, 201)
                except BusyError as e:
                    self._json({"ok": False, "error": str(e)}, 409)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)
            else:
                self._json({"ok": False, "error": "not found"}, 404)

    return DitHandler


def serve(runtime, queue, ctx, host, port):
    httpd = ThreadingHTTPServer((host, port), make_handler(runtime, queue, ctx))
    print("[http] WebUI on http://%s:%d/" % (host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[http] bye", flush=True)
    finally:
        httpd.server_close()
