"""dit server 的 HTTP 层：WebUI 静态页 + 任务 API + 成果文件（含 Range 断点）。"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.protocol import DEFAULTS, DIRECTOR_MAX_SHOTS, SAMPLERS, SCHEDULERS, dumps
from dit_svc.operations import OperationError

WEBUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui")
MAX_BODY = 96 * 1024 * 1024

# 导演台（镜头序列）前端默认值与上限；契约见仓库外 _ada_director_api.md
DIRECTOR_DEFAULTS = {
    "ref_prev_frames": 1,
    "concat": True,
    "on_error": "stop",
    "max_shots": DIRECTOR_MAX_SHOTS,
    "max_ref_images": 9,
    "length_min": 124,
    "length_max": 362,
}


_RUNTIME_HOLDER = {}   # make_handler 时登记 runtime（_veda_calls_snapshot 用）


def _veda_calls_snapshot():
    """veda 工况探针（路线 B）。

    返回结构（见 veda_attn_star7.status_report）：
      active / backend / backend_report（启动自检结果）
      cumulative —— 跨 run 累计（sparse 次数、dense 原因、compute_fraction）；
                    在 patch.on_cleanup 清零**之前**抓取，所以任务结束后仍在
      current    —— 当前进行中的 run（空闲时全 0）
    返回 None 表示 veda 未启用（或已回退 int8 且 backend 未加载）。
    """
    try:
        runtime = _RUNTIME_HOLDER.get("runtime")
        patch = getattr(runtime, "veda_patch", None) if runtime is not None else None
        from dit_svc.veda_attn_star7 import status_report
        return status_report(patch)
    except Exception:
        return None


def make_handler(runtime, queue, ctx):
    out_dir = ctx["out_dir"]
    _RUNTIME_HOLDER["runtime"] = runtime

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
                "env_ready": getattr(runtime, "env_ready", runtime.ready),
                "compute_ready": getattr(runtime, "compute_ready", runtime.ready),
                "device_healthy": getattr(runtime, "device_healthy", True),
                "api_version": 2,
                "executor": queue.executor_state(),
                "capabilities": runtime.capabilities(),
                "requirements_are_request_dependent": True,
                "weights": runtime.weights_snapshot()["models"],
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
                "veda_calls": _veda_calls_snapshot(),
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
                self._json({"defaults": DEFAULTS, "sampler": SAMPLERS,
                            "scheduler": SCHEDULERS, "director": DIRECTOR_DEFAULTS,
                            "model_recipes": getattr(runtime.args, "model_recipes", {})})
            elif path == "/api/weights":
                self._json(runtime.weights_snapshot())
            elif path.startswith("/api/weights/"):
                model_id = path[len("/api/weights/"):]
                model = runtime.weights_snapshot()["models"].get(model_id)
                if model is None:
                    err = OperationError("MODEL_NOT_FOUND", "未知模型", 404, False)
                    self._json({"ok": False, "error": err.to_dict()}, 404)
                else:
                    self._json({"ok": True, "model": model})
            elif path == "/api/operations":
                try:
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    limit = int(query.get("limit", ["20"])[0])
                    if not 1 <= limit <= 200:
                        raise ValueError("limit must be between 1 and 200")
                    self._json({"ok": True, **queue.operations_snapshot(limit)})
                except ValueError as exc:
                    err = OperationError("INVALID_REQUEST", str(exc), 400, False)
                    self._json({"ok": False, "error": err.to_dict()}, 400)
            elif path.startswith("/api/operations/"):
                operation = queue.get_operation(path[len("/api/operations/"):])
                if operation is None:
                    err = OperationError("OPERATION_NOT_FOUND", "操作不存在", 404, False)
                    self._json({"ok": False, "error": err.to_dict()}, 404)
                else:
                    self._json({"ok": True, "operation": operation.public(include_log=True)})
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
            if path.startswith("/api/operations/") and path.endswith("/cancel"):
                operation_id = path[len("/api/operations/"):-len("/cancel")]
                try:
                    if self.headers.get("Content-Length", "0") != "0":
                        payload = self._read_json_body()
                        if not isinstance(payload, dict):
                            raise OperationError("INVALID_REQUEST", "请求体必须是 JSON 对象", 400, False)
                    self._json({"ok": True, "operation": queue.cancel_operation(operation_id)})
                except OperationError as exc:
                    self._json({"ok": False, "error": exc.to_dict()}, exc.http_status)
            elif path == "/api/jobs":
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
                # refine_denoise, ref_image_size, sigmas?, profile_attn?, prompt?}
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
            elif path == "/api/director":
                # 导演台：镜头序列。{global: {mode, width, height, steps, sampler,
                # scheduler, refine_steps, refine_denoise, sigmas?, seed,
                # ref_prev_frames, concat, on_error}, shots: [{prompt, length,
                # seed?, ref_prev, ref_prev_frames?, image_b64s, image_names?}]}
                # 后续镜头可参考上一镜头的尾帧（帧级接力）。
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    from dit_svc.job_queue import BusyError
                    job = queue.submit_director(payload)
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
            elif path == "/api/te/state":
                # 返回 TE 原子化状态机快照（TE 所在 device / VRAM / 当前步骤）
                self._json({"ok": True, "te_state": runtime.te_state()})

            elif path == "/api/te/step":
                # 原子化 TE 步骤：
                # {action: "vae_swap_out"|"te_load_vram"|"te_offload_ram"|
                #           "preprocess"|"te_encode"|"vae_swap_back"|"reset",
                #  prompt?, image_b64s?, mode?, width?, height?, ref_image_size?,
                #  save_cond?: bool}
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                try:
                    import base64 as _b64
                    snap = queue.snapshot(limit=1)
                    counts = snap.get("counts") or {}
                    if counts.get("running") or counts.get("queued"):
                        self._json({"ok": False,
                                    "error": "队列非空（生成任务在跑或在排队）。TE 交换要独占显存，等它结束再试。"},
                                   409)
                        return

                    action = payload.get("action")
                    if not isinstance(action, str):
                        self._json({"ok": False, "error": "action 须为字符串"}, 400)
                        return

                    step_payload = {}
                    if action in ("preprocess", "te_encode"):
                        raw = payload.get("image_b64s") or []
                        imgs = []
                        for s in raw:
                            if not isinstance(s, str):
                                continue
                            if s.startswith("data:"):
                                s = s.split(",", 1)[1]
                            imgs.append(_b64.b64decode(s))
                        step_payload["image_bytes_list"] = imgs
                        step_payload["mode"] = payload.get("mode", "ref2va")
                        step_payload["width"] = int(payload.get("width", 640))
                        step_payload["height"] = int(payload.get("height", 960))
                        step_payload["ref_image_size"] = payload.get("ref_image_size", "match")
                    if action == "te_encode":
                        step_payload["prompt"] = payload.get("prompt", "")

                    result = runtime.te_step(action, step_payload)

                    # te_encode + save_cond 时把 cond 落盘
                    if action == "te_encode" and payload.get("save_cond"):
                        import os as _os
                        from safetensors.torch import save_file
                        cond_data = result.get("cond")
                        if cond_data is None:
                            self._json({"ok": False, "error": "encode 未产生 cond"}, 500)
                            return
                        name = "cond_step_%d.safetensors" % int(time.time())
                        out_dir = getattr(runtime.args, "out_dir", "outputs")
                        _os.makedirs(out_dir, exist_ok=True)
                        fp = _os.path.join(out_dir, name)
                        tosave = {"cond": cond_data["cond"]}
                        if cond_data.get("minimax_token_tags") is not None:
                            tosave["minimax_token_tags"] = cond_data["minimax_token_tags"]
                        save_file(tosave, fp)
                        result["cond_file"] = name
                        result["cond_bytes"] = _os.path.getsize(fp)

                    self._json(result, 200)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

            elif path == "/api/te/probe":
                # 只把 TE 权重 RAM->VRAM 再退回，不 encode，专门测试远程 GPU 承受力。
                # {cycles?: 1-10}
                payload = self._read_json_body() or {}
                try:
                    snap = queue.snapshot(limit=1)
                    counts = snap.get("counts") or {}
                    if counts.get("running") or counts.get("queued"):
                        self._json({"ok": False,
                                    "error": "队列非空（生成任务在跑或在排队）。TE 交换要独占显存，等它结束再试。"},
                                   409)
                        return
                    cycles = int(payload.get("cycles", 1))
                    if not (1 <= cycles <= 10):
                        raise ValueError("cycles 须在 1–10 之间")
                    results = []
                    for _ in range(cycles):
                        r1 = runtime.te_load_to_vram()
                        r2 = runtime.te_offload_to_ram()
                        results.append({"load": r1, "offload": r2})
                    self._json({"ok": True, "cycles": cycles, "results": results,
                                "te_state": runtime.te_state()}, 200)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

            elif path == "/api/encode":
                # 探针：只算 condition，不进 DiT（完整封装）。
                # 流程：VAE 让位 -> TE 上卡 -> encode -> TE 退回 RAM -> VAE 归位。
                # {prompt, image_b64s? 1–9, mode?, width?, height?, ref_image_size?,
                #  save_cond?: bool}
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                snap = queue.snapshot(limit=1)
                counts = snap.get("counts") or {}
                if counts.get("running") or counts.get("queued"):
                    self._json({"ok": False,
                                "error": "队列非空（生成任务在跑或在排队）。TE 交换要独占显存，等它结束再试。"},
                               409)
                    return
                try:
                    import base64 as _b64
                    import os as _os
                    prompt = payload.get("prompt", "")
                    raw = payload.get("image_b64s") or []
                    imgs = []
                    for s in raw:
                        if not isinstance(s, str):
                            continue
                        if s.startswith("data:"):
                            s = s.split(",", 1)[1]
                        imgs.append(_b64.b64decode(s))
                    mode = payload.get("mode", "ref2va")
                    width = int(payload.get("width", 640))
                    height = int(payload.get("height", 960))
                    ref_size = payload.get("ref_image_size", "match")
                    out, stats = runtime.encode_condition(
                        prompt, imgs, mode=mode, width=width, height=height,
                        ref_image_size=ref_size)
                    resp = {"ok": True, "stats": stats}
                    if payload.get("save_cond"):
                        from safetensors.torch import save_file
                        name = "cond_probe_%d.safetensors" % int(time.time())
                        out_dir = getattr(runtime.args, "out_dir", "outputs")
                        _os.makedirs(out_dir, exist_ok=True)
                        fp = _os.path.join(out_dir, name)
                        tosave = {"cond": out["cond"]}
                        if out["minimax_token_tags"] is not None:
                            tosave["minimax_token_tags"] = out["minimax_token_tags"]
                        save_file(tosave, fp)
                        resp["cond_file"] = name
                        resp["cond_bytes"] = _os.path.getsize(fp)
                    self._json(resp, 200)
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
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
                    from dit_svc.veda_attn_star7 import WARMUP_TIERS_VEDA as WARMUP_TIERS
                    tokens = payload.get("tokens")
                    if tokens is None and payload.get("level"):
                        tier = WARMUP_TIERS.get(payload.get("level"))
                        if tier is None:
                            raise ValueError("level 须为 probe|tier1|tier2|tier3")
                        # 路线 B 起 tiers 的值就是 token 数（int）；旧格式是
                        # (tokens, desc) tuple，这里两种都兼容。
                        tokens = tier[0] if isinstance(tier, (tuple, list)) else tier
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
