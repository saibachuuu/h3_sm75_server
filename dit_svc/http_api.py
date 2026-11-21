"""dit server 的 HTTP 层：WebUI 静态页 + 任务 API + 成果文件（含 Range 断点）。"""
import json
import os
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.protocol import DEFAULTS, DIRECTOR_MAX_SHOTS, SAMPLERS, SCHEDULERS, dumps
from dit_svc import api_validation
from dit_svc.operations import OperationError
from dit_svc.artifacts import ArtifactStore, safe_join

WEBUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui")
MAX_BODY = 96 * 1024 * 1024


def byte_range(header, size):
    """Return one RFC byte range; invalid/unsatisfiable requests raise ValueError."""
    if not header:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", header.strip())
    if match is None or not any(match.groups()):
        raise ValueError("invalid byte range")
    first, last = match.groups()
    if first:
        start = int(first)
        end = int(last) if last else size - 1
        if start >= size or end < start:
            raise ValueError("unsatisfiable byte range")
        return start, min(end, size - 1), 206
    suffix = int(last)
    if suffix <= 0 or size <= 0:
        raise ValueError("unsatisfiable suffix range")
    return max(0, size - suffix), size - 1, 206

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
    artifacts = ctx.get("artifacts")
    if artifacts is None:
        artifacts = ArtifactStore(out_dir)
        ctx["artifacts"] = artifacts
    runtime.artifacts = artifacts
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

        def _error(self, exc):
            error = OperationError.from_exception(exc)
            self._json({"ok": False, "error": error.to_dict()}, error.http_status)

        def _file(self, path, ctype=None, allow_range=True):
            if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                self._json({"ok": False, "error": "文件不存在"}, 404)
                return
            size = os.path.getsize(path)
            ctype = ctype or "application/octet-stream"
            range_hdr = self.headers.get("Range") if allow_range else None
            try:
                start, end, code = byte_range(range_hdr, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
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

        def _wait_operation(self, operation, timeout):
            """Wait for a terminal state without holding any worker/lease lock.

            Returns the operation, or None on timeout (the caller then reports the
            existing 202 receipt; the operation is not resubmitted).
            """
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if operation.terminal:
                    return operation
                time.sleep(0.05)
            return None

        def _operation_error_response(self, operation):
            error = operation.error
            if isinstance(error, OperationError):
                payload = error.to_dict()
                status = error.http_status
            else:
                payload = {"code": "EXECUTION_FAILED", "message": str(error),
                           "http_status": 500, "retryable": False,
                           "details": None}
                status = 500
            return payload, status

        def _post_weights(self, path):
            """POST /api/weights/<model>/<action> → 202 operation receipt."""
            rest = path[len("/api/weights/"):]
            parts = rest.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                self._json({"ok": False, "error": {
                    "code": "MODEL_NOT_FOUND", "message": "路径应为 /api/weights/<model>/<action>",
                    "http_status": 404, "retryable": False, "details": None}}, 404)
                return
            model_id, action = parts
            payload = self._read_json_body()
            if payload is None:
                self._json({"ok": False, "error": {
                    "code": "INVALID_REQUEST", "message": "请求体必须是 JSON 对象",
                    "http_status": 400, "retryable": False, "details": None}}, 400)
                return
            try:
                if not isinstance(payload, dict):
                    raise OperationError("INVALID_REQUEST", "请求体必须是 JSON 对象",
                                         400, False)
                payload = dict(payload)
                declared = payload.get("model")
                if declared is None:
                    payload["model"] = model_id
                elif declared != model_id:
                    raise OperationError(
                        "INVALID_REQUEST",
                        "URL model 与 body.model 不一致", 400, False,
                        {"url_model": model_id, "body_model": declared})
                request = api_validation.validate_weights_request(payload, action)
                if request["models"] != [model_id]:
                    raise OperationError("INVALID_REQUEST", "model 列表不匹配",
                                         400, False)
                # 提交阶段完成 catalog 建立 + selector 展开（纯 CPU/header，
                # 不构造模型、不搬设备）；未知 unit 直接 4xx。
                runtime.weights.ensure_catalog(model_id)
                if request["mode"] != "all":
                    request["units"] = runtime.weights.unit_ids(model_id, request)
                operation = queue.submit_operation(
                    "weights.%s" % action, request,
                    lambda op: runtime.run_weights_action(
                        model_id, action, request, op),
                    request_id=request["request_id"])
                receipt = operation.receipt()
                receipt["model"] = model_id
                receipt["action"] = action
                self._json(receipt, 202)
            except OperationError as exc:
                self._json({"ok": False, "error": exc.to_dict()}, exc.http_status)
            except Exception as exc:
                error = OperationError.from_exception(exc)
                self._json({"ok": False, "error": error.to_dict()},
                           error.http_status)

        def _post_legacy_te(self, path):
            """Preserve synchronous legacy response, but execute on the single worker."""
            payload = self._read_json_body()
            if not isinstance(payload, dict):
                self._error(OperationError("INVALID_REQUEST", "请求体必须是 JSON 对象",
                                           400, False))
                return
            callback = (runtime.run_te_legacy if path.endswith("/step")
                        else runtime.run_te_probe)
            try:
                operation = queue.submit_operation(
                    "legacy.te.%s" % path.rsplit("/", 1)[-1], payload,
                    lambda op: callback(payload, op))
                finished = self._wait_operation(operation, 1800.0)
                if finished is None:
                    self._json(operation.receipt(), 202)
                elif finished.status == "done":
                    self._json(finished.result, 200)
                else:
                    self._error(finished.error)
            except Exception as exc:
                self._error(exc)

        def _post_encode(self):
            """POST /api/infer/encode → 202；显式要求 te 已在显存。"""
            payload = self._read_json_body()
            if payload is None:
                self._json({"ok": False, "error": {
                    "code": "INVALID_REQUEST", "message": "请求体必须是 JSON 对象",
                    "http_status": 400, "retryable": False, "details": None}}, 400)
                return
            try:
                request = api_validation.validate_encode_request(payload)
                runtime.te.preflight_encode(request)
                operation = queue.submit_operation(
                    "infer.encode", request,
                    lambda op: runtime.run_infer_encode(request, op),
                    request_id=request["request_id"])
                receipt = operation.receipt()
                receipt["capability"] = "encode"
                self._json(receipt, 202)
            except OperationError as exc:
                self._json({"ok": False, "error": exc.to_dict()}, exc.http_status)
            except Exception as exc:
                error = OperationError.from_exception(exc)
                self._json({"ok": False, "error": error.to_dict()},
                           error.http_status)

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
            elif path.startswith("/api/artifacts/"):
                try:
                    artifact_id = path[len("/api/artifacts/"):]
                    self._json({"ok": True, "artifact": artifacts.public(artifact_id)})
                except Exception as exc:
                    self._error(exc)
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
                try:
                    name = path[len("/outputs/"):]
                    ctype = ("video/mp4" if name.endswith(".mp4")
                             else "application/json" if name.endswith(".json")
                             else "image/png" if name.endswith(".png") else None)
                    if name.startswith("stage/"):
                        name = name[len("stage/"):]
                        with artifacts.download_lease(name) as p:
                            self._file(p, ctype)
                    else:
                        self._file(safe_join(out_dir, name), ctype)
                except Exception as exc:
                    self._error(exc)
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        # ---------------- POST ----------------
        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path.startswith("/api/weights/"):
                self._post_weights(path)
                return
            if path == "/api/infer/encode":
                self._post_encode()
                return
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

            elif path in ("/api/te/step", "/api/te/probe"):
                self._post_legacy_te(path)

            elif path == "/api/encode":
                # 兼容旧同步语义：同一个 worker 执行，有界等待；超时返回已有 202 回执。
                # 不再自动 VAE 让位/TE 上卡：必须先用 POST /api/weights/te/*。
                payload = self._read_json_body()
                if payload is None:
                    self._json({"ok": False, "error": "请求体不是合法 JSON 或过大"}, 400)
                    return
                if not isinstance(payload, dict):
                    self._json({"ok": False, "error": "请求体必须是 JSON 对象"}, 400)
                    return
                # save_cond 在旧语义里控制落盘；新 operation 总是提交 artifact。
                payload = {k: v for k, v in payload.items() if k != "save_cond"}
                try:
                    request = api_validation.validate_encode_request(payload)
                    runtime.te.preflight_encode(request)
                    operation = queue.submit_operation(
                        "infer.encode", request,
                        lambda op: runtime.run_infer_encode(request, op),
                        request_id=request["request_id"])
                    finished = self._wait_operation(operation, 1800.0)
                    if finished is None:
                        self._json(operation.receipt(), 202)
                        return
                    if finished.status == "done":
                        result = finished.result or {}
                        artifact = result.get("artifact") or {}
                        filename = artifact.get("filename")
                        self._json({
                            "ok": True,
                            "stats": result.get("stats"),
                            "artifact": artifact,
                            "cond_file": filename,
                            "cond_bytes": artifact.get("bytes"),
                            "cond_url": ("/outputs/stage/%s" % filename
                                         if filename else None),
                            "operation_id": finished.id,
                        }, 200)
                    else:
                        error, status = self._operation_error_response(finished)
                        self._json({"ok": False, "error": error,
                                    "operation_id": finished.id}, status)
                except OperationError as exc:
                    self._json({"ok": False, "error": exc.to_dict()}, exc.http_status)
                except Exception as exc:
                    error = OperationError.from_exception(exc)
                    self._json({"ok": False, "error": error.to_dict()},
                               error.http_status)
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
