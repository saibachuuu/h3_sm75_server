"""串行任务队列（单 GPU，一次只跑一个；手动逐个提交）。

这里是**唯一执行器**：旧 Job（视频/VAE 探针/导演台/warmup）与新 Operation
（权重、attention、单步推理等长操作）共用同一个 worker、同一把 admission
Condition。旧 Job 的 public 字段与提交接口保持不变；新 Operation 见
``dit_svc/operations.py`` 与 ``docs/operation_executor.md``。

admission 规则（与规划 02 §8.2 一致）：

* 空闲：旧 job 接受；control operation 接受并占用唯一保留位；
* 有旧 job 运行/排队：旧 job 继续排队；control operation 一律 409；
* 有 control operation 运行/排队：旧 job 409；新 control 409（最多一个）。

所有检查都在 ``self._cond`` 内完成；preflight/文件扫描在锁外完成，callback
（可能跑 GPU）绝不持锁。
"""
import random
import threading
import time
import traceback
import uuid

from dit_svc.operations import (ACTIVE_STATUSES, Operation, OperationError,
                                TERMINAL_STATUSES, detached_error,
                                json_safe_copy, payload_fingerprint)


class BusyError(OperationError):
    """服务未就绪 / 执行器被占用。

    继承 ``OperationError`` 以便新 HTTP 层统一映射；旧 handler 仍按
    ``except BusyError`` 处理，``str(e)`` 也仍是原来的中文消息。
    """

    def __init__(self, message, code="BUSY", http_status=409, retryable=True,
                 details=None):
        super().__init__(code, message, http_status, retryable, details)


class Job:
    def __init__(self, params, kind="video"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.id = "%s-%s" % (ts, uuid.uuid4().hex[:4])
        self.kind = kind  # video | warmup | vae_probe | director
        self.params = params
        self.status = "queued"      # queued | running | done | failed | cancelled
        self.stage = "排队中"
        self.created = time.time()
        self.started = None
        self.finished = None
        self.error = None
        self.log = []
        self.timings = {}
        self.cond_stats = None
        self.cond_source = None      # 直接喂 cond（/api/jobs/cond）时：{"kind","bytes"/"path","size"}
        self.latent_probe = None
        self.mp4 = None
        self.files = {}              # 额外产物（如 vae_probe 的 PNG）：名字 -> /outputs/<file>
        self.shots = None            # 导演台序列（kind="director"）：镜头列表
        self.concat_mp4 = None       # 导演台序列的总片文件名（拼接成功时才有）

    def _label(self):
        """任务列表里显示的一行标签（四种 kind 各有来源）。"""
        if self.kind == "warmup":
            return "warmup n=%s" % self.params.get("tokens")
        return (self.params.get("prompt") or self.params.get("cond_label")
                or self.params.get("label") or "")[:120]

    def public(self, with_log=False):
        d = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - (self.created or time.time()), 1),
            "prompt": self._label(),
            "seed": self.params.get("seed"),
            "mode": self.params.get("mode"),
            "length": self.params.get("length"),
            "steps": self.params.get("steps"),
            "mp4": self.mp4,
            "output_url": ("/outputs/" + self.mp4) if self.mp4 else None,
            "files": self.files or None,
            "error": self.error,
            "timings": self.timings,
            "cond_stats": self.cond_stats,
            "latent_probe": self.latent_probe,
            "log_tail": self.log[-40:],
        }
        if self.shots is not None:      # 导演台序列：逐镜头进度给前端
            d["shots"] = self.shots
            d["concat_mp4"] = self.concat_mp4
            d["concat_url"] = ("/outputs/" + self.concat_mp4) if self.concat_mp4 else None
        if not with_log:
            d.pop("log_tail")
        return d


class QueueManager:
    def __init__(self, runtime, ctx, operation_retention=200):
        self.runtime = runtime
        self.ctx = ctx
        self.jobs = {}          # id -> Job（旧 public view）
        self.order = []         # 保持提交顺序
        # 新 operation view（与 jobs 共用同一 worker / 同一把锁）
        self.operations = {}    # id -> Operation
        self._op_order = []
        self._request_index = {}    # request_id -> operation id（有界去重）
        self._operation_retention = max(1, int(operation_retention))
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="h3-worker", daemon=True)
        self._worker.start()

    # ---------- 提交 ----------
    def _require_ready(self):
        """旧 job 的 readiness：优先 env_ready，未接入新 runtime 时回退 ready。

        新 runtime 的 ``env_ready`` 表示环境/依赖可用，不表示 DiT 权重已齐；
        因此 DiT 未加载时仍可提交独立的 vae_probe / 其他只依赖环境的旧 job。
        """
        if self.runtime.error:
            raise BusyError("模型加载失败: %s" % self.runtime.error)
        env_ready = getattr(self.runtime, "env_ready", None)
        ready = self.runtime.ready if env_ready is None else env_ready
        if not ready:
            if env_ready is None:
                raise BusyError("模型加载中（%s，已 %.0f 秒），请稍后再提交"
                                % (self.runtime.stage,
                                   self.runtime.stage_elapsed()))
            raise BusyError("环境未就绪（%s，已 %.0f 秒），请稍后再提交"
                            % (self.runtime.stage,
                               self.runtime.stage_elapsed()))

    # ---------- 旧 job 的 admission（与 operation 共用同一把锁） ----------
    def _require_job_submit(self, payload):
        """主代理可选接入的旧 job require hook（锁外调用，允许慢操作）。

        runtime 若提供 ``require_job(payload)``，在旧 job 校验后、入队前调用。
        未接入 weights 模块时不存在该属性，行为与旧版完全一致。
        hook 抛 ``OperationError`` 时转换为旧 handler 认识的 BusyError/ValueError，
        避免旧 HTTP 端点把 409/400 误报成 500。
        """
        hook = getattr(self.runtime, "require_job", None)
        if not callable(hook):
            return
        try:
            hook(payload)
        except BusyError:
            raise
        except OperationError as e:
            if e.http_status == 400:
                raise ValueError(str(e)) from e
            raise BusyError(str(e), code=e.code, http_status=e.http_status,
                            retryable=e.retryable, details=e.details) from e

    def _job_hook_error(self, e):
        """把 job admission/execution hook 的 OperationError 转成旧 job 语义。"""
        if isinstance(e, BusyError):
            return e
        if e.http_status == 400:
            return ValueError(str(e))
        return BusyError(str(e), code=e.code, http_status=e.http_status,
                         retryable=e.retryable, details=e.details)

    def _enqueue_job(self, job):
        """原子检查 control 保留位 + 廉价 job_admission 后入队。

        ``runtime.job_admission(job)`` 在 admission 锁内调用，必须廉价且不得
        回调 QueueManager；它用于在锁外 require_job preflight 之后、入队之前
        再核对 weights/recipe 状态，避免 control 在这段时间改变状态后误接。
        """
        with self._cond:
            if self._stopping.is_set():
                raise BusyError("执行器正在关闭，不再接受新任务",
                                code="SHUTTING_DOWN", http_status=503,
                                retryable=True)
            active = self._active_control_locked()
            if active is not None:
                raise BusyError(
                    "有 control 操作在运行或排队（%s %s），生成任务需等它结束后再提交"
                    % (active.kind, active.id),
                    details={"operation_id": active.id})
            hook = getattr(self.runtime, "job_admission", None)
            if callable(hook):
                try:
                    hook(job)
                except BusyError:
                    raise
                except OperationError as e:
                    raise self._job_hook_error(e) from e
                except ValueError:
                    raise
                except Exception as e:
                    raise BusyError(
                        "job_admission 检查失败: %s: %s"
                        % (type(e).__name__, e), code="ADMISSION_FAILED",
                        http_status=503, retryable=True) from e
            self.jobs[job.id] = job
            self.order.append(job.id)
            self._cond.notify_all()

    def submit(self, payload):
        from common.protocol import validate_job

        self._require_ready()
        params, err = validate_job(payload)
        if err:
            raise ValueError(err)
        if params["seed"] is None or params["seed"] < 0:
            params["seed"] = random.randint(0, 2 ** 63 - 1)
        job = Job(params)
        self._require_job_submit(payload)
        self._enqueue_job(job)
        print("[queue] submitted %s seed=%d" % (job.id, params["seed"]), flush=True)
        return job

    # ---------- 直接喂 cond 提交（不回调 cond server；参考图可选） ----------
    def submit_cond(self, payload, cond_base_dir=None):
        from common.protocol import validate_cond_job

        self._require_ready()
        spec, err = validate_cond_job(
            payload, cond_base_dir=cond_base_dir or self.ctx.get("out_dir"))
        if err:
            raise ValueError(err)
        params, cond = spec["params"], spec["cond"]
        if params["seed"] is None or params["seed"] < 0:
            params["seed"] = random.randint(0, 2 ** 63 - 1)
        job = Job(params)
        job.cond_source = cond
        self._require_job_submit(payload)
        self._enqueue_job(job)
        print("[queue] submitted %s (直接喂 cond: %s, %d 图, %d bytes) seed=%d"
              % (job.id, params["cond_label"], len(params["image_b64s"]),
                 cond["size"], params["seed"]), flush=True)
        return job

    # ---------- 导演台：镜头序列（后续镜头参考上一镜头的尾帧） ----------
    def submit_director(self, payload):
        from common.protocol import validate_director

        self._require_ready()
        spec, err = validate_director(payload)
        if err:
            raise ValueError(err)
        gp, shots = spec["global"], spec["shots"]

        # 每镜头的完整参数在执行时才组装（director.run_director）；这里只把
        # 序列级参数与镜头清单挂到任务上，public() 直接吐给前端。
        for s in shots:
            s["status"] = "pending"
            s["stage"] = ""
            s["elapsed"] = 0.0
            s["mp4"] = None
            s["output_url"] = None
            s["tail_urls"] = []
            s["used_prev_frames"] = 0
            s["own_ref_count"] = len(s["image_b64s"])
            s["error"] = None

        params = {
            "prompt": shots[0]["prompt"],       # 列表里的标签用首镜头提示词
            "global": gp,
            "mode": gp.get("mode"),
            "length": shots[0]["length"],
            "steps": gp.get("steps"),
            "seed": gp.get("seed"),
        }
        job = Job(params, kind="director")
        job.shots = shots
        job.stage = "序列排队中"
        self._require_job_submit(payload)
        self._enqueue_job(job)
        print("[queue] submitted %s (director: %d 镜头, mode=%s, 引用上段帧数=%s)"
              % (job.id, len(shots), gp.get("mode"), gp.get("ref_prev_frames")), flush=True)
        return job

    # ---------- VAE 编解码探针（不进 DiT/采样；把编解码段从采样里剥出来单测） ----------
    def submit_vae_probe(self, payload):
        from common.protocol import validate_vae_probe

        self._require_ready()
        params, err = validate_vae_probe(payload)
        if err:
            raise ValueError(err)
        if params["seed"] is None or params["seed"] < 0:
            params["seed"] = random.randint(0, 2 ** 63 - 1)
        job = Job(params, kind="vae_probe")
        job.stage = "vae 探针排队中"
        self._require_job_submit(payload)
        self._enqueue_job(job)
        print("[queue] submitted %s (vae_probe: encode=%s decode=%s latent=%s)"
              % (job.id, "image_b64" in params, params.get("decode"), params.get("latent")),
              flush=True)
        return job

    # ---------- warmup 提交（与视频任务同队列串行，互不抢显存） ----------
    def submit_warmup(self, tokens, repeat=1, compare=None, profile=False):
        self._require_ready()
        if self.runtime.attn_override is None:
            raise ValueError("当前 attention（%s）无 override 可测，warmup 仅支持 sage/w8a8/veda"
                             % self.runtime.attn)
        if not isinstance(tokens, int) or not (16 <= tokens <= 40000):
            raise ValueError("tokens 须为 16–40000 的整数")
        if not isinstance(repeat, int) or not (1 <= repeat <= 50):
            raise ValueError("repeat 须为 1–50 的整数")
        job = Job({"tokens": tokens, "repeat": repeat,
                   "compare": compare, "profile": bool(profile)}, kind="warmup")
        job.stage = "warmup 排队中"
        self._require_job_submit({"kind": "warmup", "tokens": tokens,
                                  "repeat": repeat, "compare": compare,
                                  "profile": bool(profile)})
        self._enqueue_job(job)
        print("[queue] submitted warmup %s tokens=%d x%d compare=%s profile=%s"
              % (job.id, tokens, repeat, compare, bool(profile)), flush=True)
        return job

    # ---------- 新 operation：提交 / 查询 / 取消（统一回执） ----------
    def _active_control_locked(self):
        for oid in self._op_order:
            op = self.operations.get(oid)
            if op is not None and op.status in ACTIVE_STATUSES:
                return op
        return None

    def _active_job_locked(self):
        for jid in self.order:
            job = self.jobs.get(jid)
            if job is not None and job.status in ("queued", "running"):
                return job
        return None

    def _request_operation_locked(self, request_id, fingerprint):
        """锁内查 request_id：命中同请求返回 Operation，冲突抛 409，未命中返回 None。"""
        if request_id is None:
            return None
        existing = self.operations.get(self._request_index.get(request_id))
        if existing is None:
            self._request_index.pop(request_id, None)
            return None
        if existing.fingerprint != fingerprint:
            raise OperationError(
                "REQUEST_ID_CONFLICT",
                "相同 request_id 对应不同请求", 409, False,
                {"request_id": request_id,
                 "operation_id": existing.id,
                 "existing_kind": existing.kind})
        return existing

    def _operation_preflight(self, kind, payload, preflight=None):
        """锁外执行 preflight；显式 callback 优先，否则用 runtime 可选 hook。"""
        fn = preflight
        if not callable(fn):
            fn = getattr(self.runtime, "operation_preflight", None)
        if not callable(fn):
            return None
        try:
            return fn(kind, payload)
        except OperationError:
            raise
        except ValueError as e:
            raise OperationError("INVALID_REQUEST", str(e), 400,
                                 details={"phase": "preflight"}) from e
        except Exception as e:
            raise OperationError(
                "EXECUTION_FAILED", "%s: %s" % (type(e).__name__, e), 500,
                False, {"phase": "preflight",
                        "exception": type(e).__name__}) from e

    def _operation_admission_locked(self, kind, payload):
        """runtime 可选 hook：在 admission 锁内做**廉价**状态检查。

        该 hook 不得做 GPU/文件扫描，也不得回调 QueueManager，否则会死锁。
        """
        hook = getattr(self.runtime, "operation_admission", None)
        if not callable(hook):
            return
        try:
            hook(kind, payload)
        except OperationError:
            raise
        except ValueError as e:
            raise OperationError("INVALID_REQUEST", str(e), 400,
                                 details={"phase": "admission"}) from e
        except Exception as e:
            raise OperationError(
                "EXECUTION_FAILED", "%s: %s" % (type(e).__name__, e), 500,
                False, {"phase": "admission",
                        "exception": type(e).__name__}) from e

    def _operation_revalidate(self, op, preflight_token):
        """执行前重验：runtime 可选 hook，锁外、worker 内、callback 之前。"""
        hook = getattr(self.runtime, "operation_revalidate", None)
        if not callable(hook):
            return
        try:
            hook(op.kind, op.payload, preflight_token)
        except OperationError:
            raise
        except ValueError as e:
            raise OperationError("INVALID_REQUEST", str(e), 400,
                                 details={"phase": "revalidate"}) from e
        except Exception as e:
            raise OperationError(
                "EXECUTION_FAILED", "%s: %s" % (type(e).__name__, e), 500,
                False, {"phase": "revalidate",
                        "exception": type(e).__name__}) from e

    def submit_operation(self, kind, payload, run_callback, preflight=None,
                         request_id=None):
        """提交一个 control operation，返回 ``Operation`` 回执。

        * preflight 在锁外执行（可慢），admission/去重在锁内原子完成；
        * 同一 request_id + 相同 kind/payload 返回**同一个** Operation；
          不同请求返回 409 REQUEST_ID_CONFLICT；
        * 有旧 job 或另一个 control 在运行/排队时返回 409 BUSY；
        * run_callback(operation) 在唯一 worker 中执行，结果/异常写入回执。
        """
        if not isinstance(kind, str) or not kind.strip():
            raise OperationError("INVALID_REQUEST", "kind 必须为非空字符串", 400)
        if not callable(run_callback):
            raise OperationError("INVALID_REQUEST",
                                 "run_callback 必须可调用", 400)
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                raise OperationError("INVALID_REQUEST",
                                     "request_id 必须为非空字符串", 400)

        fingerprint = payload_fingerprint(kind, payload)

        # 先去重再 preflight：重复请求直接返回原回执，不重复做耗时检查。
        with self._cond:
            if self._stopping.is_set():
                raise OperationError("SHUTTING_DOWN", "执行器正在关闭，不再接受新操作",
                                     503, True)
            existing = self._request_operation_locked(request_id, fingerprint)
        if existing is not None:
            return existing

        token = self._operation_preflight(kind, payload, preflight)

        with self._cond:
            if self._stopping.is_set():
                raise OperationError("SHUTTING_DOWN", "执行器正在关闭，不再接受新操作",
                                     503, True)
            # preflight 在锁外期间可能有人先提交了同一 request_id
            existing = self._request_operation_locked(request_id, fingerprint)
            if existing is not None:
                return existing

            active = self._active_control_locked()
            if active is not None:
                raise OperationError(
                    "BUSY",
                    "已有 control 操作在运行或排队（%s %s）"
                    % (active.kind, active.id),
                    409, True, {"operation_id": active.id})
            job = self._active_job_locked()
            if job is not None:
                raise OperationError(
                    "BUSY",
                    "有生成任务在运行或排队（%s %s），control 操作需空闲执行器"
                    % (job.kind, job.id),
                    409, True, {"job_id": job.id})

            self._operation_admission_locked(kind, payload)
            op = Operation(kind, payload, run_callback,
                           preflight=preflight, request_id=request_id,
                           fingerprint=fingerprint,
                           preflight_token=token)
            self.operations[op.id] = op
            self._op_order.append(op.id)
            if request_id is not None:
                self._request_index[request_id] = op.id
            self._prune_operations_locked()
            self._cond.notify_all()
        print("[queue] submitted operation %s (%s)%s"
              % (op.id, kind, " request_id=%s" % request_id if request_id else ""),
              flush=True)
        return op

    def get_operation(self, operation_id):
        return self.operations.get(operation_id)

    def cancel_operation(self, operation_id):
        """取消 queued operation；running/已结束返回 409。

        返回 public dict；不存在抛 OPERATION_NOT_FOUND(404)。
        """
        with self._cond:
            op = self.operations.get(operation_id)
            if op is None:
                raise OperationError("OPERATION_NOT_FOUND", "operation 不存在",
                                     404)
            if op.status == "queued":
                op.status = "cancelled"
                op.stage = "已取消"
                op.finished = time.time()
                op.cancel_reason = "client"
                op.release_inputs()
                self._cond.notify_all()
                return op.public()
            if op.status == "cancelled":
                return op.public()
            raise OperationError(
                "OPERATION_NOT_CANCELLABLE",
                "只有排队中的 operation 可以取消（当前: %s）" % op.status,
                409, False, {"operation_id": op.id, "status": op.status})

    def operations_snapshot(self, limit=20):
        """非阻塞快照：只读内存 metadata，不等待运行中的 callback。"""
        with self._cond:
            ids = list(self._op_order)[-limit:][::-1] if limit else []
            ops = [self.operations[i].public() for i in ids
                   if i in self.operations]
            counts = {"queued": 0, "running": 0, "done": 0,
                      "failed": 0, "cancelled": 0}
            running_id = None
            for op in self.operations.values():
                counts[op.status] = counts.get(op.status, 0) + 1
                if op.status == "running":
                    running_id = op.id
            return {"operations": ops, "counts": counts,
                    "running": running_id}

    def executor_state(self):
        """status 端点用的廉价执行器摘要（不复用 job/operation 原始 payload）。"""
        with self._cond:
            active_op = None
            queued_ops = 0
            for oid in self._op_order:
                op = self.operations.get(oid)
                if op is None:
                    continue
                if op.status == "running":
                    active_op = op.id
                elif op.status == "queued":
                    queued_ops += 1
            active_job = None
            queued_jobs = 0
            for jid in self.order:
                job = self.jobs.get(jid)
                if job is None:
                    continue
                if job.status == "running":
                    active_job = job.id
                elif job.status == "queued":
                    queued_jobs += 1
            if active_op or active_job:
                state = "running"
            elif queued_ops or queued_jobs:
                state = "queued"
            else:
                state = "idle"
            return {"state": state,
                    "active_operation": active_op,
                    "active_job": active_job,
                    "queued": queued_ops + queued_jobs,
                    "queued_operations": queued_ops,
                    "queued_jobs": queued_jobs,
                    "control_reserved": active_op is not None or queued_ops > 0}

    def _prune_operations_locked(self):
        """有界保留：只淘汰最老的 terminal operation，不丢 queued/running。"""
        excess = len(self._op_order) - self._operation_retention
        if excess <= 0:
            return
        removed = 0
        for oid in list(self._op_order):
            if removed >= excess:
                break
            op = self.operations.get(oid)
            if op is None:
                self._op_order.remove(oid)
                continue
            if op.status not in TERMINAL_STATUSES:
                continue
            op.release_inputs()
            self.operations.pop(oid, None)
            self._op_order.remove(oid)
            if op.request_id is not None and \
                    self._request_index.get(op.request_id) == oid:
                self._request_index.pop(op.request_id, None)
            removed += 1

    def drop(self, job_id):
        with self._cond:
            job = self.jobs.get(job_id)
            if job is None:
                return None, "任务不存在"
            if job.status != "queued":
                return None, "只有排队中的任务可以取消（当前: %s）" % job.status
            job.status = "cancelled"
            job.stage = "已取消"
            job.finished = time.time()
            return job.public(), None

    def clear_queued(self):
        n = 0
        with self._cond:
            for jid in self.order:
                job = self.jobs[jid]
                if job.status == "queued":
                    job.status = "cancelled"
                    job.stage = "已取消"
                    job.finished = time.time()
                    n += 1
        return n

    # ---------- 查询 ----------
    def get(self, job_id):
        return self.jobs.get(job_id)

    def snapshot(self, limit=50):
        with self._cond:
            ids = list(self.order)[-limit:][::-1]  # 最新在前
            jobs = [self.jobs[i].public() for i in ids]
            counts = {"queued": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
            running_id = None
            for j in self.jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
                if j.status == "running":
                    running_id = j.id
            return {"jobs": jobs, "counts": counts, "running": running_id}

    # ---------- worker（唯一执行器：operation 优先，其次旧 job） ----------
    def _next_queued_operation_locked(self):
        for oid in self._op_order:
            op = self.operations.get(oid)
            if op is not None and op.status == "queued":
                return op
        return None

    def _next_queued_job_locked(self):
        for jid in self.order:
            job = self.jobs.get(jid)
            if job is not None and job.status == "queued":
                return job
        return None

    def _loop(self):
        while True:
            with self._cond:
                if self._stopping.is_set():
                    return
                op = self._next_queued_operation_locked()
                job = None
                if op is not None:
                    op.status = "running"
                    op.started = time.time()
                    op.stage = "执行中"
                else:
                    job = self._next_queued_job_locked()
                if op is None and job is None:
                    self._cond.wait(timeout=5.0)
                    continue
                if job is not None:
                    job.status = "running"
                    job.started = time.time()
            if op is not None:
                self._run_operation(op)
            else:
                self._run_job(job)

    def _run_job(self, job):
        """旧 job 的执行分发；保持原有异常字段/日志/完成语义。"""
        print("[queue] running %s (%s)" % (job.id, job.kind), flush=True)
        terminal_status, terminal_stage = "done", "完成"
        try:
            # 执行前重验：提交后到真正开跑之间，weights/env 可能已变化。
            hook = getattr(self.runtime, "require_job_execution", None)
            if callable(hook):
                hook(job)
            if job.kind == "warmup":
                self._run_warmup(job)
            elif job.kind == "vae_probe":
                from dit_svc import vae_probe
                vae_probe.run_vae_probe(job, self.ctx)
            elif job.kind == "director":
                from dit_svc import director
                director.run_director(job, self.ctx)
            else:
                from dit_svc import pipeline
                pipeline.run_job(job, self.ctx)
        except Exception as e:
            terminal_status, terminal_stage = "failed", "失败"
            if isinstance(e, OperationError):
                job.error = "%s: %s" % (e.code, e.message)
            else:
                job.error = "%s: %s" % (type(e).__name__, e)
            job.log.append("ERROR: %s" % job.error)
            job.log.append(traceback.format_exc()[-2000:])
            traceback.print_exc()
            print("[queue] job %s FAILED: %s" % (job.id, job.error), flush=True)
        finally:
            # Legacy jobs use the same GPU lifetime boundary as new operations.
            # Keep their reservation running until asynchronous work is drained.
            for name in ("job_cleanup", "job_synchronize"):
                hook = getattr(self.runtime, name, None)
                if callable(hook):
                    try:
                        hook(job)
                    except Exception as exc:
                        terminal_status, terminal_stage = "failed", "失败"
                        if not job.error:
                            job.error = "%s: %s" % (type(exc).__name__, exc)
                        job.log.append("ERROR: %s: %s" % (name, exc))
            with self._cond:
                job.status = terminal_status
                job.stage = terminal_stage
                job.finished = time.time()
                self._cond.notify_all()

    def _operation_cleanup(self, op):
        """callback 后、terminal 前的可选 cleanup/synchronize hooks。

        两个 hook 都会尝试；同步完成前 operation 保持 running/control 保留位，
        队列不会提前宣称 idle。返回第一个错误（已脱栈）。
        """
        first = None
        for name in ("operation_cleanup", "operation_synchronize"):
            hook = getattr(self.runtime, name, None)
            if not callable(hook):
                continue
            try:
                hook(op)
            except Exception as e:
                err = detached_error(e)
                op.log.append("ERROR: %s: %s: %s"
                              % (name, err.code, err.message))
                if first is None:
                    first = err
                traceback.print_exc()
        return first

    def _finalize_operation(self, op, err, result):
        """校验 JSON、cleanup/synchronize、脱栈、terminal、释放输入引用。"""
        clean_result = None
        if err is None:
            try:
                clean_result = json_safe_copy(result, "operation result")
            except ValueError as e:
                err = OperationError("EXECUTION_FAILED", str(e), 500, False,
                                     {"phase": "result"})

        # callback 可能绕过 set_stage 直接写 progress；无论成功失败都清理。
        try:
            op.progress = json_safe_copy(op.progress, "operation progress")
        except ValueError as e:
            op.progress = None
            if err is None:
                err = OperationError("EXECUTION_FAILED", str(e), 500, False,
                                     {"phase": "progress"})

        cleanup_err = self._operation_cleanup(op)
        if cleanup_err is not None:
            if err is None:
                err = cleanup_err
            else:
                op.log.append("cleanup error (保留首个错误): %s: %s"
                              % (cleanup_err.code, cleanup_err.message))

        if err is not None:
            err = detached_error(err)
            op.result = None
        else:
            op.result = clean_result

        with self._cond:
            if err is None:
                op.status = "done"
                op.stage = "完成"
            else:
                op.status = "failed"
                op.stage = "失败"
                op.error = err
            op.finished = time.time()
            op.release_inputs()
            self._prune_operations_locked()
            self._cond.notify_all()
        if err is not None:
            print("[queue] operation %s FAILED: %s: %s"
                  % (op.id, err.code, err.message), flush=True)

    def _run_operation(self, op):
        """锁外执行 operation；preflight 重验 → runtime revalidate → callback。"""
        print("[queue] running operation %s (%s)" % (op.id, op.kind), flush=True)
        try:
            # 提交时只做了一次 admission；执行前在 worker 内重跑 preflight。
            token = self._operation_preflight(op.kind, op.payload, op.preflight)
            self._operation_revalidate(op, token)
            result = op.cb(op)
        except Exception as e:
            raw = e
            op.log.append("ERROR: %s: %s" % (type(e).__name__, e))
            if not isinstance(e, OperationError):
                op.log.append(traceback.format_exc()[-2000:])
                traceback.print_exc()
            self._finalize_operation(op, raw, None)
        else:
            self._finalize_operation(op, None, result)

    def stop(self, timeout=5.0):
        """停止 worker；queued 的 operation/job 标记 cancelled，running 等其自然结束。

        返回 worker 是否已退出。可在测试/服务退出时调用；不应在 callback 内调用
        自己以外的线程 stop 阻塞过久。
        """
        self._stopping.set()
        now = time.time()
        with self._cond:
            for oid in self._op_order:
                op = self.operations.get(oid)
                if op is not None and op.status == "queued":
                    op.status = "cancelled"
                    op.stage = "服务关闭"
                    op.finished = now
                    op.cancel_reason = "shutdown"
                    op.release_inputs()
            for jid in self.order:
                job = self.jobs.get(jid)
                if job is not None and job.status == "queued":
                    job.status = "cancelled"
                    job.stage = "服务关闭"
                    job.finished = now
            self._cond.notify_all()
        worker = self._worker
        if worker is not None and worker.is_alive() \
                and worker is not threading.current_thread():
            worker.join(timeout)
        return not (worker is not None and worker.is_alive())

    def shutdown(self, timeout=5.0):
        return self.stop(timeout=timeout)

    # ---------- warmup 执行（队列内串行；用当前 override + fallback 落点） ----------
    def _run_warmup(self, job):
        from dit_svc import veda_attn
        runtime = self.runtime
        torch = runtime.torch
        tokens = job.params["tokens"]
        repeat = job.params.get("repeat", 1)
        job.stage = "warmup 执行中（n=%d x%d）" % (tokens, repeat)
        # veda 路线 B 下 runtime.attn_override 是 None（handle 是 VedaPatch）；
        # 真正的 override 装在 unet.model_options 里（就地安装，不走 clone）。
        override = runtime.attn_override
        if override is None and runtime.attn == "veda" and runtime.unet is not None:
            override = runtime.unet.model_options.get("transformer_options", {}).get(
                "optimized_attention_override")
        attention = runtime.attention
        fallback = getattr(attention, "optimized_attention", None)
        if override is None or fallback is None:
            raise RuntimeError("无 override/fallback attention 可用（attn=%s）" % runtime.attn)
        # 注意：这里是无 transformer_options 的裸调，gate 必然判 dense ——
        # 所以下面的 ms 数是 **dense 回退** 的耗时，不是稀疏加速比。
        # 稀疏是否真跑只看 veda_calls（真任务累计，见 /api/status）。

        def _call(q, k, v, heads):
            return override(fallback, q, k, v, heads, skip_reshape=True)
        result = veda_attn.run_single_warmup(
            torch, runtime.device, _call, tokens, repeat=repeat,
            log=lambda s: (job.log.append(s), print("[queue %s] %s" % (job.id, s), flush=True)))
        job.timings["warmup"] = result["ms_max"]
        job.cond_stats = {"warmup": result}
        from dit_svc.veda_attn_star7 import status_report
        job.cond_stats["veda_calls"] = status_report(runtime.veda_patch)
        job.log.append("warmup 结果(dense回退口径): %s" % result)
        job.log.append("veda 工况: %s" % job.cond_stats["veda_calls"])

        # 可选：同一对 Q/K/V 上跟稠密 int8 对比（veda 档下 optimized_attention 就是 int8）
        if job.params.get("compare") == "int8":
            job.stage = "attention bench（veda vs int8, n=%d）" % tokens

            def _veda_call(a, b, c):
                return override(fallback, a, b, c, a.shape[1], skip_reshape=True)

            def _dense_call(a, b, c):
                return fallback(a, b, c, a.shape[1], skip_reshape=True)

            bench = veda_attn.bench_attention(
                torch, runtime.device, _veda_call, _dense_call, tokens,
                repeat=max(1, min(3, repeat)), profile=bool(job.params.get("profile")),
                log=lambda s: (job.log.append(s), print("[queue %s] %s" % (job.id, s), flush=True)))
            job.cond_stats["bench"] = bench
            job.log.append("bench 汇总: %s" % bench)
