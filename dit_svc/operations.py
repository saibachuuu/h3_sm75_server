"""统一 operation 回执、public view 与错误分类（仅标准库）。

设计边界（见 docs/operation_executor.md）：

* 旧 ``Job``（视频/VAE 探针/导演台）继续由 ``job_queue.QueueManager`` 执行，
  ``Job.public()`` 字段不变；
* 新长操作（权重、attention、单步推理等）走 ``Operation``：提交即返回回执，
  客户端用 ``GET /api/operations/<id>`` 轮询，不依赖 HTTP 连接是否断开；
* operation 与 job 共用同一个 worker / 同一把 admission Condition，因此
  “检查队列 + 占用保留位”是原子的；
* ``OperationError`` 是新协议的结构化错误；旧 ``BusyError`` 继承它，但
  ``str(BusyError(...))`` 仍返回原消息，旧 HTTP handler 无需改动。

本模块不 import torch、pipeline 或任何 GPU 侧模块。
"""
import hashlib
import json
import time
import uuid

TERMINAL_STATUSES = frozenset(("done", "failed", "cancelled"))
ACTIVE_STATUSES = frozenset(("queued", "running"))


def json_safe_copy(value, what="value"):
    """严格校验并复制成纯 JSON 值；未知类型/NaN/循环引用抛 ValueError。

    用 dumps→loads 同时完成校验和**脱离原对象**（result 里不再持有 Tensor、
    bytes 或 callback 闭包引用的内部结构）。GET 回执不会因为未知类型 500。
    """
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
        text.encode("utf-8")
        return json.loads(text)
    except Exception as e:
        # 包括 TypeError/ValueError/NaN/循环引用/Unicode/RecursionError；
        # 未知校验异常一律归一为 ValueError，worker 不会因此退出。
        raise ValueError("%s 不是纯 JSON: %s" % (what, e)) from e


def json_or_marker(value, what="value"):
    """public view 用：合法就返回副本，非法返回小的 marker，绝不返回 raw 对象。"""
    if value is None:
        return None
    try:
        return json_safe_copy(value, what)
    except Exception as e:
        return {"invalid": type(value).__name__, "reason": str(e)}


class OperationError(Exception):
    """新 API 契约里的 error envelope（可安全转成 dict / JSON）。

    code/message 必填；http_status 是“如果这个错误发生在提交阶段，HTTP 应该
    返回什么”，执行阶段失败时 operation 本身仍返回 200，status=failed。
    """

    def __init__(self, code, message, http_status=500, retryable=False,
                 details=None):
        self.code = str(code or "EXECUTION_FAILED")
        self.message = str(message)
        self.http_status = int(http_status)
        self.retryable = bool(retryable)
        self.details = details
        # Exception 的 args 用 message，保证旧代码 str(e) 仍是可读消息
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "http_status": self.http_status,
            "retryable": self.retryable,
            "details": json_or_marker(self.details, "error details"),
        }

    @classmethod
    def from_exception(cls, exc):
        """把 callback/preflight 抛出的普通异常归类成新协议错误。"""
        if isinstance(exc, cls):
            return exc
        if isinstance(exc, ValueError):
            return cls("INVALID_REQUEST", str(exc), 400, False,
                       {"exception": type(exc).__name__})
        return cls("EXECUTION_FAILED", "%s: %s" % (type(exc).__name__, exc),
                   500, False, {"exception": type(exc).__name__})


def detached_error(err):
    """把异常复制成不携带 traceback/cause/context 的 OperationError 快照。

    直接保存原始异常会把回调栈帧（可能含 Tensor/模型/96MB 输入）挂在
    ``__traceback__`` 上；operation 进入 terminal 后只保留这个脱栈副本。
    """
    if not isinstance(err, OperationError):
        err = OperationError.from_exception(err)
    details = json_or_marker(err.details, "operation error details")
    clean = OperationError(err.code, err.message, err.http_status,
                           err.retryable, details)
    clean.__traceback__ = None
    clean.__cause__ = None
    clean.__context__ = None
    return clean


def _canonical_value(value):
    """把 payload 规范化成稳定、可 JSON 序列化的结构（用于 request_id 去重）。

    只影响指纹，不改变真实 payload。bytes 用 sha256，未知对象用类型 + repr；
    同一进程内相同请求得到相同指纹即可，不做跨版本持久化承诺。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(),
                "__len__": len(value)}
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(k): _canonical_value(v) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(v) for v in value), key=repr)
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def payload_fingerprint(kind, payload):
    """kind + 规范化 payload 的 sha256；相同 request_id 用它判断是否同一请求。"""
    canonical = json.dumps(
        {"kind": kind, "payload": _canonical_value(payload)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def operation_id():
    # 随机 ID，不带秒级时间戳，避免并发/重启后碰撞
    return "op_" + uuid.uuid4().hex


class Operation:
    """一个长操作的内存记录。只有 worker 会写 running 之后的字段。

    ``cb`` / ``payload`` 不进入 public view；GET 只看 ``public()``。
    ``public()`` 本身不加锁，调用方（QueueManager）在 Condition 内复制字段，
    保证不会读到半更新的 eviction 状态。
    """

    def __init__(self, kind, payload, cb, preflight=None, request_id=None,
                 fingerprint=None, preflight_token=None):
        self.id = operation_id()
        self.kind = kind
        self.payload = payload
        self.cb = cb
        self.preflight = preflight
        self.request_id = request_id
        self.fingerprint = fingerprint
        self.preflight_token = preflight_token
        self.status = "queued"       # queued | running | done | failed | cancelled
        self.stage = "排队中"
        self.created = time.time()
        self.started = None
        self.finished = None
        self.progress = None
        self.result = None
        self.error = None            # OperationError | None
        self.cancel_reason = None
        self.log = []

    @property
    def terminal(self):
        return self.status in TERMINAL_STATUSES

    def set_stage(self, stage, progress=None):
        """callback 内更新进度；不需要/不应该获取 admission lock。"""
        self.stage = str(stage)
        if progress is not None:
            try:
                self.progress = json_safe_copy(progress, "operation progress")
            except ValueError as e:
                raise OperationError(
                    "EXECUTION_FAILED", str(e), 500, False,
                    {"phase": "progress"}) from e

    def release_inputs(self):
        """进入 terminal/cancel 后释放 payload、callback、preflight 强引用。

        fingerprint/kind/result/error 摘要保留，供有限窗口内的回执查询。
        """
        self.payload = None
        self.cb = None
        self.preflight = None
        self.preflight_token = None

    def public(self, include_log=False):
        elapsed = (self.finished or time.time()) - (self.created or time.time())
        error = self.error
        if isinstance(error, OperationError):
            error = error.to_dict()
        elif error is not None:
            error = {"code": "EXECUTION_FAILED", "message": str(error),
                     "http_status": 500, "retryable": False, "details": None}
        d = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            # 即使 callback 绕过 set_stage 直接写字段，GET 也绝不吐 raw Tensor
            "progress": json_or_marker(self.progress, "operation progress"),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round(elapsed, 3),
            "request_id": self.request_id,
            "result": json_or_marker(self.result, "operation result"),
            "error": error,
            "operation_url": "/api/operations/" + self.id,
        }
        if include_log:
            d["log_tail"] = [str(x) for x in self.log[-40:]]
        return d

    def receipt(self):
        """POST 直接返回给客户端的 202 回执。"""
        return {
            "ok": True,
            "operation_id": self.id,
            "status": self.status,
            "operation_url": "/api/operations/" + self.id,
        }
