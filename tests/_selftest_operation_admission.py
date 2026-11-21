"""自检：统一 operation executor / 原子 admission（仅标准库，不需要 torch）。

覆盖：
  * receipt / 状态 / 错误 envelope；
  * 两个并发 control 最多一个获准；
  * control 与旧 job 的原子互斥；旧 job 运行期间 job 仍可排队；
  * preflight 在锁外、执行前重验；
  * queued cancellation、request_id 去重与冲突、保留上限；
  * callback 异常不封锁队列；
  * GET/operations snapshot 不等待 callback；
  * shutdown 干净退出；
  * runtime 可选 require hooks（主代理尚未接入 weights 模块时默认无 hook）。

运行：python tests/_selftest_operation_admission.py
"""
import gc
import json
import os
import sys
import threading
import time
import traceback
import weakref

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc.job_queue import BusyError, QueueManager  # noqa: E402
from dit_svc.operations import OperationError  # noqa: E402


_ABSENT = object()


class FakeRuntime:
    """只提供 QueueManager 旧 job 路径需要的字段；不 import torch。"""

    def __init__(self, ready=True, env_ready=_ABSENT):
        self.ready = ready
        if env_ready is not _ABSENT:
            self.env_ready = env_ready       # 模拟新 runtime；缺省即 legacy
        self.error = None
        self.stage = "fake"
        self.stage_t0 = time.perf_counter()
        self.attn_override = None
        self.attn = "none"

    def stage_elapsed(self):
        return time.perf_counter() - self.stage_t0


class RefNode:
    """弱引用测试用的小对象（不进入 payload 指纹的强引用链）。"""

    pass


class FakeQueue(QueueManager):
    """可控旧 job 执行时机的测试队列。"""

    def __init__(self, runtime=None, block_jobs=False, operation_retention=200):
        self.block_jobs = block_jobs
        self.job_started = threading.Event()
        self.job_release = threading.Event()
        self.job_ids = []
        super().__init__(runtime or FakeRuntime(), {},
                         operation_retention=operation_retention)

    def _run_job(self, job):
        self.job_ids.append(job.id)
        self.job_started.set()
        if self.block_jobs:
            self.job_release.wait(5.0)
        job.status = "done"
        job.stage = "完成"
        job.finished = time.time()

    def stop(self, timeout=5.0):
        # 测试失败时也不要把 join 卡在阻塞的假 job 上
        self.job_release.set()
        return super().stop(timeout=timeout)


class PausedQueue(QueueManager):
    """worker 不取任务：确定性观察 queued operation / 取消。"""

    def __init__(self, runtime=None):
        super().__init__(runtime or FakeRuntime(), {})

    def _loop(self):
        while not self._stopping.is_set():
            with self._cond:
                self._cond.wait(timeout=0.02)


def job_payload(prompt="test"):
    # validate_job 只检查非空 base64 字符串，不在 admission 阶段解码
    return {"prompt": prompt, "image_b64": "aGVsbG8=", "seed": 1}


def wait_for(fn, desc, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("超时: %s" % desc)


def wait_status(queue, op_id, status, timeout=5.0):
    return wait_for(
        lambda: (queue.get_operation(op_id) is not None
                 and queue.get_operation(op_id).status == status
                 and queue.get_operation(op_id)),
        "operation %s -> %s" % (op_id, status), timeout)


def expect_error(fn, code):
    try:
        fn()
    except OperationError as e:
        assert e.code == code, "期望 %s，实际 %s: %s" % (code, e.code, e)
        assert isinstance(e.to_dict(), dict)
        for key in ("code", "message", "http_status", "retryable", "details"):
            assert key in e.to_dict(), "error envelope 缺字段 %s" % key
        return e
    raise AssertionError("没有抛出 OperationError(%s)" % code)


fails = []


def check(name, fn):
    try:
        detail = fn()
        print("  OK   %-46s %s" % (name, detail))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-46s %s: %s" % (name, type(e).__name__, e))
        traceback.print_exc()
        fails.append(name)


# ---------------------------------------------------------------- 用例

def t_receipt_and_result():
    q = FakeQueue()
    try:
        def cb(op):
            op.set_stage("处理中", progress=0.5)
            return {"completed": ["dit.blocks.13"], "n": 1}

        op = q.submit_operation(
            "weights.disk2ram", {"model": "dit", "mode": "all"}, cb,
            request_id="r-1")
        receipt = op.receipt()
        assert receipt["ok"] is True
        assert receipt["operation_id"] == op.id
        assert receipt["operation_url"] == "/api/operations/" + op.id
        wait_status(q, op.id, "done")
        pub = op.public()
        assert pub["result"]["n"] == 1
        assert pub["error"] is None
        assert pub["kind"] == "weights.disk2ram"
        assert pub["request_id"] == "r-1"
        assert "payload" not in pub and "cb" not in pub
        assert q.operations_snapshot()["counts"]["done"] == 1
        return "id=%s" % op.id
    finally:
        q.stop()


def t_preflight_twice():
    q = FakeQueue()
    events = []
    try:
        def preflight(kind, payload):
            events.append("preflight:%s" % payload.get("v"))
            return {"revision": len(events)}

        def cb(op):
            events.append("callback")
            return "ok"

        op = q.submit_operation("ctrl", {"v": 1}, cb, preflight=preflight)
        wait_status(q, op.id, "done")
        assert events.count("preflight:1") == 2, events
        assert events[-1] == "callback", events
        return "顺序=%s" % events
    finally:
        q.stop()


def t_two_controls_atomic():
    q = PausedQueue()
    barrier = threading.Barrier(2)
    results = []
    lock = threading.Lock()

    def submit(i):
        barrier.wait()
        try:
            op = q.submit_operation("ctrl", {"i": i}, lambda o: "ok")
            with lock:
                results.append(("ok", op.id))
        except OperationError as e:
            with lock:
                results.append(("err", e.code, e.http_status))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert len(results) == 2, results
        oks = [r for r in results if r[0] == "ok"]
        errs = [r for r in results if r[0] == "err"]
        assert len(oks) == 1 and len(errs) == 1, results
        assert errs[0][1:] == ("BUSY", 409), errs
        assert q.operations_snapshot()["counts"]["queued"] == 1
        # 释放保留位后可以再提交
        q.cancel_operation(oks[0][1])
        op2 = q.submit_operation("ctrl", {"i": 2}, lambda o: "ok")
        assert op2.status == "queued"
        return "接受=%s 拒绝=%s" % (oks[0][1], errs[0][1])
    finally:
        q.stop()


def t_control_job_barrier():
    # 跑多轮，两侧都可能先拿到锁；每轮必须恰好一个获准。
    rounds = 5
    wins = {"job": 0, "control": 0}
    for r in range(rounds):
        q = FakeQueue(block_jobs=True)
        release = threading.Event()
        barrier = threading.Barrier(2)
        results = []
        lock = threading.Lock()

        def control_submit():
            barrier.wait()
            try:
                op = q.submit_operation(
                    "ctrl", {"r": r},
                    lambda o: (release.wait(5.0), "ok")[1])
                with lock:
                    results.append(("control", op.id))
            except OperationError as e:
                with lock:
                    results.append(("control-err", e.code, e.http_status))

        def job_submit():
            barrier.wait()
            try:
                job = q.submit(job_payload("r%d" % r))
                with lock:
                    results.append(("job", job.id))
            except BusyError as e:
                with lock:
                    results.append(("job-err", str(e)))

        threads = [threading.Thread(target=control_submit),
                   threading.Thread(target=job_submit)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(5)
            oks = [x for x in results if x[0] in ("control", "job")]
            assert len(oks) == 1, "round %d: %s" % (r, results)
            wins[oks[0][0]] += 1
        finally:
            release.set()
            q.job_release.set()
            q.stop()
    assert wins["job"] + wins["control"] == rounds, wins
    return "job胜=%d control胜=%d" % (wins["job"], wins["control"])


def t_jobs_queue_while_job_running():
    q = FakeQueue(block_jobs=True)
    try:
        j1 = q.submit(job_payload("one"))
        wait_for(lambda: q.job_started.is_set(), "job1 started")
        assert j1.status == "running", j1.status

        j2 = q.submit(job_payload("two"))       # 旧 job 运行中，job 仍可排队
        assert j2.status == "queued", j2.status

        err = expect_error(
            lambda: q.submit_operation("ctrl", {}, lambda o: "ok"), "BUSY")
        assert err.details["job_id"] in (j1.id, j2.id)

        snap = q.snapshot()
        assert snap["counts"]["running"] == 1, snap["counts"]
        assert snap["counts"]["queued"] == 1, snap["counts"]
        assert snap["running"] == j1.id, snap["running"]

        q.job_release.set()
        wait_for(lambda: j2.status == "done", "job2 done")
        snap2 = q.snapshot()
        assert snap2["counts"]["done"] == 2, snap2["counts"]
        return "j1=%s j2=%s" % (j1.id, j2.id)
    finally:
        q.job_release.set()
        q.stop()


def t_control_rejects_job():
    q = PausedQueue()
    try:
        op = q.submit_operation("ctrl", {}, lambda o: "ok")
        assert op.status == "queued"
        try:
            q.submit(job_payload("blocked"))
        except BusyError as e:
            assert "control" in str(e), str(e)
            assert e.http_status == 409
        else:
            raise AssertionError("control 占用时旧 job 没有 409")
        assert q.snapshot()["counts"]["queued"] == 0
        return "op=%s" % op.id
    finally:
        q.stop()


def t_cancel_queued():
    q = PausedQueue()
    try:
        op = q.submit_operation("ctrl", {}, lambda o: "ok")
        pub = q.cancel_operation(op.id)
        assert pub["status"] == "cancelled", pub
        assert pub["finished"] is not None
        assert q.cancel_operation(op.id)["status"] == "cancelled"  # 幂等
        assert q.operations_snapshot()["counts"]["cancelled"] == 1

        expect_error(lambda: q.cancel_operation("op_missing"),
                     "OPERATION_NOT_FOUND")

        # 取消后保留位释放，可以继续提交
        op2 = q.submit_operation("ctrl", {"n": 2}, lambda o: "ok")
        assert op2.status == "queued"
        return "op=%s op2=%s" % (op.id, op2.id)
    finally:
        q.stop()


def t_cancel_running_rejected():
    q = FakeQueue()
    release = threading.Event()
    try:
        op = q.submit_operation(
            "ctrl", {}, lambda o: (release.wait(5.0), "ok")[1])
        wait_status(q, op.id, "running")
        err = expect_error(lambda: q.cancel_operation(op.id),
                           "OPERATION_NOT_CANCELLABLE")
        assert err.http_status == 409
        assert err.details["status"] == "running"
        return "op=%s" % op.id
    finally:
        release.set()
        q.stop()


def t_request_id_dedup():
    q = FakeQueue()
    calls = []
    try:
        def cb(op):
            calls.append(op.id)
            return "ok"

        op1 = q.submit_operation("ctrl", {"a": 1, "b": [2]}, cb,
                                 request_id="same")
        op2 = q.submit_operation("ctrl", {"b": [2], "a": 1}, cb,
                                 request_id="same")  # key 顺序不影响指纹
        assert op1.id == op2.id, (op1.id, op2.id)
        wait_status(q, op1.id, "done")
        assert len(calls) == 1, calls
        assert q.operations_snapshot()["counts"]["done"] == 1

        err = expect_error(
            lambda: q.submit_operation("ctrl", {"a": 2}, cb,
                                       request_id="same"),
            "REQUEST_ID_CONFLICT")
        assert err.http_status == 409
        assert err.details["operation_id"] == op1.id
        return "id=%s calls=%d" % (op1.id, len(calls))
    finally:
        q.stop()


def t_dedup_skips_preflight():
    q = PausedQueue()
    calls = []
    try:
        def preflight(kind, payload):
            calls.append(payload.get("n"))
            return {"rev": 1}

        op1 = q.submit_operation("ctrl", {"n": 1}, lambda o: "ok",
                                 preflight=preflight, request_id="skip")
        op2 = q.submit_operation("ctrl", {"n": 1}, lambda o: "ok",
                                 preflight=preflight, request_id="skip")
        assert op1.id == op2.id
        assert calls == [1], calls   # 重复请求直接回原回执，不重复 preflight
        return "preflight 只调用 %d 次" % len(calls)
    finally:
        q.stop()


def t_request_dedup_bounded():
    q = FakeQueue(operation_retention=1)
    try:
        op1 = q.submit_operation("ctrl", {"n": 1}, lambda o: "ok",
                                 request_id="old")
        wait_status(q, op1.id, "done")
        op2 = q.submit_operation("ctrl", {"n": 2}, lambda o: "ok",
                                 request_id="new")
        wait_status(q, op2.id, "done")
        assert q.get_operation(op1.id) is None, "保留上限没有淘汰旧回执"
        # 旧 operation 已淘汰，同 request_id 可以开始新请求
        op3 = q.submit_operation("ctrl", {"n": 1}, lambda o: "ok",
                                 request_id="old")
        assert op3.id != op1.id
        return "op1淘汰 op3=%s" % op3.id
    finally:
        q.stop()


def t_callback_exception_does_not_block():
    q = FakeQueue()
    try:
        def bad(op):
            raise RuntimeError("boom")

        op1 = q.submit_operation("ctrl", {"n": 1}, bad)
        wait_status(q, op1.id, "failed")
        err = op1.public()["error"]
        assert err["code"] == "EXECUTION_FAILED", err
        assert err["http_status"] == 500, err
        assert err["retryable"] is False, err

        def typed(op):
            raise OperationError("MISSING_UNITS", "缺权重", 409, False,
                                 {"missing": ["dit"]})

        op2 = q.submit_operation("ctrl", {"n": 2}, typed)
        wait_status(q, op2.id, "failed")
        assert op2.public()["error"]["code"] == "MISSING_UNITS"

        op3 = q.submit_operation("ctrl", {"n": 3}, lambda o: "after")
        wait_status(q, op3.id, "done")
        assert op3.public()["result"] == "after"
        return "failed=2 done=1"
    finally:
        q.stop()


def t_preflight_failure_paths():
    q = FakeQueue()
    state = {"n": 0}
    called = []
    try:
        def flaky(kind, payload):
            state["n"] += 1
            if state["n"] > 1:
                raise OperationError("MISSING_UNITS", "执行前缺失", 409, False)
            return {"rev": 1}

        op = q.submit_operation("ctrl", {}, lambda o: called.append(o.id),
                                preflight=flaky)
        wait_status(q, op.id, "failed")
        assert called == [], "preflight 失败后 callback 不应执行"
        assert op.public()["error"]["code"] == "MISSING_UNITS"

        def deny(kind, payload):
            raise OperationError("ENV_NOT_READY", "环境未就绪", 503, True)

        err = expect_error(
            lambda: q.submit_operation("ctrl", {}, lambda o: "ok",
                                       preflight=deny),
            "ENV_NOT_READY")
        assert err.http_status == 503 and err.retryable is True
        # 提交阶段失败不占用保留位
        op2 = q.submit_operation("ctrl", {"ok": 1}, lambda o: "ok")
        wait_status(q, op2.id, "done")
        return "failed=%s then done=%s" % (op.id, op2.id)
    finally:
        q.stop()


def t_runtime_hooks():
    runtime = FakeRuntime()
    events = []

    def preflight(kind, payload):
        events.append("preflight")
        return {"rev": 7}

    def admission(kind, payload):
        events.append("admission")

    def revalidate(kind, payload, token):
        events.append("revalidate:%s" % ((token or {}).get("rev")))

    runtime.operation_preflight = preflight
    runtime.operation_admission = admission
    runtime.operation_revalidate = revalidate
    q = FakeQueue(runtime)
    try:
        op = q.submit_operation(
            "ctrl", {}, lambda o: (events.append("callback"), "ok")[1])
        wait_status(q, op.id, "done")
        assert events == ["preflight", "admission", "preflight",
                          "revalidate:7", "callback"], events
        return "hooks=%s" % events
    finally:
        q.stop()


def t_runtime_admission_denies():
    runtime = FakeRuntime()

    def admission(kind, payload):
        raise OperationError("ENV_NOT_READY", "环境未就绪", 503, True)

    runtime.operation_admission = admission
    q = FakeQueue(runtime)
    try:
        err = expect_error(
            lambda: q.submit_operation("ctrl", {}, lambda o: "ok"),
            "ENV_NOT_READY")
        assert err.http_status == 503
        assert q.operations_snapshot()["counts"]["queued"] == 0
        return "503"
    finally:
        q.stop()


def t_runtime_require_job():
    runtime = FakeRuntime()
    calls = []

    def require_job(payload):
        calls.append(payload.get("prompt"))
        raise OperationError("MISSING_UNITS", "缺权重", 409, False)

    runtime.require_job = require_job
    q = FakeQueue(runtime)
    try:
        try:
            q.submit(job_payload("hooked"))
        except BusyError as e:
            assert str(e) == "缺权重", str(e)
            assert e.code == "MISSING_UNITS"
            assert e.http_status == 409
        else:
            raise AssertionError("require_job 拒绝后 job 仍然提交成功")
        assert calls == ["hooked"], calls
        assert q.snapshot()["counts"]["queued"] == 0
        return "BusyError=%s" % calls[0]
    finally:
        q.stop()


def t_snapshot_nonblocking():
    q = FakeQueue()
    started = threading.Event()
    release = threading.Event()
    try:
        def cb(op):
            started.set()
            release.wait(5.0)
            return "late"

        op = q.submit_operation("ctrl", {"huge": "x" * 100}, cb)
        assert started.wait(2.0)
        t0 = time.perf_counter()
        snap = q.operations_snapshot(limit=20)
        snap2 = q.snapshot(limit=20)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, "snapshot 被 callback 阻塞 %.3fs" % elapsed
        assert snap["running"] == op.id, snap
        assert snap["counts"]["running"] == 1, snap["counts"]
        assert snap2["counts"]["running"] == 0, snap2["counts"]
        assert "payload" not in snap["operations"][0]

        state = q.executor_state()
        assert state["state"] == "running" and state["control_reserved"]
        release.set()
        wait_status(q, op.id, "done")
        assert q.executor_state()["state"] == "idle"
        return "snapshot %.4fs" % elapsed
    finally:
        release.set()
        q.stop()


def t_client_abandon_keeps_server_working():
    q = FakeQueue()
    try:
        release = threading.Event()
        op = q.submit_operation(
            "ctrl", {}, lambda o: (release.wait(5.0), {"ok": 1})[1])
        # 模拟客户端提交后断线：不再查询/不等待 HTTP 响应
        release.set()
        wait_status(q, op.id, "done")
        assert q.operations_snapshot()["operations"][0]["result"] == {"ok": 1}
        return "server 继续完成 %s" % op.id
    finally:
        q.stop()


def t_shutdown_clean():
    q = FakeQueue()
    assert q.stop(timeout=2.0) is True
    assert not q._worker.is_alive()
    err = expect_error(
        lambda: q.submit_operation("ctrl", {}, lambda o: "ok"),
        "SHUTTING_DOWN")
    assert err.http_status == 503
    try:
        q.submit(job_payload("after-stop"))
    except BusyError as e:
        assert e.code == "SHUTTING_DOWN"
    else:
        raise AssertionError("关闭后旧 job 仍然提交成功")

    q2 = PausedQueue()
    op = q2.submit_operation("ctrl", {}, lambda o: "ok")
    assert op.status == "queued"
    assert q2.stop(timeout=2.0) is True
    assert not q2._worker.is_alive()
    assert op.status == "cancelled"
    snap = q2.operations_snapshot()
    assert snap["counts"]["cancelled"] == 1, snap["counts"]

    q3 = PausedQueue()
    job = q3.submit(job_payload("queued"))
    assert job.status == "queued"
    assert q3.stop(timeout=2.0) is True
    assert not q3._worker.is_alive()
    assert job.status == "cancelled"
    return "worker 已退出，queued op/job 已取消"


def t_readiness_env_ready():
    # legacy：没有 env_ready，仍看 ready
    q = FakeQueue(FakeRuntime(ready=False))
    try:
        try:
            q.submit(job_payload("legacy-not-ready"))
        except BusyError as e:
            assert "模型加载中" in str(e), str(e)
        else:
            raise AssertionError("legacy ready=false 仍接受 job")
        op = q.submit_operation("ctrl", {}, lambda o: "ok")
        wait_status(q, op.id, "done")      # control 不要求模型 ready
    finally:
        q.stop()

    # 新 runtime：env_ready=true 但 DiT 未齐，独立 vae_probe/旧 job 可提交
    q2 = FakeQueue(FakeRuntime(ready=False, env_ready=True))
    try:
        job = q2.submit(job_payload("env-ready"))
        wait_for(lambda: job.status == "done", "env_ready job done")
    finally:
        q2.stop()

    # env_ready=false 即使 legacy ready=true 也应拒绝
    q3 = FakeQueue(FakeRuntime(ready=True, env_ready=False))
    try:
        try:
            q3.submit(job_payload("env-not-ready"))
        except BusyError as e:
            assert "环境未就绪" in str(e), str(e)
        else:
            raise AssertionError("env_ready=false 仍接受 job")
    finally:
        q3.stop()
    return "legacy/env_ready 分支正确"


def t_job_admission_lock():
    runtime = FakeRuntime()
    q = FakeQueue(runtime)
    seen = {}

    def admission(job):
        seen["owned"] = q._cond._is_owned()
        seen["status"] = job.status
        raise OperationError("MISSING_UNITS", "admission 缺权重", 409, False,
                             {"phase": "job_admission"})

    runtime.job_admission = admission
    try:
        try:
            q.submit(job_payload("admission"))
        except BusyError as e:
            assert e.code == "MISSING_UNITS", e.code
            assert e.http_status == 409
        else:
            raise AssertionError("job_admission 拒绝后仍入队")
        assert seen.get("owned") is True, seen
        assert seen.get("status") == "queued", seen
        assert q.snapshot()["counts"]["queued"] == 0
        return "锁内 hook=%s" % seen
    finally:
        q.stop()


def t_require_job_execution():
    runtime = FakeRuntime()
    q = QueueManager(runtime, {})       # 用生产 _run_job，才能在 dispatch 前调用 hook
    seen = []

    def execution_hook(job):
        seen.append(job.status)
        raise OperationError("MISSING_UNITS", "执行前缺权重", 409, False)

    runtime.require_job_execution = execution_hook
    try:
        job = q.submit(job_payload("exec-hook"))
        wait_for(lambda: job.status == "failed", "job execution hook failed")
        assert seen == ["running"], seen
        assert "MISSING_UNITS" in job.error, job.error
        return "执行前 hook=%s" % seen
    finally:
        q.stop()


def t_json_result_and_progress():
    q = FakeQueue()
    try:
        op = q.submit_operation("ctrl", {}, lambda o: b"raw-bytes")
        wait_status(q, op.id, "failed")
        pub = op.public()
        json.dumps(pub)                 # 必须能直接 JSON 化
        assert pub["result"] is None, pub
        assert pub["error"]["code"] == "EXECUTION_FAILED", pub["error"]
        assert pub["error"]["details"]["phase"] == "result", pub["error"]
        assert op.result is None

        def direct_progress(op):
            op.progress = b"raw-progress"
            return "ok"

        op2 = q.submit_operation("ctrl", {}, direct_progress)
        wait_status(q, op2.id, "failed")
        pub2 = op2.public()
        json.dumps(pub2)
        assert op2.progress is None, op2.progress
        assert pub2["error"]["details"]["phase"] == "progress", pub2["error"]

        def bad_set_stage(op):
            op.set_stage("step", progress=b"raw")
            return "unreachable"

        op3 = q.submit_operation("ctrl", {}, bad_set_stage)
        wait_status(q, op3.id, "failed")
        json.dumps(op3.public())
        assert op3.public()["error"]["details"]["phase"] == "progress"
        return "result/progress 非 JSON 均失败且清理"
    finally:
        q.stop()


def t_error_details_json_and_detach():
    q = FakeQueue()
    try:
        def cb(op):
            raise OperationError("MISSING_UNITS", "缺权重", 409, False,
                                 {"raw": b"not-json"})

        op = q.submit_operation("ctrl", {}, cb)
        wait_status(q, op.id, "failed")
        pub = op.public()
        json.dumps(pub)
        assert pub["error"]["code"] == "MISSING_UNITS", pub["error"]
        details = pub["error"]["details"]
        assert details["invalid"] == "dict", details
        assert "bytes" in details["reason"], details
        assert op.error.__traceback__ is None
        assert op.error.__cause__ is None
        assert op.error.__context__ is None
        return "details 已 sanitize，异常已脱栈"
    finally:
        q.stop()


def t_cleanup_and_synchronize():
    runtime = FakeRuntime()
    events = []
    sync_started = threading.Event()
    sync_release = threading.Event()

    def cleanup(op):
        events.append(("cleanup", op.status))

    def synchronize(op):
        events.append(("sync-start", op.status))
        sync_started.set()
        sync_release.wait(5.0)
        events.append(("sync-end", op.status))

    runtime.operation_cleanup = cleanup
    runtime.operation_synchronize = synchronize
    q = FakeQueue(runtime)
    try:
        op = q.submit_operation(
            "ctrl", {}, lambda o: (events.append(("callback", o.status)), "ok")[1])
        assert sync_started.wait(2.0), events
        # 同步未完成时仍是 running，control 保留位不能释放
        assert op.status == "running", op.status
        assert q.operations_snapshot()["running"] == op.id
        assert q.executor_state()["state"] == "running"
        expect_error(lambda: q.submit_operation("ctrl", {}, lambda o: "ok"),
                     "BUSY")
        sync_release.set()
        wait_status(q, op.id, "done")
        assert events == [("callback", "running"), ("cleanup", "running"),
                          ("sync-start", "running"), ("sync-end", "running")], events

        # cleanup 抛错：即使 callback 成功也失败；synchronize 仍被调用
        runtime2 = FakeRuntime()
        sync_calls = []

        def cleanup_bad(op):
            raise RuntimeError("cleanup failed")

        runtime2.operation_cleanup = cleanup_bad
        runtime2.operation_synchronize = lambda op: sync_calls.append(op.id)
        q2 = FakeQueue(runtime2)
        try:
            op2 = q2.submit_operation("ctrl", {}, lambda o: "ok")
            wait_status(q2, op2.id, "failed")
            assert op2.public()["error"]["code"] == "EXECUTION_FAILED"
            assert sync_calls == [op2.id], sync_calls
        finally:
            q2.stop()

        # callback 失败路径也要 cleanup
        runtime3 = FakeRuntime()
        cleanup_calls = []
        runtime3.operation_cleanup = lambda op: cleanup_calls.append(op.id)
        q3 = FakeQueue(runtime3)
        try:
            op3 = q3.submit_operation(
                "ctrl", {}, lambda o: (_ for _ in ()).throw(RuntimeError("bad")))
            wait_status(q3, op3.id, "failed")
            assert cleanup_calls == [op3.id], cleanup_calls
        finally:
            q3.stop()
        return "sync 前保持 running；成功/失败均 cleanup"
    finally:
        sync_release.set()
        q.stop()


def t_release_input_refs():
    class Cb:
        def __init__(self, node):
            self.node = node

        def __call__(self, op):
            return {"ok": True}

    class Pre:
        def __init__(self, node):
            self.node = node

        def __call__(self, kind, payload):
            return {"token": 1}

    q = FakeQueue()
    try:
        payload_node, cb_node, pre_node = RefNode(), RefNode(), RefNode()
        refs = [weakref.ref(payload_node), weakref.ref(cb_node),
                weakref.ref(pre_node)]
        op = q.submit_operation("ctrl", {"node": payload_node},
                                Cb(cb_node), preflight=Pre(pre_node))
        del payload_node, cb_node, pre_node
        wait_status(q, op.id, "done")
        assert op.payload is None and op.cb is None
        assert op.preflight is None and op.preflight_token is None
        gc.collect()
        assert all(r() is None for r in refs), [r() for r in refs]

        frame_refs = []

        class BadCb:
            def __call__(self, op):
                local = RefNode()
                frame_refs.append(weakref.ref(local))
                raise RuntimeError("boom")

        op2 = q.submit_operation("ctrl", {"n": 2}, BadCb())
        wait_status(q, op2.id, "failed")
        wait_for(lambda: q.executor_state()["state"] == "idle", "worker idle")
        assert op2.payload is None and op2.cb is None
        gc.collect()
        assert frame_refs and frame_refs[0]() is None, frame_refs

        q2 = PausedQueue()
        try:
            p_node, c_node = RefNode(), RefNode()
            refs2 = [weakref.ref(p_node), weakref.ref(c_node)]
            op3 = q2.submit_operation("ctrl", {"node": p_node}, Cb(c_node))
            del p_node, c_node
            q2.cancel_operation(op3.id)
            assert op3.payload is None and op3.cb is None
            gc.collect()
            assert all(r() is None for r in refs2), [r() for r in refs2]

            p_node, c_node = RefNode(), RefNode()
            refs3 = [weakref.ref(p_node), weakref.ref(c_node)]
            op4 = q2.submit_operation("ctrl", {"node": p_node}, Cb(c_node))
            del p_node, c_node
            q2.stop()
            assert op4.payload is None and op4.cb is None
            gc.collect()
            assert all(r() is None for r in refs3), [r() for r in refs3]
        finally:
            q2.stop()
        return "success/failure/cancel/shutdown 后闭包与 payload 回收"
    finally:
        q.stop()


print("== 统一 operation executor / admission 自检（无 torch） ==")
check("receipt / public view / result", t_receipt_and_result)
check("preflight 锁外且执行前重验", t_preflight_twice)
check("两个 control 并发最多一个获准", t_two_controls_atomic)
check("control vs job 原子互斥（多轮）", t_control_job_barrier)
check("job 运行中 job 仍可排队", t_jobs_queue_while_job_running)
check("control 占用时旧 job 409", t_control_rejects_job)
check("queued cancellation / 404", t_cancel_queued)
check("running cancellation 409", t_cancel_running_rejected)
check("request_id 相同请求返回同回执", t_request_id_dedup)
check("重复请求不再跑 preflight", t_dedup_skips_preflight)
check("request_id 去重有界可淘汰", t_request_dedup_bounded)
check("callback 异常不封锁队列", t_callback_exception_does_not_block)
check("preflight/提交失败路径", t_preflight_failure_paths)
check("runtime 可选 operation hooks", t_runtime_hooks)
check("runtime admission 拒绝", t_runtime_admission_denies)
check("runtime require_job（旧 job）", t_runtime_require_job)
check("snapshot 不等待 callback", t_snapshot_nonblocking)
check("客户端断线后服务继续", t_client_abandon_keeps_server_working)
check("shutdown 干净退出", t_shutdown_clean)
check("env_ready/legacy readiness 分支", t_readiness_env_ready)
check("job_admission 在锁内且可拒绝", t_job_admission_lock)
check("require_job_execution 执行前重验", t_require_job_execution)
check("result/progress 纯 JSON 校验", t_json_result_and_progress)
check("error details 纯 JSON 且脱栈", t_error_details_json_and_detach)
check("cleanup/synchronize 后才 terminal", t_cleanup_and_synchronize)
check("terminal 后释放 payload/闭包/帧", t_release_input_refs)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK（未 import torch：%s）"
      % ("yes" if "torch" not in sys.modules else "NO"))
