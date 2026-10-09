"""串行任务队列（单 GPU，一次只跑一个；手动逐个提交）。"""
import json
import random
import threading
import time
import traceback
import uuid


class BusyError(Exception):
    """服务未就绪（模型加载中/加载失败）。"""


class Job:
    def __init__(self, params):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.id = "%s-%s" % (ts, uuid.uuid4().hex[:4])
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
        self.latent_probe = None
        self.mp4 = None

    def public(self, with_log=False):
        d = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - (self.created or time.time()), 1),
            "prompt": self.params.get("prompt", "")[:120],
            "seed": self.params.get("seed"),
            "mode": self.params.get("mode"),
            "length": self.params.get("length"),
            "steps": self.params.get("steps"),
            "mp4": self.mp4,
            "output_url": ("/outputs/" + self.mp4) if self.mp4 else None,
            "error": self.error,
            "timings": self.timings,
            "cond_stats": self.cond_stats,
            "latent_probe": self.latent_probe,
            "log_tail": self.log[-40:],
        }
        if not with_log:
            d.pop("log_tail")
        return d


class QueueManager:
    def __init__(self, runtime, ctx):
        self.runtime = runtime
        self.ctx = ctx
        self.jobs = {}          # id -> Job
        self.order = []         # 保持提交顺序
        self._cond = threading.Condition()
        self._worker = threading.Thread(target=self._loop, name="h3-worker", daemon=True)
        self._worker.start()

    # ---------- 提交 ----------
    def submit(self, payload):
        from common.protocol import validate_job

        if self.runtime.error:
            raise BusyError("模型加载失败: %s" % self.runtime.error)
        if not self.runtime.ready:
            raise BusyError("模型加载中（%s，已 %.0f 秒），请稍后再提交"
                            % (self.runtime.stage, self.runtime.stage_elapsed()))
        params, err = validate_job(payload)
        if err:
            raise ValueError(err)
        if params["seed"] is None or params["seed"] < 0:
            params["seed"] = random.randint(0, 2 ** 63 - 1)
        job = Job(params)
        with self._cond:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self._cond.notify_all()
        print("[queue] submitted %s seed=%d" % (job.id, params["seed"]), flush=True)
        return job

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

    # ---------- worker ----------
    def _loop(self):
        from dit_svc import pipeline
        while True:
            with self._cond:
                job = None
                for jid in self.order:
                    cand = self.jobs[jid]
                    if cand.status == "queued":
                        job = cand
                        break
                if job is None:
                    self._cond.wait(timeout=5.0)
                    continue
                job.status = "running"
                job.started = time.time()
            print("[queue] running %s" % job.id, flush=True)
            try:
                pipeline.run_job(job, self.ctx)
                job.status = "done"
                job.stage = "完成"
            except Exception as e:
                job.status = "failed"
                job.stage = "失败"
                job.error = "%s: %s" % (type(e).__name__, e)
                job.log.append("ERROR: %s" % job.error)
                job.log.append(traceback.format_exc()[-2000:])
                traceback.print_exc()
                print("[queue] job %s FAILED: %s" % (job.id, job.error), flush=True)
            finally:
                job.finished = time.time()
