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
    def __init__(self, params, kind="video"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.id = "%s-%s" % (ts, uuid.uuid4().hex[:4])
        self.kind = kind  # video | warmup
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
            "prompt": (self.params.get("prompt") or self.params.get("cond_label") or "")[:120]
                      if self.kind == "video" else ("warmup n=%s" % self.params.get("tokens")),
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
    def _require_ready(self):
        if self.runtime.error:
            raise BusyError("模型加载失败: %s" % self.runtime.error)
        if not self.runtime.ready:
            raise BusyError("模型加载中（%s，已 %.0f 秒），请稍后再提交"
                            % (self.runtime.stage, self.runtime.stage_elapsed()))

    def submit(self, payload):
        from common.protocol import validate_job

        self._require_ready()
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
        with self._cond:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self._cond.notify_all()
        print("[queue] submitted %s (直接喂 cond: %s, %d 图, %d bytes) seed=%d"
              % (job.id, params["cond_label"], len(params["image_b64s"]),
                 cond["size"], params["seed"]), flush=True)
        return job

    # ---------- warmup 提交（与视频任务同队列串行，互不抢显存） ----------
    def submit_warmup(self, tokens, repeat=1):
        self._require_ready()
        if self.runtime.attn_override is None:
            raise ValueError("当前 attention（%s）无 override 可测，warmup 仅支持 sage/w8a8/veda"
                             % self.runtime.attn)
        if not isinstance(tokens, int) or not (16 <= tokens <= 40000):
            raise ValueError("tokens 须为 16–40000 的整数")
        if not isinstance(repeat, int) or not (1 <= repeat <= 50):
            raise ValueError("repeat 须为 1–50 的整数")
        job = Job({"tokens": tokens, "repeat": repeat}, kind="warmup")
        job.stage = "warmup 排队中"
        with self._cond:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self._cond.notify_all()
        print("[queue] submitted warmup %s tokens=%d x%d" % (job.id, tokens, repeat), flush=True)
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
            print("[queue] running %s (%s)" % (job.id, job.kind), flush=True)
            try:
                if job.kind == "warmup":
                    self._run_warmup(job)
                else:
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

    # ---------- warmup 执行（队列内串行；用当前 override + fallback 落点） ----------
    def _run_warmup(self, job):
        from dit_svc import veda_attn
        runtime = self.runtime
        torch = runtime.torch
        tokens = job.params["tokens"]
        repeat = job.params.get("repeat", 1)
        job.stage = "warmup 执行中（n=%d x%d）" % (tokens, repeat)
        override = runtime.attn_override
        attention = runtime.attention
        fallback = getattr(attention, "optimized_attention", None)
        if fallback is None:
            raise RuntimeError("无 fallback attention 可用")

        def _call(q, k, v, heads):
            return override(fallback, q, k, v, heads, skip_reshape=True)
        result = veda_attn.run_single_warmup(
            torch, runtime.device, _call, tokens, repeat=repeat,
            log=lambda s: (job.log.append(s), print("[queue %s] %s" % (job.id, s), flush=True)))
        job.timings["warmup"] = result["ms_max"]
        job.cond_stats = {"warmup": result}
        job.log.append("warmup 结果: %s" % result)
