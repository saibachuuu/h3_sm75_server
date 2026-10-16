"""导演台：镜头序列（「后续镜头参考旧有镜头」= 帧级接力）。

一个 `kind="director"` 的任务里串行跑 N 个镜头，每个镜头复用 `pipeline.run_job`
生成一条 mp4；镜头 k 生成时按下一个镜头的需要把结尾若干帧存成 PNG
（`<id>.tail0.png` = 最后一帧），镜头 k+1 若开了「引用上段」就把这些 PNG 作为
参考图注入自己的 `image_b64s`（ref2va 最多 1–9 张；fl2va 只取最后那一帧）。

与 ComfyUI 导演台插件的差别（重要，别误以为可以照搬）：
插件的接续是把上一段的 **AV latent 尾窗**相位对齐后硬锁成关键帧条件
（`director/segment_continuity.py` / `h3_motion_context.py`），要改采样器内部，
API 层面搬不过来。本服务能做到的是**帧级接力** —— 同一意图下当前架构的最强形态：
上一镜头真实解码出来的画面，作为下一镜头的参考图，画面内容/人物/光线因此延续。

契约见仓库外的 `_ada_director_api.md`。
"""
import base64
import os
import random
import subprocess
import time


def _b64_from_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _next_shot_tail_count(shots, i, gp):
    """镜头 i 需要为下一个镜头产出几张尾帧（没有下一个 / 下一个不引用 → 0）。"""
    if i + 1 >= len(shots):
        return 0
    nxt = shots[i + 1]
    if not nxt.get("ref_prev"):
        return 0
    n = nxt.get("ref_prev_frames")
    if n is None:
        n = gp.get("ref_prev_frames", 1)
    return max(0, min(9, int(n or 0)))


def _concat_mp4(job, ctx, out_dir):
    """把各镜头 mp4 拼成总片 `<seqid>.mp4`。

    先试无损拼接（-c copy，同编码同参数时最快）；失败再退回重编码。
    任一步失败都不致命：日志记一句，总片留空，各镜头产物照旧可用。
    """
    parts = [s.get("mp4") for s in job.shots
             if s.get("status") == "done" and s.get("mp4")]
    if len(parts) < 2:
        return None
    ffmpeg = ctx["runtime"].ffmpeg
    tmp_dir = os.path.join(out_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    list_path = os.path.join(tmp_dir, job.id + ".concat.txt")
    final_name = job.id + ".mp4"
    final_path = os.path.join(out_dir, final_name)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.join(out_dir, p).replace("\\", "/"))

    base = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    attempts = (
        (base + ["-c", "copy", final_path], "copy"),
        (base + ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_path], "重编码"),
    )
    for cmd, label in attempts:
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        except Exception as e:                                  # noqa: BLE001
            job.log.append("拼接总片（%s）异常：%s" % (label, e))
            continue
        if proc.returncode == 0 and os.path.isfile(final_path):
            job.log.append("总片拼接（%s）：%s (%.1f MB)"
                           % (label, final_name, os.path.getsize(final_path) / 1e6))
            return final_name
        job.log.append("拼接总片（%s）失败 rc=%s: %s"
                       % (label, proc.returncode,
                          proc.stderr.decode("utf-8", "replace")[-400:]))
    return None


def run_director(job, ctx):
    """执行一条完整的镜头序列（在队列 worker 里串行跑，独占 GPU）。"""
    from common.protocol import validate_job
    from dit_svc import pipeline

    out_dir = ctx["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    gp = dict(job.params["global"])
    shots = job.shots
    total = len(shots)
    mode = gp.get("mode", "ref2va")
    prev_tail_b64 = []          # 上一镜头尾帧：[tail0（最后一帧）, tail1, …]
    n_done = 0
    stopped = False

    for i, shot in enumerate(shots):
        if stopped:
            shot["status"] = "skipped"
            shot["stage"] = "已跳过"
            continue

        shot["status"] = "running"
        shot["stage"] = "准备"
        job.stage = "镜头 %d/%d：准备" % (i + 1, total)
        t0 = time.time()

        # ---- 组装参考图：[上段尾帧…] + [自带图…]，ref2va 截到 9 张 ----
        own = list(shot.get("image_b64s") or [])
        want_prev = bool(shot.get("ref_prev")) and i > 0 and bool(prev_tail_b64)
        pfr = int(shot.get("ref_prev_frames") or 0)
        used_prev = 0
        if want_prev and mode == "fl2va":
            refs = [prev_tail_b64[0]]                    # fl2va 只能 1 张 → 取最后那一帧
            used_prev = 1
        elif want_prev:
            take = prev_tail_b64[:pfr] if pfr > 0 else []
            refs = (take + own)[:9]
            used_prev = len(take)
        else:
            refs = own[:9]
        shot["used_prev_frames"] = used_prev

        if not refs:
            shot["status"] = "failed"
            shot["stage"] = "失败"
            shot["error"] = "没有可用参考图（首镜头必须自带参考图）"
            shot["elapsed"] = round(time.time() - t0, 1)
            for s2 in shots[i + 1:]:
                s2["status"] = "skipped"
                s2["stage"] = "已跳过"
            raise ValueError("镜头 %d：没有可用参考图" % (i + 1))

        names = list(shot.get("image_names") or [])
        if used_prev:
            names = ["上段尾帧%d.png" % (k + 1) for k in range(used_prev)] + names

        payload = dict(gp)
        for k in ("ref_prev_frames", "concat", "on_error"):
            payload.pop(k, None)
        payload.update({
            "prompt": shot["prompt"],
            "image_b64s": refs,
            "image_name": (names[0] if names else "ref.png"),
            "length": shot["length"],
            "seed": shot["seed"],
            "dump_tail_frames": _next_shot_tail_count(shots, i, gp),
        })

        params, err = validate_job(payload)
        if err:
            shot["status"] = "failed"
            shot["stage"] = "失败"
            shot["error"] = "参数校验失败：%s" % err
            shot["elapsed"] = round(time.time() - t0, 1)
            for s2 in shots[i + 1:]:
                s2["status"] = "skipped"
                s2["stage"] = "已跳过"
            raise ValueError("镜头 %d：%s" % (i + 1, err))

        # 单任务提交时队列会把 -1 换成随机 seed；序列直接调 pipeline，这里补上
        if params["seed"] is None or params["seed"] < 0:
            params["seed"] = random.randint(0, 2 ** 63 - 1)

        sub = _ShotJob(job, params, i + 1, total)
        sub.id = "%s-s%d" % (job.id, i + 1)
        job.log.append("[镜头 %d/%d] %s seed=%s length=%s 参考图 %d 张（其中上段 %d）"
                       % (i + 1, total, sub.id, params["seed"], params["length"],
                          len(refs), used_prev))

        try:
            pipeline.run_job(sub, ctx)
        except Exception as e:                                  # noqa: BLE001
            shot["status"] = "failed"
            shot["stage"] = "失败"
            shot["error"] = "%s: %s" % (type(e).__name__, e)
            shot["elapsed"] = round(time.time() - t0, 1)
            job.log.append("[镜头 %d/%d] 失败：%s" % (i + 1, total, shot["error"]))
            job.log.extend("[镜头 %d] %s" % (i + 1, ln) for ln in sub.log[-30:])
            if gp.get("on_error") == "continue":
                prev_tail_b64 = []
                for s2 in shots[i + 1:]:
                    s2["ref_prev"] = False        # 断了链，后续退化为不引用
                continue
            for s2 in shots[i + 1:]:
                s2["status"] = "skipped"
                s2["stage"] = "已跳过"
            raise

        # ---- 收集本镜头产出 ----
        job.log.extend("[镜头 %d] %s" % (i + 1, ln) for ln in sub.log)
        shot["status"] = "done"
        shot["stage"] = "完成"
        shot["seed"] = params["seed"]
        shot["elapsed"] = round(time.time() - t0, 1)
        shot["mp4"] = sub.mp4
        shot["output_url"] = ("/outputs/" + sub.mp4) if sub.mp4 else None
        shot["tail_urls"] = list((sub.files or {}).get("tail_pngs") or [])
        n_done += 1

        # ---- 给下一个镜头准备尾帧 base64 ----
        prev_tail_b64 = []
        for url in shot["tail_urls"]:
            p = os.path.join(out_dir, os.path.basename(url))
            if os.path.isfile(p):
                prev_tail_b64.append(_b64_from_file(p))

    # ---- 收尾：拼总片 ----
    if gp.get("concat", True) and n_done >= 2:
        job.stage = "拼接总片"
        final = _concat_mp4(job, ctx, out_dir)
        if final:
            job.mp4 = final
            job.concat_mp4 = final
            job.files["concat"] = "/outputs/" + final
    if job.mp4 is None and n_done >= 1:
        # 没拼总片（关了拼接 / 只成功一个镜头 / 拼接失败）：顶层 mp4 指向最后成功的
        # 镜头，保证轮询端至少有一条能播的成片；逐镜头产物在 shots[] 里各给各的。
        last = next((s for s in reversed(shots)
                     if s.get("status") == "done" and s.get("mp4")), None)
        if last:
            job.mp4 = last["mp4"]

    job.log.append("序列结束：%d/%d 镜头成功%s"
                   % (n_done, total, "，总片 %s" % job.mp4 if job.mp4 else ""))


class _ShotJob:
    """单个镜头的子任务壳。

    不继承 `Job`：只实现 `pipeline.run_job` 真正用到的那些属性，并把 `stage`
    的写入同步回序列任务与镜头视图，这样 `/api/jobs` 轮询能看到实时阶段。
    """

    def __init__(self, parent, params, shot_no, total):
        self._parent = parent
        self._shot_no = shot_no
        self._total = total
        self.kind = "video"
        self.params = params
        self.status = "running"
        self.stage = "准备"
        self.created = time.time()
        self.started = time.time()
        self.finished = None
        self.error = None
        self.log = []
        self.timings = {}
        self.cond_stats = None
        self.cond_source = None
        self.latent_probe = None
        self.mp4 = None
        self.files = {}
        self.id = "shot"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "stage":
            parent = self.__dict__.get("_parent")
            if parent is None:
                return
            no = self.__dict__.get("_shot_no", 0)
            parent.stage = "镜头 %d/%d：%s" % (no, self.__dict__.get("_total", no), value)
            try:
                shot = parent.shots[no - 1]
                shot["stage"] = value
                shot["elapsed"] = round(
                    time.time() - self.__dict__.get("started", time.time()), 1)
            except (IndexError, KeyError, TypeError):
                pass

    def public(self, with_log=False):
        return {"id": self.id, "status": self.status, "stage": self.stage}
