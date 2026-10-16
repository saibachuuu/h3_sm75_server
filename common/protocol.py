"""API 约定与默认参数（cond server 与 dit server 共用）。

本模块只依赖标准库，两个入口都可以安全 import。
"""
import base64
import binascii
import json
import os

DEFAULT_COND_PORT = 8189
DEFAULT_DIT_PORT = 8190
# 远程 dit server 默认经 ssh -R 9310 -> 本机 cond server 8189 回调（方案 A）
DEFAULT_COND_URL = "http://127.0.0.1:9310"

# H3 生成默认值：与 infer_h3_turing.py 已验证的单 pass 基线保持一致
# （refine_steps=0 是已验收配置；双 pass 尚未验收，只作为可选项暴露）
DEFAULTS = {
    "mode": "ref2va",           # ref2va | fl2va
    "width": 640,
    "height": 960,
    "length": 124,              # 帧数（24 fps）
    "steps": 4,
    "sampler": "res_multistep",
    "scheduler": "simple",
    "seed": 12345,              # -1 = 提交时随机
    "refine_steps": 0,
    "refine_denoise": 0.85,
    "ref_image_size": "match",  # match | max
    "fps": 24,
}

SAMPLERS = (
    "res_multistep", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
    "ddim", "uni_pc", "lms", "heun", "dpm_fast", "dpm_adaptive",
)
SCHEDULERS = ("simple", "normal", "karras", "exponential", "sgm_uniform", "beta", "linear")

# cond 文件契约（slim：只含文本编码器输出，不带图像像素；参考帧由 dit 侧
# 从任务 image_b64 本地解码 + VAE 编码）
COND_KEYS_REQUIRED = ("cond", "minimax_token_tags")

# 直接喂 cond（POST /api/jobs/cond）时的 cond 负载上限：内联 base64 或 cond_file 原始字节
MAX_COND_BYTES = 96 * 1024 * 1024

# 参考帧目标尺寸（cond 侧 Qwen 输入与 dit 侧 VAE 输入共用，保持一致）
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048


def target_frame_size(mode, raw_w, raw_h, width, height, ref_image_size="match"):
    """返回 (tw, th)。fl2va 直接对齐画布；ref2va 按策略等比缩小后向 32 对齐。"""
    import math
    if mode == "fl2va":
        return int(width), int(height)
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (raw_w * raw_h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(raw_w, raw_h))
    tw = max(CANVAS_MULTIPLE, round(raw_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(raw_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _bool_param(payload, key, default):
    """宽松布尔：bool / 0,1 / "true|false|yes|no|on|off"。返回 (值, 错误)。"""
    v = payload.get(key, default)
    if isinstance(v, bool):
        return v, None
    if isinstance(v, int) and v in (0, 1):
        return bool(v), None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True, None
        if s in ("false", "0", "no", "off"):
            return False, None
    return None, "%s 必须是布尔值（true/false）" % key


def validate_vae_probe(payload: dict):
    """校验 POST /api/vae_probe（只跑视频 VAE 编解码，不进 DiT、不采样）。

    两个探针至少选一个：
      image_b64  给了就做编码探针（+ round-trip 重建对比与 PNG，write_png 控制）
      decode     true（默认）就按 width/height/length 造 latent 做整段解码探针
    返回 (cleaned_params, None) 或 (None, 错误信息)。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    out = {"label": "vae_probe"}
    img = payload.get("image_b64")
    if img is not None:
        if not isinstance(img, str) or not img.strip():
            return None, "image_b64 必须是非空 base64 字符串"
        out["image_b64"] = img.strip()
        out["image_name"] = str(payload.get("image_name") or "probe.png")[:200]

    mode = payload.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return None, "mode 只能是 ref2va 或 fl2va"
    out["mode"] = mode

    ref_size = payload.get("ref_image_size", DEFAULTS["ref_image_size"])
    if ref_size not in ("match", "max"):
        return None, "ref_image_size 只能是 match 或 max"
    out["ref_image_size"] = ref_size

    def _int(key, lo, hi, default):
        v = payload.get(key, default)
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None, "%s 必须是整数" % key
        if not (lo <= v <= hi):
            return None, "%s 超出范围 [%d, %d]" % (key, lo, hi)
        return v, None

    for key, lo, hi in (("width", 256, 1600), ("height", 256, 1600),
                        ("length", 16, 400), ("fps", 1, 60)):
        v, err = _int(key, lo, hi, DEFAULTS.get(key))
        if err:
            return None, err
        out[key] = v
    v, err = _int("seed", -1, 2 ** 63 - 1, DEFAULTS["seed"])
    if err:
        return None, err
    out["seed"] = v
    if out["width"] % 32 or out["height"] % 32:
        return None, "width / height 必须是 32 的倍数"

    latent = payload.get("latent", "noise")
    if latent not in ("noise", "zeros"):
        return None, "latent 只能是 noise（seed 随机）或 zeros"
    out["latent"] = latent

    for key in ("decode", "write_mp4", "write_png"):
        v, err = _bool_param(payload, key, True)
        if err:
            return None, err
        out[key] = v
    if not out["decode"] and "image_b64" not in out:
        return None, "至少要给 image_b64（编码探针）或 decode=true（解码探针）"
    return out, None

def _image_payloads(payload, mode, min_count, max_count):
    """取参考图 base64 列表（image_b64s，兼容旧单图 image_b64）。返回 (list, 错误)。"""
    b64s = payload.get("image_b64s")
    if b64s is None:
        # 兼容旧单图客户端
        b64 = payload.get("image_b64")
        b64s = [b64] if isinstance(b64, str) and b64 else []
    if (not isinstance(b64s, list) or not (min_count <= len(b64s) <= max_count)
            or not all(isinstance(s, str) and s for s in b64s)):
        compat = "（旧单图 image_b64 仍兼容）" if min_count else ""
        return None, ("参考图 image_b64s 须为 %d–%d 张 base64 字符串%s"
                      % (min_count, max_count, compat))
    if mode == "fl2va" and len(b64s) > 1:
        return None, "fl2va 仅支持单图，多图请用 ref2va"
    return b64s, None


def _fill_sampling_params(payload, out):
    """mode / ref_image_size / sampler / scheduler / 尺寸 / 步数 / seed / sigmas → out。

    /api/jobs 与 /api/jobs/cond 共用。返回错误字符串或 None。
    """
    mode = payload.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return "mode 只能是 ref2va 或 fl2va"
    out["mode"] = mode

    ref_size = payload.get("ref_image_size", DEFAULTS["ref_image_size"])
    if ref_size not in ("match", "max"):
        return "ref_image_size 只能是 match 或 max"
    out["ref_image_size"] = ref_size

    sampler = payload.get("sampler", DEFAULTS["sampler"])
    if sampler not in SAMPLERS:
        return "sampler 不支持：%s" % sampler
    out["sampler"] = sampler

    scheduler = payload.get("scheduler", DEFAULTS["scheduler"])
    if scheduler not in SCHEDULERS:
        return "scheduler 不支持：%s" % scheduler
    out["scheduler"] = scheduler

    def _int(key, lo, hi):
        v = payload.get(key, DEFAULTS.get(key))
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None, "%s 必须是整数" % key
        if not (lo <= v <= hi):
            return None, "%s 超出范围 [%d, %d]" % (key, lo, hi)
        return v, None

    for key, lo, hi in (("width", 256, 1600), ("height", 256, 1600),
                        ("length", 16, 400), ("steps", 1, 50),
                        ("refine_steps", 0, 20), ("seed", -1, 2 ** 63 - 1)):
        v, err = _int(key, lo, hi)
        if err:
            return err
        out[key] = v

    try:
        out["refine_denoise"] = float(payload.get("refine_denoise", DEFAULTS["refine_denoise"]))
    except (TypeError, ValueError):
        return "refine_denoise 必须是数字"
    if not (0.0 < out["refine_denoise"] <= 1.0):
        return "refine_denoise 必须在 (0, 1] 内"

    # 画布需对齐 32（与 _empty_av_latent / embed 对齐逻辑一致）
    if out["width"] % 32 or out["height"] % 32:
        return "width / height 必须是 32 的倍数"

    # 可选：把该镜头结尾 N 帧存成 PNG（<id>.tail0.png = 最后一帧），供「后续镜头
    # 参考旧有镜头」的帧级接力；导演台序列任务内部会用到。
    v = payload.get("dump_tail_frames", 0)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "dump_tail_frames 必须是整数"
    if not (0 <= v <= 9):
        return "dump_tail_frames 超出范围 [0, 9]"
    out["dump_tail_frames"] = v

    out["sigmas"] = payload.get("sigmas")  # 可选，逗号分隔，覆盖 scheduler/steps
    if out["sigmas"] is not None:
        if not isinstance(out["sigmas"], str):
            return "sigmas 必须是逗号分隔的字符串"
        try:
            [float(s) for s in out["sigmas"].split(",") if s.strip()]
        except ValueError:
            return "sigmas 格式错误（应如 1.0,0.5,0.0）"

    # 可选诊断：把第一个采样步的 CUDA kernel 明细（含 python 调用栈）写进任务日志
    v, err = _bool_param(payload, "profile_attn", False)
    if err:
        return err
    out["profile_attn"] = v
    return None


def validate_job(payload: dict):
    """校验 POST /api/jobs 的 JSON（prompt + 1–9 张参考图 → 回调本机 cond server 编码）。

    返回 (cleaned_params, None) 或 (None, 错误信息)。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "prompt 不能为空"

    mode = payload.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return None, "mode 只能是 ref2va 或 fl2va"
    b64s, err = _image_payloads(payload, mode, 1, 9)
    if err:
        return None, err

    out = {"prompt": prompt.strip(), "image_b64s": b64s,
           # 首图兼容：dit 本地解码 / sidecar 沿用旧字段名
           "image_b64": b64s[0],
           "image_name": str(payload.get("image_name") or "ref.png")[:200]}
    err = _fill_sampling_params(payload, out)
    if err:
        return None, err
    return out, None


def validate_cond_job(payload: dict, cond_base_dir=None):
    """校验 POST /api/jobs/cond（直接喂 cond 生成任务，任务内不回调 cond server）。

    cond 来源必须且只能给一个：
      cond_b64   cond safetensors 字节的 base64（内联上传，≤ MAX_COND_BYTES 原始字节）
      cond_file  dit 机器上已有的 cond 文件：绝对路径，或相对 cond_base_dir（默认 out_dir）的文件名
    参考图 image_b64s 可选（0–9 张）：给了就在 dit 侧本地解码 + VAE 编码成
    minimax_refs（ref2va）或 minimax_keyframes（fl2va）块；不给就是纯文本条件
    （只用 cond / minimax_token_tags[+ attention_mask / pooled_output]）。
    prompt 可选，仅作任务标签与归档。

    返回 ({"params": ..., "cond": {"kind": "b64"|"file", ...}}, None) 或 (None, 错误信息)。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    mode = payload.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return None, "mode 只能是 ref2va 或 fl2va"
    b64s, err = _image_payloads(payload, mode, 0, 9)
    if err:
        return None, err

    cond_b64 = payload.get("cond_b64")
    cond_file = payload.get("cond_file")
    if bool(cond_b64) == bool(cond_file):
        return None, "cond 来源必须且只能给一个：cond_b64（内联 base64）或 cond_file（文件路径）"

    if cond_b64:
        if not isinstance(cond_b64, str):
            return None, "cond_b64 必须是 base64 字符串"
        limit_mb = MAX_COND_BYTES // (1 << 20)
        if len(cond_b64) > MAX_COND_BYTES * 4 // 3 + 4:
            return None, "cond_b64 过大（原始字节上限 %d MB）" % limit_mb
        try:
            data = base64.b64decode(cond_b64, validate=True)
        except (binascii.Error, ValueError):
            return None, "cond_b64 不是合法 base64"
        if not (0 < len(data) <= MAX_COND_BYTES):
            return None, "cond_b64 为空或超过 %d MB" % limit_mb
        cond = {"kind": "b64", "bytes": data, "size": len(data)}
        label = "cond:inline"
    else:
        if not isinstance(cond_file, str) or not cond_file.strip():
            return None, "cond_file 必须是非空字符串路径"
        path = cond_file.strip()
        if not os.path.isabs(path):
            if not cond_base_dir:
                return None, "cond_file 用相对路径时必须提供 cond_base_dir（out_dir）"
            path = os.path.join(cond_base_dir, path)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return None, "cond_file 不存在或不是普通文件: %s" % path
        size = os.path.getsize(path)
        if not (0 < size <= MAX_COND_BYTES):
            return None, "cond_file 为空或超过 %d MB: %s" % (MAX_COND_BYTES // (1 << 20), path)
        cond = {"kind": "file", "path": path, "size": size}
        label = "cond:" + os.path.basename(path)

    prompt = payload.get("prompt")
    out = {"prompt": prompt.strip() if isinstance(prompt, str) else "",
           "image_b64s": b64s,
           "image_b64": b64s[0] if b64s else None,
           "image_name": str(payload.get("image_name") or "ref.png")[:200],
           "cond_label": label[:200]}
    err = _fill_sampling_params(payload, out)
    if err:
        return None, err
    return {"params": out, "cond": cond}, None


# ---------------------------------------------------------------- 导演台（镜头序列）

# 一条序列最多几个镜头（防手滑提交超长序列占满 GPU）
DIRECTOR_MAX_SHOTS = 24


def validate_director(payload: dict):
    """校验 POST /api/director（镜头序列；后续镜头可参考上一镜头的尾帧）。

    返回 ({"global": {...}, "shots": [...]}, None) 或 (None, 错误信息)。
    每个镜头里 length / seed / ref_prev_frames 都在这一步解析成具体值，
    执行侧（dit_svc/director.py）不再猜默认值。

    硬约束：每个镜头最终必须能拿到 ≥1 张参考图 —— 首镜头必须自带图；
    后续镜头「自带图 ≥1」或「ref_prev=true」二选一。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    g = payload.get("global") or {}
    if not isinstance(g, dict):
        return None, "global 必须是 JSON 对象"

    shots_in = payload.get("shots")
    if not isinstance(shots_in, list) or not shots_in:
        return None, "shots 至少要有 1 个镜头"
    if len(shots_in) > DIRECTOR_MAX_SHOTS:
        return None, "镜头数上限 %d（当前 %d）" % (DIRECTOR_MAX_SHOTS, len(shots_in))

    mode = g.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return None, "mode 只能是 ref2va 或 fl2va"

    # 公共参数借用单任务校验拿取值范围，再摘掉单镜头专属字段
    probe = dict(g)
    probe["prompt"] = "probe"
    probe["image_b64s"] = ["AA=="]
    gp, err = validate_job(probe)
    if err:
        return None, "global 参数有误：%s" % err
    for k in ("prompt", "image_b64s", "image_b64", "image_name", "dump_tail_frames"):
        gp.pop(k, None)

    v = g.get("ref_prev_frames", 1)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None, "ref_prev_frames 必须是整数"
    if not (0 <= v <= 9):
        return None, "ref_prev_frames 超出范围 [0, 9]"
    gp["ref_prev_frames"] = v

    ok, err = _bool_param(g, "concat", True)
    if err:
        return None, err
    gp["concat"] = ok

    on_error = g.get("on_error", "stop")
    if on_error not in ("stop", "continue"):
        return None, "on_error 只能是 stop 或 continue"
    gp["on_error"] = on_error

    gseed = gp.get("seed")

    shots = []
    for i, s in enumerate(shots_in):
        if not isinstance(s, dict):
            return None, "镜头 %d 必须是 JSON 对象" % (i + 1)
        prompt = s.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None, "镜头 %d：prompt 不能为空" % (i + 1)

        b64s = s.get("image_b64s")
        if b64s is None:
            b64s = []
        if not isinstance(b64s, list) or not all(isinstance(x, str) and x for x in b64s):
            return None, "镜头 %d：image_b64s 必须是 base64 字符串列表" % (i + 1)
        if len(b64s) > 9:
            return None, "镜头 %d：自带参考图最多 9 张" % (i + 1)

        try:
            length = int(s.get("length", DEFAULTS["length"]))
        except (TypeError, ValueError):
            return None, "镜头 %d：length 必须是整数" % (i + 1)
        if not (16 <= length <= 400):
            return None, "镜头 %d：length 超出范围 [16, 400]" % (i + 1)

        ref_prev, err = _bool_param(s, "ref_prev", i > 0)
        if err:
            return None, "镜头 %d：%s" % (i + 1, err)
        if i == 0:
            ref_prev = False          # 首镜头没有上段可引用

        rpf = s.get("ref_prev_frames")
        rpf = v if rpf is None else rpf
        try:
            rpf = int(rpf)
        except (TypeError, ValueError):
            return None, "镜头 %d：ref_prev_frames 必须是整数" % (i + 1)
        if not (0 <= rpf <= 9):
            return None, "镜头 %d：ref_prev_frames 超出范围 [0, 9]" % (i + 1)

        seed = s.get("seed")
        if seed is None:
            if gseed is None:
                seed = DEFAULTS["seed"]
            elif gseed < 0:
                seed = -1             # 每镜头各自随机
            else:
                seed = gseed + i
        else:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                return None, "镜头 %d：seed 必须是整数" % (i + 1)
            if not (-1 <= seed <= 2 ** 63 - 1):
                return None, "镜头 %d：seed 超出范围" % (i + 1)

        names = s.get("image_names") or []
        if not isinstance(names, list):
            names = []
        names = [str(n)[:200] for n in names]

        if not b64s and not (i > 0 and ref_prev):
            return None, ("镜头 %d：必须自带参考图（第 2 个镜头起也可以改为开启"
                          "「引用上段」）" % (i + 1))
        if mode == "fl2va" and len(b64s) > 1:
            return None, "镜头 %d：fl2va 每镜头最多 1 张自带参考图" % (i + 1)

        shots.append({
            "index": i, "prompt": prompt.strip(), "length": length, "seed": seed,
            "ref_prev": ref_prev, "ref_prev_frames": rpf,
            "image_b64s": b64s, "image_names": names,
        })

    return {"global": gp, "shots": shots}, None
