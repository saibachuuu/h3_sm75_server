"""API 约定与默认参数（cond server 与 dit server 共用）。

本模块只依赖标准库，两个入口都可以安全 import。
"""
import json

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

# cond 文件契约：与 embed_h3.py 输出一致，infer_h3_turing.read_cond 可直接消费
COND_KEYS_REQUIRED = ("cond", "minimax_token_tags")
COND_KEYS_FRAME = ("first_frame", "ref_image")  # 至少存在其一


def dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def validate_job(payload: dict):
    """校验 POST /api/jobs 的 JSON。返回 (cleaned_params, None) 或 (None, 错误信息)。"""
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "prompt 不能为空"
    prompt = prompt.strip()

    image_b64 = payload.get("image_b64")
    if not isinstance(image_b64, str) or not image_b64:
        return None, "缺少参考图 image_b64（ref2va/fl2va 均需要图像输入）"

    def _int(key, lo, hi):
        v = payload.get(key, DEFAULTS.get(key))
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None, f"{key} 必须是整数"
        if not (lo <= v <= hi):
            return None, f"{key} 超出范围 [{lo}, {hi}]"
        return v, None

    out = {"prompt": prompt, "image_b64": image_b64,
           "image_name": str(payload.get("image_name") or "ref.png")[:200]}

    mode = payload.get("mode", DEFAULTS["mode"])
    if mode not in ("ref2va", "fl2va"):
        return None, "mode 只能是 ref2va 或 fl2va"
    out["mode"] = mode

    ref_size = payload.get("ref_image_size", DEFAULTS["ref_image_size"])
    if ref_size not in ("match", "max"):
        return None, "ref_image_size 只能是 match 或 max"
    out["ref_image_size"] = ref_size

    sampler = payload.get("sampler", DEFAULTS["sampler"])
    if sampler not in SAMPLERS:
        return None, f"sampler 不支持：{sampler}"
    out["sampler"] = sampler

    scheduler = payload.get("scheduler", DEFAULTS["scheduler"])
    if scheduler not in SCHEDULERS:
        return None, f"scheduler 不支持：{scheduler}"
    out["scheduler"] = scheduler

    for key, lo, hi in (("width", 256, 1600), ("height", 256, 1600),
                        ("length", 16, 400), ("steps", 1, 50),
                        ("refine_steps", 0, 20), ("seed", -1, 2 ** 63 - 1)):
        v, err = _int(key, lo, hi)
        if err:
            return None, err
        out[key] = v

    try:
        out["refine_denoise"] = float(payload.get("refine_denoise", DEFAULTS["refine_denoise"]))
    except (TypeError, ValueError):
        return None, "refine_denoise 必须是数字"
    if not (0.0 < out["refine_denoise"] <= 1.0):
        return None, "refine_denoise 必须在 (0, 1] 内"

    # 画布需对齐 32（与 _empty_av_latent / embed 对齐逻辑一致）
    if out["width"] % 32 or out["height"] % 32:
        return None, "width / height 必须是 32 的倍数"

    out["sigmas"] = payload.get("sigmas")  # 可选，逗号分隔，覆盖 scheduler/steps
    if out["sigmas"] is not None:
        if not isinstance(out["sigmas"], str):
            return None, "sigmas 必须是逗号分隔的字符串"
        try:
            [float(s) for s in out["sigmas"].split(",") if s.strip()]
        except ValueError:
            return None, "sigmas 格式错误（应如 1.0,0.5,0.0）"
    return out, None
