"""dit server 新协议（api_version=2）请求校验 —— 仅依赖标准库。

与 ``common/protocol.py`` 的分工：
- 旧 jobs / director / vae_probe 的共享规则保持不动，本模块只做新原语的严格校验；
- 采样参数取值范围复用 common 的私有 helper（``_image_payloads`` /
  ``_fill_sampling_params``），但本模块不修改 common，避免影响 cond server；
- 错误统一抛 :class:`ApiError`，带 ``code`` / ``http_status`` / ``retryable`` /
  ``details``，主代理接入 ``dit_svc/operations.py`` 时可直接转 operation 错误或
  error envelope（``to_error()`` 已给出 03 号文档的 envelope 形状）。

strict 约定（03 §1）：
- 新布尔字段只接受 JSON bool（0/1/"true" 一律拒绝）；
- 新数字字段只接受 JSON int（不接受 "124" 这类字符串），范围仍按 common 规则；
- 请求体未知字段直接拒绝，不静默忽略拼错字段。
"""
import base64
import binascii
import re

from common.protocol import (
    DEFAULTS,
    MAX_COND_BYTES,
    _fill_sampling_params,
    _image_payloads,
)

# ------------------------------------------------------------------ 常量

WEIGHTS_ACTIONS = ("disk2ram", "ram2vram", "vram2ram", "purgefromram")
MODEL_IDS = ("dit", "vae.video", "vae.audio", "te", "veda.bundle")
SELECTOR_MODES = ("all", "units")

MAX_SELECTORS = 4096            # 单请求选择器条数上限
MAX_RANGE_SPAN = 4096           # 单个 a-b 区间展开长度上限
MAX_REQUEST_ID_LEN = 200
MAX_PROMPT_CHARS = 8000
MAX_FILE_REF_CHARS = 4096
MAX_IMAGE_B64_CHARS = 48 * 1024 * 1024      # 单张 base64 字符上限（≈36MiB 原始）
MAX_IMAGES_TOTAL_BYTES = 64 * 1024 * 1024   # 解码后参考图总字节上限

ARTIFACT_ID_RE = re.compile(r"^(cond|latent)_[0-9a-f]{32}$")

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_SELECTOR_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+$")
_RANGE_TAIL_RE = re.compile(r"^(\d+)-(\d+)$")


# ------------------------------------------------------------------ 错误

class ApiError(Exception):
    """带协议分类的错误；主代理可将其映射到 operations 的错误类型。"""

    def __init__(self, message, *, code="INVALID_REQUEST", http_status=400,
                 retryable=False, details=None):
        super(ApiError, self).__init__(message)
        self.code = code
        self.http_status = int(http_status)
        self.retryable = bool(retryable)
        self.details = details

    def to_error(self):
        """03 §1 的 error envelope。"""
        err = {
            "code": self.code,
            "message": str(self),
            "http_status": self.http_status,
            "retryable": self.retryable,
        }
        if self.details is not None:
            err["details"] = self.details
        return {"ok": False, "error": err}


def invalid(message, *, field=None, details=None):
    if field is not None:
        details = dict(details or {})
        details.setdefault("field", field)
    return ApiError(message, code="INVALID_REQUEST", http_status=400,
                    details=details)


def invalid_selector(message, *, selector=None):
    details = {"selector": selector} if selector is not None else None
    return ApiError(message, code="INVALID_SELECTOR", http_status=400,
                    details=details)


# ------------------------------------------------------------------ 基础检查

def _require_object(payload, *, where="请求体"):
    if not isinstance(payload, dict):
        raise invalid("%s必须是 JSON 对象" % where)


def _reject_unknown(payload, allowed, *, where="请求体"):
    unknown = sorted(k for k in payload if k not in allowed)
    if unknown:
        raise invalid("%s包含不支持的字段: %s" % (where, ", ".join(unknown)),
                      details={"unknown_fields": unknown})


def _strict_bool(payload, key, default, *, where="请求体"):
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise invalid("%s.%s 必须是 JSON bool（true/false），不接受 %r"
                      % (where, key, value), field=key)
    return value


def _strict_int(payload, key, default, *, lo=None, hi=None, where="请求体"):
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise invalid("%s.%s 必须是 JSON 整数，不接受 %r" % (where, key, value),
                      field=key)
    if lo is not None and value < lo:
        raise invalid("%s.%s 超出范围 [%d, %d]" % (where, key, lo, hi),
                      field=key)
    if hi is not None and value > hi:
        raise invalid("%s.%s 超出范围 [%d, %d]" % (where, key, lo, hi),
                      field=key)
    return value


def _string(payload, key, *, required=False, max_len=200, allow_empty=False,
            where="请求体"):
    if key not in payload:
        if required:
            raise invalid("%s缺少必需字段 %s" % (where, key), field=key)
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise invalid("%s.%s 必须是字符串" % (where, key), field=key)
    value = value.strip()
    if not value and not allow_empty:
        raise invalid("%s.%s 不能为空" % (where, key), field=key)
    if len(value) > max_len:
        raise invalid("%s.%s 过长（上限 %d 字符）" % (where, key, max_len),
                      field=key)
    return value


def _request_id(payload):
    rid = _string(payload, "request_id", max_len=MAX_REQUEST_ID_LEN)
    if rid is not None and any(ord(c) < 0x20 for c in rid):
        raise invalid("request_id 不能包含控制字符", field="request_id")
    return rid


def _decode_b64(value, *, label, max_bytes):
    if not isinstance(value, str) or not value:
        raise invalid("%s 必须是非空 base64 字符串" % label)
    if len(value) > MAX_IMAGE_B64_CHARS:
        raise ApiError("%s 过大（上限 %d 字符）" % (label, MAX_IMAGE_B64_CHARS),
                       code="PAYLOAD_TOO_LARGE", http_status=413)
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise invalid("%s 不是合法 base64" % label)
    if not data:
        raise invalid("%s 解码后为空" % label)
    if len(data) > max_bytes:
        raise ApiError("%s 解码后超过上限 %d MB"
                       % (label, max_bytes // (1 << 20)),
                       code="PAYLOAD_TOO_LARGE", http_status=413)
    return data


def _check_images(b64s):
    """逐张检查 base64 合法性与解码后总字节上限（不保留解码结果）。"""
    total = 0
    for i, b64 in enumerate(b64s):
        data = _decode_b64(b64, label="image_b64s[%d]" % i,
                           max_bytes=MAX_IMAGES_TOTAL_BYTES)
        total += len(data)
        if total > MAX_IMAGES_TOTAL_BYTES:
            raise ApiError("参考图解码后总字节超过上限 %d MB"
                           % (MAX_IMAGES_TOTAL_BYTES // (1 << 20)),
                           code="PAYLOAD_TOO_LARGE", http_status=413)
    return total


# ------------------------------------------------------------------ weights 选择器

def parse_selector(selector):
    """解析单个 selector，返回 (kind, head, a, b)。

    kind: ``range`` / ``exact`` / ``group``。
    语法：精确 ID ``dit.blocks.13``、组名 ``dit.blocks``、闭区间
    ``dit.blocks.12-15``。拒绝通配符、负数、反斜杠、空段、前导/尾随点。
    """
    if not isinstance(selector, str) or not selector:
        raise invalid_selector("selector 必须是非空字符串", selector=selector)
    if len(selector) > 256:
        raise invalid_selector("selector 过长（上限 256 字符）", selector=selector)
    if any(c in selector for c in "*?[]\\/%\x00"):
        raise invalid_selector("selector 不允许通配符/反斜杠/路径分隔符/百分号: %r"
                               % selector, selector=selector)
    if selector.startswith(".") or selector.endswith(".") or ".." in selector:
        raise invalid_selector("selector 不允许空段或 '..': %r" % selector,
                               selector=selector)

    head, sep, tail = selector.rpartition(".")
    if not sep or not head or not tail:
        raise invalid_selector("selector 必须形如 <prefix>.<id|a-b>: %r"
                               % selector, selector=selector)
    for token in head.split("."):
        if not token or not _SELECTOR_TOKEN_RE.match(token):
            raise invalid_selector("selector 前缀含非法字符: %r" % selector,
                                   selector=selector)

    m = _RANGE_TAIL_RE.match(tail)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            raise invalid_selector("区间上下界逆序（%d-%d）: %r"
                                   % (a, b, selector), selector=selector)
        if b - a + 1 > MAX_RANGE_SPAN:
            raise invalid_selector("区间展开长度超过上限 %d: %r"
                                   % (MAX_RANGE_SPAN, selector), selector=selector)
        return "range", head, a, b
    if "-" in tail:
        raise invalid_selector("selector 不支持负数或多余连字符: %r" % selector,
                               selector=selector)
    if tail.isdigit():
        return "exact", head + "." + tail, None, None
    if not _SELECTOR_TOKEN_RE.match(tail):
        raise invalid_selector("selector 含非法字符: %r" % selector,
                               selector=selector)
    return "group", selector, None, None


def expand_units(selectors, catalog):
    """把 selector 展开成按 catalog 原序去重的 unit ID 列表。

    ``catalog`` 需提供 ``unit_ids``（有序序列）与可选 ``groups``（组名→unit 列表）。
    catalog 未就绪（unit_ids 为空）时抛 ENV_NOT_READY，而不是返回空选择。
    """
    unit_ids = list(getattr(catalog, "unit_ids", None) or ())
    if not unit_ids:
        raise ApiError("unit catalog 尚未就绪，无法展开选择器",
                       code="ENV_NOT_READY", http_status=503)
    groups = dict(getattr(catalog, "groups", None) or {})
    unit_set = set(unit_ids)
    selected = set()

    for selector in selectors:
        kind, head, a, b = parse_selector(selector)
        if kind == "exact":
            if head not in unit_set:
                raise invalid_selector("未知 unit: %s" % head, selector=selector)
            selected.add(head)
        elif kind == "group":
            matched = groups.get(selector)
            if matched is None:
                raise invalid_selector("未知组名: %s" % selector, selector=selector)
            matched = list(matched)
            unknown = [u for u in matched if u not in unit_set]
            if unknown:
                raise invalid_selector("组 %s 含 catalog 外 unit: %s"
                                       % (selector, unknown[:3]), selector=selector)
            selected.update(matched)
        else:  # range
            prefix = head + "."
            hits = []
            for unit in unit_ids:
                if not unit.startswith(prefix):
                    continue
                rest = unit[len(prefix):]
                if not rest.isdigit():
                    continue
                value = int(rest)
                if a <= value <= b:
                    hits.append(unit)
            if not hits:
                raise invalid_selector("区间未命中任何 unit: %s" % selector,
                                       selector=selector)
            selected.update(hits)

    return [u for u in unit_ids if u in selected]


# ------------------------------------------------------------------ weights 请求

_RECIPE_ALLOWED = {"kind", "path", "proj", "mmproj", "options"}


def _validate_recipe(recipe, *, where):
    if not isinstance(recipe, dict):
        raise invalid("%s 必须是 JSON 对象" % where)
    _reject_unknown(recipe, _RECIPE_ALLOWED, where=where)
    kind = _string(recipe, "kind", required=True, max_len=64, where=where)
    if kind not in ("4b", "gguf", "checkpoint"):
        raise invalid("%s.kind 不支持: %s" % (where, kind), field="kind")
    for key, max_len in (("path", 1024), ("proj", 200), ("mmproj", 1024)):
        value = _string(recipe, key, max_len=max_len, where=where)
        if value is not None and "\x00" in value:
            raise invalid("%s.%s 不能包含 NUL" % (where, key), field=key)
    options = recipe.get("options")
    if options is not None and not isinstance(options, dict):
        raise invalid("%s.options 必须是 JSON 对象" % where, field="options")
    return {"kind": kind, "path": recipe.get("path"),
            "proj": recipe.get("proj"), "mmproj": recipe.get("mmproj"),
            "options": dict(options or {})}


def validate_weights_request(payload, action):
    """校验四种 weights 动作共用的请求结构（03 §3）。

    返回 cleaned dict；selector 展开（需要 catalog）由 :func:`expand_units`
    在动作执行前完成，这里只做语法与结构检查。
    """
    if action not in WEIGHTS_ACTIONS:
        raise ApiError("未知 weights 动作: %r" % (action,),
                       code="INVALID_REQUEST", http_status=400)
    _require_object(payload)
    _reject_unknown(payload, {"model", "models", "mode", "units", "dry_run",
                              "request_id", "recipe", "recipes"})

    has_model, has_models = "model" in payload, "models" in payload
    if has_model == has_models:
        raise invalid("model 与 models 必须且只能给一个")
    if has_model:
        model = _string(payload, "model", required=True, max_len=64)
        if model not in MODEL_IDS:
            raise ApiError("未知 model: %s" % model, code="MODEL_NOT_FOUND",
                           http_status=404)
        models = [model]
    else:
        raw = payload["models"]
        if not isinstance(raw, list) or not raw:
            raise invalid("models 必须是非空字符串列表")
        if len(raw) > len(MODEL_IDS):
            raise invalid("models 成员数上限 %d" % len(MODEL_IDS))
        models = []
        for i, item in enumerate(raw):
            if not isinstance(item, str):
                raise invalid("models[%d] 必须是字符串" % i)
            if item not in MODEL_IDS:
                raise ApiError("未知 model: %s" % item, code="MODEL_NOT_FOUND",
                               http_status=404)
            if item not in models:
                models.append(item)
        if not models:
            raise invalid("models 去重后为空")

    mode = payload.get("mode", "all")
    if not isinstance(mode, str) or mode not in SELECTOR_MODES:
        raise invalid("mode 只能是 all 或 units", field="mode")

    units_in = payload.get("units")
    if mode == "all":
        if units_in is not None:
            raise invalid("mode=all 时不接受 units 字段（防止歧义）", field="units")
        units = []
    else:
        if len(models) > 1:
            raise invalid("多模型首版只支持 mode=all")
        if not isinstance(units_in, list) or not units_in:
            raise invalid("mode=units 必须提供非空 units 列表", field="units")
        if len(units_in) > MAX_SELECTORS:
            raise invalid("units 条数超过上限 %d" % MAX_SELECTORS, field="units")
        units = []
        for i, selector in enumerate(units_in):
            parse_selector(selector)  # 语法错误直接抛出
            if selector not in units:
                units.append(selector)

    dry_run = _strict_bool(payload, "dry_run", False)
    request_id = _request_id(payload)

    recipe = payload.get("recipe")
    recipes = payload.get("recipes")
    if action != "disk2ram":
        if recipe is not None or recipes is not None:
            raise invalid("recipe/recipes 仅 disk2ram 支持")
    if recipe is not None and recipes is not None:
        raise invalid("recipe 与 recipes 不能同时提供")
    recipe_out = None
    recipes_out = None
    if recipe is not None:
        if len(models) != 1:
            raise invalid("多模型请使用 recipes: {model_id: {...}}")
        recipe_out = _validate_recipe(recipe, where="recipe")
    if recipes is not None:
        if len(models) == 1:
            raise invalid("单模型请使用 recipe，不用 recipes")
        if not isinstance(recipes, dict) or not recipes:
            raise invalid("recipes 必须是非空 JSON 对象")
        unknown = sorted(m for m in recipes if m not in models)
        if unknown:
            raise invalid("recipes 含未在 models 中声明的模型: %s"
                          % ", ".join(unknown))
        recipes_out = {m: _validate_recipe(v, where="recipes.%s" % m)
                       for m, v in recipes.items()}

    return {
        "action": action,
        "models": models,
        "mode": mode,
        "units": units,
        "dry_run": dry_run,
        "request_id": request_id,
        "recipe": recipe_out,
        "recipes": recipes_out,
    }


# ------------------------------------------------------------------ infer.encode

_ENCODE_ALLOWED = {"prompt", "image_b64s", "mode", "width", "height",
                   "ref_image_size", "request_id"}


def validate_encode_request(payload, *, allow_text_only=False):
    """校验 POST /api/infer/encode（03 §6）。

    ``allow_text_only`` 只能由已公布 text_only capability 的 TE adapter 置 True；
    默认 0 图被拒绝，不虚构能力。
    """
    _require_object(payload)
    _reject_unknown(payload, _ENCODE_ALLOWED)

    prompt = _string(payload, "prompt", required=True,
                     max_len=MAX_PROMPT_CHARS)

    mode = payload.get("mode", DEFAULTS["mode"])
    if not isinstance(mode, str) or mode not in ("ref2va", "fl2va"):
        raise invalid("mode 只能是 ref2va 或 fl2va", field="mode")

    ref_size = payload.get("ref_image_size", DEFAULTS["ref_image_size"])
    if not isinstance(ref_size, str) or ref_size not in ("match", "max"):
        raise invalid("ref_image_size 只能是 match 或 max", field="ref_image_size")

    min_images = 0 if allow_text_only else 1
    b64s, err = _image_payloads(payload, mode, min_images, 9)
    if err:
        raise invalid(err, field="image_b64s")
    _check_images(b64s)

    width = _strict_int(payload, "width", DEFAULTS["width"], lo=256, hi=1600)
    height = _strict_int(payload, "height", DEFAULTS["height"], lo=256, hi=1600)
    if width % 32 or height % 32:
        raise invalid("width / height 必须是 32 的倍数")

    return {
        "prompt": prompt,
        "image_b64s": list(b64s),
        "mode": mode,
        "width": width,
        "height": height,
        "ref_image_size": ref_size,
        "request_id": _request_id(payload),
    }


# ------------------------------------------------------------------ infer.sample

_SAMPLE_ALLOWED = {"cond_artifact", "cond_file", "cond_b64", "image_b64s",
                   "mode", "ref_image_size", "width", "height", "length",
                   "steps", "sampler", "scheduler", "seed", "refine_steps",
                   "refine_denoise", "sigmas", "dump_tail_frames",
                   "profile_attn", "request_id"}
_SAMPLE_STRICT_INTS = (("width", 256, 1600), ("height", 256, 1600),
                       ("length", 16, 400), ("steps", 1, 50),
                       ("seed", -1, 2 ** 63 - 1), ("refine_steps", 0, 20),
                       ("dump_tail_frames", 0, 9))


def _cond_source(payload, prefix):
    """cond/latent 来源三选一：<prefix>_artifact / <prefix>_file / <prefix>_b64。"""
    artifact = payload.get("%s_artifact" % prefix)
    file_ref = payload.get("%s_file" % prefix)
    b64 = payload.get("%s_b64" % prefix) if prefix == "cond" else None
    given = [name for name, value in (("artifact", artifact), ("file", file_ref),
                                      ("b64", b64)) if value is not None]
    if len(given) != 1:
        raise invalid("%s 来源必须且只能给一个：%s_artifact / %s_file%s"
                      % (prefix, prefix, prefix,
                         " / %s_b64" % prefix if prefix == "cond" else ""))
    if artifact is not None:
        if not isinstance(artifact, str) or not ARTIFACT_ID_RE.match(artifact):
            raise invalid("%s_artifact 必须形如 %s_<32位hex>"
                          % (prefix, "cond" if prefix == "cond" else "latent"),
                          field="%s_artifact" % prefix)
        if not artifact.startswith(prefix + "_"):
            raise invalid("%s_artifact 前缀不正确" % prefix,
                          field="%s_artifact" % prefix)
        return {"kind": "artifact", "id": artifact}
    if file_ref is not None:
        if not isinstance(file_ref, str) or not file_ref.strip():
            raise invalid("%s_file 必须是非空字符串" % prefix,
                          field="%s_file" % prefix)
        file_ref = file_ref.strip()
        if len(file_ref) > MAX_FILE_REF_CHARS:
            raise invalid("%s_file 路径过长" % prefix, field="%s_file" % prefix)
        if "\x00" in file_ref:
            raise invalid("%s_file 不能包含 NUL" % prefix,
                          field="%s_file" % prefix)
        return {"kind": "file", "path": file_ref}
    data = _decode_b64(b64, label="cond_b64", max_bytes=MAX_COND_BYTES)
    return {"kind": "b64", "bytes": data}


def validate_sample_request(payload):
    """校验 POST /api/infer/sample（03 §6）。"""
    _require_object(payload)
    _reject_unknown(payload, _SAMPLE_ALLOWED)

    mode = payload.get("mode", DEFAULTS["mode"])
    if not isinstance(mode, str) or mode not in ("ref2va", "fl2va"):
        raise invalid("mode 只能是 ref2va 或 fl2va", field="mode")
    if "profile_attn" in payload:
        _strict_bool(payload, "profile_attn", False)
    for key, lo, hi in _SAMPLE_STRICT_INTS:
        _strict_int(payload, key, DEFAULTS.get(key), lo=lo, hi=hi)
    if "refine_denoise" in payload and (
            isinstance(payload["refine_denoise"], bool)
            or not isinstance(payload["refine_denoise"], (int, float))):
        raise invalid("refine_denoise 必须是数字", field="refine_denoise")
    if "sigmas" in payload and payload["sigmas"] is not None and (
            not isinstance(payload["sigmas"], str)):
        raise invalid("sigmas 必须是逗号分隔的字符串", field="sigmas")

    b64s, err = _image_payloads(payload, mode, 0, 9)
    if err:
        raise invalid(err, field="image_b64s")
    _check_images(b64s)

    source = _cond_source(payload, "cond")

    params = {}
    err = _fill_sampling_params(payload, params)
    if err:
        raise invalid(err)
    params["image_b64s"] = list(b64s)
    params["image_b64"] = b64s[0] if b64s else None
    params["mode"] = mode

    return {"cond_source": source, "params": params,
            "request_id": _request_id(payload)}


# ------------------------------------------------------------------ infer.decode

_DECODE_ALLOWED = {"latent_artifact", "latent_file", "write_mp4", "write_wav",
                   "include_audio", "fps", "dump_tail_frames", "request_id"}


def validate_decode_request(payload):
    """校验 POST /api/infer/decode（03 §6 / 04 §6）。

    带音频的 MP4 首版只允许 24fps；其它 fps 在解码前拒绝，而不是输出可能
    音画错位的成片。
    """
    _require_object(payload)
    _reject_unknown(payload, _DECODE_ALLOWED)

    source = _cond_source(payload, "latent")
    write_mp4 = _strict_bool(payload, "write_mp4", True)
    write_wav = _strict_bool(payload, "write_wav", False)
    include_audio = _strict_bool(payload, "include_audio", True)
    fps = _strict_int(payload, "fps", DEFAULTS["fps"], lo=1, hi=60)
    dump_tail_frames = _strict_int(payload, "dump_tail_frames", 0, lo=0, hi=9)

    if not (write_mp4 or write_wav or dump_tail_frames > 0):
        raise invalid("decode 至少要选择一种输出（write_mp4/write_wav/dump_tail_frames）")
    if write_mp4 and include_audio and fps != 24:
        raise ApiError(
            "带音频的 MP4 首版只支持 24fps（请求 fps=%d）；请改用 fps=24、"
            "去掉音轨或单独输出 WAV" % fps,
            code="UNSUPPORTED_FPS", http_status=422)

    return {
        "latent_source": source,
        "write_mp4": write_mp4,
        "write_wav": write_wav,
        "include_audio": include_audio,
        "fps": fps,
        "dump_tail_frames": dump_tail_frames,
        "request_id": _request_id(payload),
    }


# ------------------------------------------------------------------ artifact 引用

def validate_artifact_id(artifact_id, *, kind=None):
    """校验 artifact ID 语法；``kind`` 给定时同时校验前缀。"""
    if not isinstance(artifact_id, str) or not ARTIFACT_ID_RE.match(artifact_id):
        raise invalid("artifact ID 必须形如 cond_<32位hex> 或 latent_<32位hex>")
    if kind is not None and not artifact_id.startswith(kind + "_"):
        raise invalid("artifact ID 前缀必须是 %s_" % kind)
    return artifact_id
