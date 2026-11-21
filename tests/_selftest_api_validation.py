"""自检：dit_svc/api_validation.py —— 纯标准库，不需要 torch / GPU。

覆盖：
- 四种 weights 动作共用同一 selectors/recipe 契约（模型互斥、mode/units、range 展开）
- 严格 JSON bool/int、未知字段拒绝、request_id
- infer.encode / infer.sample / infer.decode 的来源互斥、fps 策略、payload 上限
- artifact ID 语法

注意：image_b64s 只校验 base64 与解码字节数，不校验 PNG/JPEG 结构（那属于
infer 层的解码检查），所以这里用任意短字节的 base64 作夹具。
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc.api_validation import (  # noqa: E402
    ApiError, expand_units, parse_selector, validate_artifact_id,
    validate_decode_request, validate_encode_request, validate_sample_request,
    validate_weights_request,
)

fails = []
IMG_B64 = base64.b64encode(b"faux-image-bytes").decode("ascii")
COND_B64 = base64.b64encode(b"faux-cond-bytes").decode("ascii")
AID = "cond_" + "0" * 32
LID = "latent_" + "1" * 32


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("返回 False")
        print("  OK   %-52s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-52s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


def expect_error(name, fn, *, code=None, http=None, contains=None):
    def run():
        try:
            fn()
        except ApiError as e:
            if code is not None and e.code != code:
                raise AssertionError("code=%s 期望 %s" % (e.code, code))
            if http is not None and e.http_status != http:
                raise AssertionError("http_status=%d 期望 %d" % (e.http_status, http))
            if contains is not None and contains not in str(e):
                raise AssertionError("message %r 不含 %r" % (str(e), contains))
            env = e.to_error()
            assert env["ok"] is False and env["error"]["code"] == e.code
            return "%s/%d: %s" % (e.code, e.http_status, str(e)[:40])
        raise AssertionError("没有抛出 ApiError")
    check(name, run)


class FakeUnitCatalog(object):
    """模拟真实 ``UnitCatalog``：``unit_ids()`` 是**方法**，groups 是属性。

    主代理接入时发现的正是这点：真实实现只有 ``unit_ids(self)`` 方法，旧
    ``list(catalog.unit_ids)`` 会拿到 bound method 并 TypeError。
    """

    def __init__(self):
        self._ids = (["dit.embed"] +
                     ["dit.blocks.%d" % i for i in range(4)] +
                     ["dit.refiner", "dit.modules.rope",
                      "vae.video.encoder"])
        self.groups = {"dit.blocks": ["dit.blocks.%d" % i for i in range(4)]}

    def unit_ids(self):
        return list(self._ids)


class StubCatalogAttr(object):
    """早期 stub 形态：unit_ids 是属性。两种形态都必须支持。"""

    def __init__(self):
        self.unit_ids = ["a.b.one", "a.b.two"]
        self.groups = {"a.b": ["a.b.one", "a.b.two"]}


class FakeCatalog0(object):
    def unit_ids(self):
        return []

    groups = {}


class GappyCatalog(object):
    """索引不连续（缺 2），用于验证 range 必须逐索引全覆盖。"""

    def __init__(self):
        self._ids = ["dit.blocks.0", "dit.blocks.1", "dit.blocks.3"]
        self.groups = {}

    def unit_ids(self):
        return list(self._ids)


print("== selector 解析 ==")
check("exact", lambda: parse_selector("dit.blocks.13")[0] == "exact")
check("group", lambda: parse_selector("dit.blocks")[0] == "group")
check("range", lambda: parse_selector("dit.blocks.12-15")[2:] == (12, 15))
expect_error("拒绝通配符", lambda: parse_selector("dit.blocks.*"),
             code="INVALID_SELECTOR", http=400)
expect_error("拒绝反斜杠", lambda: parse_selector("dit\\blocks"), code="INVALID_SELECTOR")
expect_error("拒绝 ..", lambda: parse_selector("dit..blocks"), code="INVALID_SELECTOR")
expect_error("拒绝逆区间", lambda: parse_selector("dit.blocks.15-12"),
             code="INVALID_SELECTOR", contains="逆序")
expect_error("拒绝超大区间", lambda: parse_selector("dit.blocks.0-99999"),
             code="INVALID_SELECTOR", contains="上限")
expect_error("拒绝空段", lambda: parse_selector("dit..1"), code="INVALID_SELECTOR")
expect_error("拒绝负数", lambda: parse_selector("dit.blocks.-3"), code="INVALID_SELECTOR")

print("== selector 展开 ==")
cat = FakeUnitCatalog()
check("exact 展开",
      lambda: expand_units(["dit.blocks.2"], cat) == ["dit.blocks.2"])
check("range 按 catalog 原序",
      lambda: expand_units(["dit.blocks.2-3"], cat) ==
      ["dit.blocks.2", "dit.blocks.3"])
check("组名展开",
      lambda: len(expand_units(["dit.blocks"], cat)) == 4)
check("重叠选择去重且保序",
      lambda: expand_units(["dit.blocks.0-1", "dit.blocks.1", "dit.blocks"],
                           cat) == ["dit.blocks.0", "dit.blocks.1",
                                    "dit.blocks.2", "dit.blocks.3"])
expect_error("未知 unit", lambda: expand_units(["dit.blocks.99"], cat),
             code="INVALID_SELECTOR", contains="未知 unit")
expect_error("未知组", lambda: expand_units(["dit.nope"], cat),
             code="INVALID_SELECTOR", contains="未知组名")
expect_error("空 catalog", lambda: expand_units(["dit.blocks.0"], FakeCatalog0()),
             code="ENV_NOT_READY", http=503)

print("== catalog 接口：unit_ids() 方法 / 属性 stub ==")
check("方法形态 unit_ids() 可用",
      lambda: expand_units(["dit.blocks.1"], cat) == ["dit.blocks.1"])
_check_attr_cat = StubCatalogAttr()
check("属性形态 stub 也可用",
      lambda: expand_units(["a.b.two"], _check_attr_cat) == ["a.b.two"])
check("属性形态组名可用",
      lambda: expand_units(["a.b"], _check_attr_cat) == ["a.b.one", "a.b.two"])


class BadCatalogShape(object):
    def unit_ids(self):
        return 42

    groups = {}


expect_error("unit_ids 形状不对报 ENV_NOT_READY",
             lambda: expand_units(["dit.blocks.0"], BadCatalogShape()),
             code="ENV_NOT_READY", http=503)

print("== 非数字精确 ID 不得被当成组名 ==")
for exact_id in ("dit.refiner", "dit.modules.rope", "vae.video.encoder"):
    check("精确选择 %s" % exact_id,
          lambda i=exact_id: expand_units([i], cat) == [i])
    check("groups 不含 %s 也仍精确" % exact_id,
          lambda i=exact_id: i not in cat.groups)
check("精确 ID 与组名混用保序",
      lambda: expand_units(["dit.refiner", "dit.blocks"], cat) ==
      ["dit.blocks.0", "dit.blocks.1", "dit.blocks.2", "dit.blocks.3",
       "dit.refiner"])

print("== range 必须逐索引全覆盖（不静默 clamp）==")
gap = GappyCatalog()
expect_error("缺中间索引的 range 400",
             lambda: expand_units(["dit.blocks.0-3"], gap),
             code="INVALID_SELECTOR", contains="越界")
check("恰好全覆盖的 range 通过",
      lambda: expand_units(["dit.blocks.0-1"], gap) ==
      ["dit.blocks.0", "dit.blocks.1"])
expect_error("range 上界越界 400",
             lambda: expand_units(["dit.blocks.0-4"], gap),
             code="INVALID_SELECTOR", contains="越界")
expect_error("0-99 对 4 项 catalog 必须 400（不能只取 0-3）",
             lambda: expand_units(["dit.blocks.0-99"], cat),
             code="INVALID_SELECTOR", contains="越界")


print("== weights 请求（四动作同一契约）==")
for action in ("disk2ram", "ram2vram", "vram2ram", "purgefromram"):
    check("%s: 单模型 units" % action,
          lambda a=action: validate_weights_request(
              {"model": "dit", "mode": "units", "units": ["dit.blocks.1-2"]}, a)["units"] ==
          ["dit.blocks.1-2"])
    check("%s: all 默认" % action,
          lambda a=action: validate_weights_request({"model": "dit"}, a)["mode"] == "all")
    check("%s: 多模型 all" % action,
          lambda a=action: validate_weights_request(
              {"models": ["dit", "vae.video"], "mode": "all"}, a)["models"] ==
          ["dit", "vae.video"])

W = lambda p, a="disk2ram": validate_weights_request(p, a)  # noqa: E731
expect_error("model 与 models 互斥",
             lambda: W({"model": "dit", "models": ["te"]}), contains="只能给一个")
expect_error("两者都不给",
             lambda: W({"mode": "all"}), contains="只能给一个")
expect_error("未知 model",
             lambda: W({"model": "nope"}), code="MODEL_NOT_FOUND", http=404)
expect_error("未知 models 成员",
             lambda: W({"models": ["dit", "nope"]}), code="MODEL_NOT_FOUND", http=404)
expect_error("all 不接受 units",
             lambda: W({"model": "dit", "mode": "all", "units": ["dit.blocks.0"]}),
             contains="不接受 units")
expect_error("units 空列表",
             lambda: W({"model": "dit", "mode": "units", "units": []}),
             contains="非空")
expect_error("多模型 units 拒绝",
             lambda: W({"models": ["dit", "te"], "mode": "units",
                        "units": ["dit.blocks.0"]}), contains="只支持 mode=all")
expect_error("mode 非枚举",
             lambda: W({"model": "dit", "mode": "some"}), contains="mode")
expect_error("dry_run 只接受 bool",
             lambda: W({"model": "dit", "dry_run": 1}), contains="JSON bool")
expect_error("未知字段拒绝",
             lambda: W({"model": "dit", "keep": "cpu"}), contains="keep")
expect_error("非 disk2ram 拒绝 recipe",
             lambda: W({"model": "te", "recipe": {"kind": "4b"}}, "ram2vram"),
             contains="仅 disk2ram")

check("disk2ram 单模型 recipe",
      lambda: W({"model": "te", "recipe": {"kind": "4b", "path": "D:\\w.safetensors",
                                           "proj": "clip_proj"}})["recipe"]["kind"] == "4b")
check("disk2ram 多模型 recipes",
      lambda: W({"models": ["dit", "te"],
                 "recipes": {"dit": {"kind": "checkpoint", "path": "D:\\d.safetensors"},
                             "te": {"kind": "4b"}}})["recipes"]["te"]["kind"] == "4b")
expect_error("单模型不用 recipes",
             lambda: W({"model": "te", "recipes": {"te": {"kind": "4b"}}}),
             contains="recipes")
expect_error("多模型不用单 recipe",
             lambda: W({"models": ["dit", "te"], "recipe": {"kind": "4b"}}),
             contains="recipes")
expect_error("recipes 含未在 models 中声明的模型",
             lambda: W({"models": ["dit", "te"],
                        "recipes": {"dit": {"kind": "4b"}, "nope": {"kind": "4b"}}}),
             contains="未在 models")
expect_error("recipe.kind 非法",
             lambda: W({"model": "te", "recipe": {"kind": "fp8"}}), contains="kind")
expect_error("recipe 未知字段",
             lambda: W({"model": "te", "recipe": {"kind": "4b", "device": "cuda"}}),
             contains="device")
expect_error("request_id 控制字符",
             lambda: W({"model": "dit", "request_id": "a\x01b"}), contains="控制字符")

print("== disk2ram 顶层 options ==")
check("disk2ram 顶层 options {} 通过",
      lambda: W({"model": "te", "mode": "all", "options": {},
                 "recipe": {"kind": "4b"}})["options"] == {})
check("顶层 options 保留内容（优先级留给主代理）",
      lambda: W({"model": "dit", "options": {"mlp_chunk": 2048}})["options"] ==
      {"mlp_chunk": 2048})
check("顶层 options 缺省为 None",
      lambda: W({"model": "dit"})["options"] is None)
for action in ("ram2vram", "vram2ram", "purgefromram"):
    expect_error("%s 拒绝顶层 options" % action,
                 lambda a=action: W({"model": "dit", "options": {}}, a),
                 contains="仅 disk2ram")
expect_error("顶层 options 非对象",
             lambda: W({"model": "dit", "options": []}), contains="JSON 对象")

print("== recipe.kind：非 TE 可缺省 checkpoint，TE 可继承 defaults ==")
info_dit = W({"model": "dit", "recipe": {"path": "D:\\d.safetensors"}})
check("非 TE 缺 kind 通过且不伪造 kind",
      lambda: info_dit["recipe"]["kind"] is None and
      info_dit["recipe"]["path"] == "D:\\d.safetensors")
info_te = W({"model": "te", "recipe": {"path": "D:\\te.safetensors"}})
check("TE 缺 kind 通过（继承 defaults）",
      lambda: info_te["recipe"]["kind"] is None)
check("TE 显式 4b 仍保留",
      lambda: W({"model": "te", "recipe": {"kind": "4b"}})["recipe"]["kind"] == "4b")
check("多模型 recipes 各自可缺 kind",
      lambda: W({"models": ["dit", "te"],
                 "recipes": {"dit": {"path": "D:\\d.safetensors"},
                             "te": {}}})["recipes"]["dit"]["kind"] is None)
expect_error("显式 kind 非枚举仍拒绝（非 TE）",
             lambda: W({"model": "dit", "recipe": {"kind": "fp8"}}), contains="kind")
expect_error("显式 kind 非枚举仍拒绝（TE）",
             lambda: W({"model": "te", "recipe": {"kind": "fp8"}}), contains="kind")
expect_error("recipe.options 非对象",
             lambda: W({"model": "dit", "recipe": {"options": "x"}}),
             contains="JSON 对象")

print("== infer.encode ==")
E = validate_encode_request
check("encode 基本",
      lambda: E({"prompt": "a cat", "image_b64s": [IMG_B64]})["mode"] == "ref2va")
check("encode fl2va 单图",
      lambda: E({"prompt": "a", "image_b64s": [IMG_B64], "mode": "fl2va"})["mode"] == "fl2va")
expect_error("encode 0 图默认拒绝",
             lambda: E({"prompt": "a", "image_b64s": []}), contains="1–9")
expect_error("encode 0 图在无 text_only 能力时仍拒绝",
             lambda: E({"prompt": "a"}, allow_text_only=False), contains="1–9")
check("encode allow_text_only 允许 0 图",
      lambda: E({"prompt": "a"}, allow_text_only=True)["image_b64s"] == [])
expect_error("fl2va 多图",
             lambda: E({"prompt": "a", "mode": "fl2va",
                        "image_b64s": [IMG_B64, IMG_B64]}), contains="单图")
expect_error("encode 非法 base64",
             lambda: E({"prompt": "a", "image_b64s": ["!!notb64!!"]}),
             contains="base64")
expect_error("encode 空 prompt", lambda: E({"prompt": "  ", "image_b64s": [IMG_B64]}),
             contains="prompt")
expect_error("encode width 字符串",
             lambda: E({"prompt": "a", "image_b64s": [IMG_B64], "width": "640"}),
             contains="JSON 整数")
expect_error("encode width 非 32 倍数",
             lambda: E({"prompt": "a", "image_b64s": [IMG_B64], "width": 650}),
             contains="32 的倍数")
expect_error("encode 未知字段",
             lambda: E({"prompt": "a", "image_b64s": [IMG_B64], "save_cond": False}),
             contains="save_cond")

print("== infer.sample ==")
S = validate_sample_request
check("sample 用 cond_artifact",
      lambda: S({"cond_artifact": AID})["cond_source"]["id"] == AID)
check("sample 参数继承 common 规则",
      lambda: S({"cond_file": "cond_x.safetensors"})["params"]["steps"] == 4)
expect_error("sample cond 来源全缺", lambda: S({}), contains="只能给一个")
expect_error("sample cond 来源重复",
             lambda: S({"cond_artifact": AID, "cond_b64": COND_B64}),
             contains="只能给一个")
expect_error("sample artifact ID 非法",
             lambda: S({"cond_artifact": "cond_123"}), contains="32位hex")
expect_error("sample latent_artifact 冒充 cond",
             lambda: S({"cond_artifact": LID}), contains="前缀")
expect_error("sample cond_file 含 NUL",
             lambda: S({"cond_file": "a\x00b"}), contains="NUL")
check("sample cond_file 语法层接受绝对路径（安全解析在 ArtifactStore）",
      lambda: S({"cond_file": "C:\\out\\stage\\x.safetensors"})["cond_source"]["kind"] == "file")
expect_error("sample steps 字符串",
             lambda: S({"cond_artifact": AID, "steps": "4"}), contains="JSON 整数")
expect_error("sample sampler 非法",
             lambda: S({"cond_artifact": AID, "sampler": "nope"}), contains="sampler")
expect_error("sample sigmas 非字符串",
             lambda: S({"cond_artifact": AID, "sigmas": [1.0, 0.0]}), contains="逗号分隔")
expect_error("sample profile_attn 非 bool",
             lambda: S({"cond_artifact": AID, "profile_attn": "true"}),
             contains="JSON bool")

print("== infer.decode ==")
D = validate_decode_request
check("decode 默认", lambda: D({"latent_artifact": LID})["fps"] == 24)
check("decode 无 mp4 时可改 fps",
      lambda: D({"latent_artifact": LID, "write_mp4": False,
                 "write_wav": True, "fps": 30})["fps"] == 30)
expect_error("decode 改 fps 且带音频 mp4",
             lambda: D({"latent_artifact": LID, "fps": 30}),
             code="UNSUPPORTED_FPS", http=422)
check("decode 改 fps 且不含音轨",
      lambda: D({"latent_artifact": LID, "fps": 30, "include_audio": False})["fps"] == 30)
expect_error("decode 没有输出",
             lambda: D({"latent_artifact": LID, "write_mp4": False,
                        "write_wav": False}), contains="至少要选择")
expect_error("decode latent 来源缺",
             lambda: D({"write_mp4": True}), contains="只能给一个")
expect_error("decode 三种来源重复",
             lambda: D({"latent_artifact": LID, "latent_file": "x.safetensors"}),
             contains="只能给一个")
expect_error("decode 未知字段",
             lambda: D({"latent_artifact": LID, "save_latent": True}),
             contains="save_latent")

print("== artifact ID ==")
check("cond id", lambda: validate_artifact_id(AID, kind="cond") == AID)
expect_error("id 大写 hex", lambda: validate_artifact_id("cond_" + "A" * 32),
             contains="32位hex")
expect_error("id 前缀不符", lambda: validate_artifact_id(AID, kind="latent"),
             contains="前缀")
expect_error("id 路径注入", lambda: validate_artifact_id("cond_../../x"),
             contains="32位hex")

print()
if fails:
    print("FAILED %d: %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL OK")
