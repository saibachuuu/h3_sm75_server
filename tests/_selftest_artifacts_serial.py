"""自检：dit_svc/artifacts.py 序列化层（torch + safetensors，CPU 可跑）。

用真实 Comfy NestedTensor 语义做容器夹具：
    class NestedTensor:
        def __init__(self, tensors): self.tensors = list(tensors); self.is_nested = True
        def unbind(self): return self.tensors
来源：remote comfy/nested_tensor.py（本地副本
explicit_api_sources_20261121/comfy_extras/nested_tensor.py）。若该副本存在就
直接 import 它，保证测的是真类；否则用等价的最小夹具，并在输出里说明。

video [B,24,T,H//16,W//16]，audio [B,32,2,T] 来自
comfy_extras/nodes_minimax_h3.py 的 `_av_latent`。

运行：<isolated env python> tests/_selftest_artifacts_serial.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from dit_svc.artifacts import (  # noqa: E402
    ArtifactError, ArtifactStore, FORMAT_AVLATENT, FORMAT_COND,
    load_av_latent, load_cond, probe_av_latent, probe_tensor_chunked,
    save_av_latent, save_cond, split_av_latent,
)

fails = []
TMP = tempfile.mkdtemp(prefix="h3serial_")
SRC = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "opencode",
                   "explicit_api_sources_20261121", "comfy_extras",
                   "nested_tensor.py")
ALT_SRC = r"C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_sources_20261121\comfy_extras\nested_tensor.py"


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("返回 False")
        print("  OK   %-52s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-52s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


def expect(name, fn, *, code=None, contains=None):
    def run():
        try:
            fn()
        except ArtifactError as e:
            if code is not None and e.code != code:
                raise AssertionError("code=%s 期望 %s" % (e.code, code))
            if contains is not None and contains not in str(e):
                raise AssertionError("message %r 不含 %r" % (str(e), contains))
            return "%s: %s" % (e.code, str(e)[:46])
        raise AssertionError("没有抛出 ArtifactError")
    check(name, run)


# ---------------------------------------------------------------- 容器夹具

def _load_nested_class():
    for path in (SRC, ALT_SRC):
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_real_nested", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("  [fixture] 使用真实 NestedTensor: %s" % path)
            return mod.NestedTensor, "real"

    class NestedTensor(object):
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        def unbind(self):
            return self.tensors

    print("  [fixture] 未找到真实源码副本，使用等价最小夹具")
    return NestedTensor, "fixture"


NestedTensor, FIXTURE_KIND = _load_nested_class()

# 让 store 重建时能找到这个类（真实运行环境里是 comfy.nested_tensor）
import types  # noqa: E402

_mod = types.ModuleType("comfy_extras.nested_tensor")
_mod.NestedTensor = NestedTensor
sys.modules.setdefault("comfy_extras", types.ModuleType("comfy_extras"))
sys.modules["comfy_extras.nested_tensor"] = _mod

# ---------------------------------------------------------------- 夹具


def make_av(batch=1, t=3, h=4, w=5, at=6, vdtype=torch.float32,
            adtype=torch.bfloat16):
    video = torch.randn(batch, 24, t, h, w, dtype=vdtype)
    audio = torch.randn(batch, 32, 2, at, dtype=adtype)
    return {"samples": NestedTensor((video, audio))}


def base_meta(**over):
    md = {"format": FORMAT_AVLATENT, "width": 640, "height": 960,
          "requested_length": 124, "effective_latent_frames": 3,
          "native_fps": 24, "mode": "ref2va", "checkpoint_fingerprint": "abc"}
    md.update(over)
    return md


print("== 拆分：shape 判别与顺序 ==")


def _split_shapes():
    lat = make_av()
    tensors, extra = split_av_latent(lat)
    assert tuple(tensors["video"].shape) == (1, 24, 3, 4, 5)
    assert tuple(tensors["audio"].shape) == (1, 32, 2, 6)
    assert extra["container_order"] == "video,audio"
    assert str(tensors["audio"].dtype) == "torch.bfloat16", tensors["audio"].dtype
    assert tensors["video"].device.type == "cpu"
    assert tensors["video"].is_contiguous()
    return extra["container_type"]


check("真实容器 shape/dtype/order", _split_shapes)


def _reversed_pair():
    lat = make_av()
    v, a = lat["samples"].tensors
    tensors, extra = split_av_latent({"samples": (a, v)})
    assert extra["container_order"] == "audio,video"
    assert tuple(tensors["video"].shape)[1] == 24
    return extra["container_order"]


check("二元 tuple 反序仍能按 shape 还原", _reversed_pair)


def _dict_container():
    tensors, extra = split_av_latent({"samples": {"video": torch.zeros(1, 24, 2, 2, 2),
                                                 "audio": torch.zeros(1, 32, 2, 4)}})
    assert extra["container_order"] == "video,audio"
    return "ok"


check("映射容器 {video,audio}", _dict_container)


def _noncontiguous():
    lat = make_av()
    v = lat["samples"].tensors[0].transpose(3, 4)
    assert not v.is_contiguous()
    tensors, _ = split_av_latent({"samples": NestedTensor((v, lat["samples"].tensors[1]))})
    assert tensors["video"].is_contiguous()
    assert tuple(tensors["video"].shape) == (1, 24, 3, 5, 4)
    return "ok"


check("非 contiguous 输入转 contiguous", _noncontiguous)


def _bad_shape():
    lat = {"samples": NestedTensor((torch.zeros(1, 7, 3, 4, 5), torch.zeros(1, 32, 2, 6)))}
    split_av_latent(lat)


expect("无法识别的 shape 拒绝", _bad_shape, code="INVALID_ARTIFACT",
       contains="既不是")
expect("未知 latent 字段拒绝（不静默丢弃）",
       lambda: split_av_latent(dict(make_av(), custom_thing=1)),
       code="INVALID_ARTIFACT", contains="custom_thing")
expect("缺 samples 拒绝",
       lambda: split_av_latent({"noise_mask": torch.zeros(1)}),
       code="INVALID_ARTIFACT", contains="samples")

print("== avlatent 落盘/读回 ==")
out_dir = os.path.join(TMP, "out")
store = ArtifactStore(out_dir, quota_bytes=1 << 30, disk_reserve_bytes=0)
lat = make_av()
info = save_av_latent(store, lat, base_meta(), filename="av1.safetensors")
check("save 返回 avlatent 摘要",
      lambda: info["format"] == FORMAT_AVLATENT and info["bytes"] > 0
      and info["kind"] == "latent")

back, md = load_av_latent(store, "av1.safetensors")
check("roundtrip 形状一致",
      lambda: tuple(back["samples"].tensors[0].shape) == (1, 24, 3, 4, 5)
      and tuple(back["samples"].tensors[1].shape) == (1, 32, 2, 6))
check("roundtrip dtype 保持",
      lambda: back["samples"].tensors[0].dtype is torch.float32
      and back["samples"].tensors[1].dtype is torch.bfloat16)
check("roundtrip 容器类型还原",
      lambda: isinstance(back["samples"], NestedTensor)
      and back["samples"].is_nested is True)
check("metadata requested_length/native_fps",
      lambda: md["requested_length"] == 124 and md["native_fps"] == 24.0)
check("metadata 记录 shapes/dtypes",
      lambda: md["video_shape"] == "[1, 24, 3, 4, 5]" and "float32" in md["dtypes"])
check("数值完全一致",
      lambda: torch.equal(back["samples"].tensors[0], lat["samples"].tensors[0])
      and torch.equal(back["samples"].tensors[1], lat["samples"].tensors[1]))
check("不残留临时文件",
      lambda: [n for n in os.listdir(store.stage_dir) if n.startswith(".tmp_")] == [])

expect("缺 requested_length 拒绝",
       lambda: save_av_latent(store, make_av(), {"format": FORMAT_AVLATENT,
                                                 "native_fps": 24}),
       code="INVALID_REQUEST", contains="requested_length")
expect("native_fps!=24 拒绝",
       lambda: save_av_latent(store, make_av(), base_meta(native_fps=30)),
       code="INVALID_REQUEST", contains="24")
expect("format 不符拒绝",
       lambda: save_av_latent(store, make_av(), base_meta(format="x.v9")),
       code="INVALID_REQUEST", contains="h3.avlatent.v1")

print("== noise_mask 与可选字段 ==")


def _mask_roundtrip():
    lat = make_av()
    v, a = lat["samples"].tensors
    lat["noise_mask"] = NestedTensor((torch.ones(1, 1, 3, 4, 5), torch.ones(1, 1, 2, 6)))
    info = save_av_latent(store, lat, base_meta(), filename="av_mask.safetensors")
    back, md = load_av_latent(store, "av_mask.safetensors")
    assert isinstance(back["noise_mask"], NestedTensor), type(back["noise_mask"])
    assert tuple(back["noise_mask"].tensors[0].shape) == (1, 1, 3, 4, 5)
    return "ok"


check("noise_mask 拆成 video/audio 后可重建", _mask_roundtrip)

print("== cond ==")
cond_ok = {"cond": torch.randn(1, 8, 64, dtype=torch.bfloat16),
           "minimax_token_tags": torch.zeros(1, 8, dtype=torch.long),
           "attention_mask": torch.ones(1, 8, dtype=torch.long)}
cinfo = save_cond(store, cond_ok, {"mode": "ref2va", "width": 640},
                  filename="cond1.safetensors")
check("save_cond 摘要", lambda: cinfo["format"] == FORMAT_COND)
ct, cmd = load_cond(store, "cond1.safetensors")
check("cond roundtrip dtype/tags",
      lambda: ct["cond"].dtype is torch.bfloat16
      and ct["minimax_token_tags"].dtype is torch.long)
check("cond metadata 记录 tags 来源",
      lambda: cmd["token_tags_source"] == "encoder" and cmd["mode"] == "ref2va")
check("cond 无 GPU/cpu 位移",
      lambda: all(t.device.type == "cpu" for t in ct.values()))

expect("tags=None 拒绝写成功文件",
       lambda: save_cond(store, {"cond": torch.zeros(1, 4, 8),
                                 "minimax_token_tags": None}, {}),
       code="INVALID_ARTIFACT", contains="minimax_token_tags")
expect("缺 tags 键拒绝",
       lambda: save_cond(store, {"cond": torch.zeros(1, 4, 8)}, {}),
       code="INVALID_ARTIFACT", contains="minimax_token_tags")
expect("cond 未知字段拒绝",
       lambda: save_cond(store, dict(cond_ok, foo=torch.zeros(1)), {}),
       code="INVALID_ARTIFACT", contains="foo")


def _synth():
    info = save_cond(store, {"cond": torch.zeros(1, 4, 8),
                             "minimax_token_tags": None},
                     {"mode": "ref2va"},
                     filename="cond_synth.safetensors",
                     synthesize_tags=lambda cond: torch.zeros(
                         tuple(cond.shape[:-1]), dtype=torch.long))
    back, md = load_cond(store, "cond_synth.safetensors")
    assert md["token_tags_source"] == "synthesized", md
    assert tuple(back["minimax_token_tags"].shape) == (1, 4)
    return "ok"


check("显式 synthesize_tags 策略可写入并标注来源", _synth)


def _tag_shape_mismatch():
    save_cond(store, {"cond": torch.zeros(1, 8, 64),
                      "minimax_token_tags": torch.zeros(1, 7, dtype=torch.long)},
              {})


expect("tags/cond token 维不一致拒绝", _tag_shape_mismatch,
       code="INVALID_ARTIFACT", contains="不匹配")


def _legacy():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "cond_legacy.safetensors")
    save_file({"cond": torch.zeros(1, 4, 8),
               "minimax_token_tags": torch.zeros(1, 4, dtype=torch.long)}, p)
    back, md = load_cond(store, "cond_legacy.safetensors")
    assert md.get("format") is None
    return "legacy 无 metadata 仍可读"


check("legacy cond 兼容", _legacy)


def _legacy_missing():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "cond_broken.safetensors")
    save_file({"cond": torch.zeros(1, 4, 8)}, p)
    load_cond(store, "cond_broken.safetensors")


expect("缺必需 key 的 legacy 文件拒绝", _legacy_missing,
       code="INVALID_ARTIFACT", contains="minimax_token_tags")

print("== 数值探针 ==")


def _probe():
    t = torch.tensor([1.0, float("nan"), 3.0, float("inf"), -2.0])
    info = probe_tensor_chunked(t, chunk_elems=2)
    assert info["nan"] == 1 and info["inf"] == 1, info
    assert info["min"] == -2.0 and info["max"] == 3.0, info
    assert info["elements"] == 5
    return info


check("分块 NaN/Inf/min/max", _probe)
check("空 tensor 探针",
      lambda: probe_tensor_chunked(torch.zeros(0))["elements"] == 0)
check("整型 tensor 不报错",
      lambda: probe_tensor_chunked(torch.arange(5))["max"] == 4.0)


def _probe_latent():
    lat = {"video": torch.zeros(1, 24, 2, 2, 2), "audio": torch.ones(1, 32, 2, 4)}
    info = probe_av_latent(lat)
    assert info["ok"] is True
    lat["audio"][0, 0, 0, 0] = float("nan")
    info = probe_av_latent(lat)
    assert info["ok"] is False and info["audio"]["nan"] == 1
    return "ok"


check("整体探针标记 ok=False", _probe_latent)

print("== 注入 hook（不依赖 comfy 类）==")

class MyContainer(object):
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True


def _custom_hook():
    lat = {"samples": MyContainer((torch.zeros(2, 24, 1, 2, 2),
                                  torch.zeros(2, 32, 2, 3)))}
    tensors, extra = split_av_latent(
        lat, split_container=lambda c: (c.tensors[0], c.tensors[1], "video,audio"))
    assert extra["container_order"] == "video,audio"
    assert tuple(tensors["video"].shape)[0] == 2
    return "ok"


check("自定义 split_container", _custom_hook)


def _custom_build():
    info = save_av_latent(store, make_av(), base_meta(), filename="av_hook.safetensors")
    back, _ = load_av_latent(
        store, "av_hook.safetensors",
        build_container=lambda v, a, order: {"video": v, "audio": a})
    assert set(back["samples"]) == {"video", "audio"}
    return "ok"


check("自定义 build_container", _custom_build)

print("== batch_index 无损 roundtrip ==")


def _batch_roundtrip():
    lat = make_av()
    lat["batch_index"] = [0, 3, 7]
    save_av_latent(store, lat, base_meta(), filename="av_bi.safetensors")
    back, md = load_av_latent(store, "av_bi.safetensors")
    assert back.get("batch_index") == [0, 3, 7], back.get("batch_index")
    assert md.get("batch_index") == [0, 3, 7], md.get("batch_index")
    return "ok"


check("batch_index 列表 roundtrip", _batch_roundtrip)


def _batch_tensor():
    lat = make_av()
    lat["batch_index"] = torch.tensor([1, 2])
    save_av_latent(store, lat, base_meta(), filename="av_bi2.safetensors")
    back, _ = load_av_latent(store, "av_bi2.safetensors")
    assert back["batch_index"] == [1, 2], back["batch_index"]
    return "tensor -> list"


check("batch_index tensor 无损转列表", _batch_tensor)

expect("batch_index 字符串拒绝",
       lambda: split_av_latent(dict(make_av(), batch_index="0,1")),
       code="INVALID_ARTIFACT", contains="JSON 整数列表")
expect("batch_index 浮点拒绝",
       lambda: split_av_latent(dict(make_av(), batch_index=[1.5])),
       code="INVALID_ARTIFACT", contains="不是整数")

print("== noise_mask 形态约束 ==")


def _mask_pairs_only():
    lat = make_av()
    lat["noise_mask_video"] = torch.ones(1, 1, 3, 4, 5)
    try:
        split_av_latent(lat)
    except ArtifactError as e:
        assert e.code == "INVALID_ARTIFACT"
        assert "成对" in str(e), str(e)
        return "拒绝"
    raise AssertionError("只有 video mask 却没报错")


check("只有 noise_mask_video 时 422", _mask_pairs_only)


def _mask_both():
    lat = make_av()
    lat["noise_mask"] = torch.ones(1, 1, 3, 4, 5)
    lat["noise_mask_video"] = torch.ones(1, 1, 3, 4, 5)
    lat["noise_mask_audio"] = torch.ones(1, 1, 2, 6)
    try:
        split_av_latent(lat)
    except ArtifactError as e:
        assert "不能同时" in str(e), str(e)
        return "拒绝"
    raise AssertionError("unified + split 同时存在却没报错")


check("unified + split 同时存在 422", _mask_both)


def _mask_5d_unified():
    lat = make_av()
    lat["noise_mask"] = torch.ones(1, 1, 3, 4, 5)
    try:
        split_av_latent(lat)
    except ArtifactError as e:
        assert "audio 部分" in str(e), str(e)
        return "拒绝"
    raise AssertionError("5D 单块 mask 缺 audio 却没报错")


check("5D 单块 mask 缺 audio 422", _mask_5d_unified)

print("== 读取前 header 校验 ==")


def _no_metadata():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "lat_nometa.safetensors")
    save_file({"video": torch.zeros(1, 24, 2, 2, 2),
               "audio": torch.zeros(1, 32, 2, 4)}, p)
    try:
        load_av_latent(store, "lat_nometa.safetensors")
    except ArtifactError as e:
        assert e.code == "INVALID_ARTIFACT" and "format" in str(e), str(e)
        return "拒绝"
    raise AssertionError("缺 metadata 的 latent 被接受")


check("latent 缺 metadata format 拒绝", _no_metadata)


def _bad_fps():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "lat_fps.safetensors")
    save_file({"video": torch.zeros(1, 24, 2, 2, 2),
               "audio": torch.zeros(1, 32, 2, 4)}, p,
              metadata={"format": FORMAT_AVLATENT, "requested_length": "124",
                        "native_fps": "30"})
    try:
        load_av_latent(store, "lat_fps.safetensors")
    except ArtifactError as e:
        assert "24" in str(e), str(e)
        return "拒绝"
    raise AssertionError("native_fps=30 被接受")


check("latent native_fps!=24 拒绝", _bad_fps)


def _missing_length():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "lat_nolen.safetensors")
    save_file({"video": torch.zeros(1, 24, 2, 2, 2),
               "audio": torch.zeros(1, 32, 2, 4)}, p,
              metadata={"format": FORMAT_AVLATENT, "native_fps": "24"})
    try:
        load_av_latent(store, "lat_nolen.safetensors")
    except ArtifactError as e:
        assert "requested_length" in str(e), str(e)
        return "拒绝"
    raise AssertionError("缺 requested_length 被接受")


check("latent 缺 requested_length 拒绝", _missing_length)


def _bad_video_shape():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "lat_shape.safetensors")
    save_file({"video": torch.zeros(1, 16, 2, 2, 2),
               "audio": torch.zeros(1, 32, 2, 4)}, p,
              metadata={"format": FORMAT_AVLATENT, "requested_length": "124",
                        "native_fps": "24"})
    try:
        load_av_latent(store, "lat_shape.safetensors")
    except ArtifactError as e:
        assert "video shape" in str(e), str(e)
        return "拒绝"
    raise AssertionError("C=16 的 video 被接受")


check("latent video 通道数错误拒绝", _bad_video_shape)


def _unknown_key():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "lat_extra.safetensors")
    save_file({"video": torch.zeros(1, 24, 2, 2, 2),
               "audio": torch.zeros(1, 32, 2, 4),
               "mystery": torch.zeros(1)}, p,
              metadata={"format": FORMAT_AVLATENT, "requested_length": "124",
                        "native_fps": "24"})
    try:
        load_av_latent(store, "lat_extra.safetensors")
    except ArtifactError as e:
        assert "未支持" in str(e), str(e)
        return "拒绝"
    raise AssertionError("未知 key 被接受")


check("latent 未知 key 拒绝", _unknown_key)


def _bad_build():
    info = save_av_latent(store, make_av(), base_meta(), filename="av_bad.safetensors")
    try:
        load_av_latent(store, "av_bad.safetensors",
                       build_container=lambda v, a, o: tuple([a, v]))
    except ArtifactError as e:
        assert "顺序" in str(e) or "内容" in str(e), str(e)
        return "拒绝"
    raise AssertionError("注入的 build 调换顺序却没报错")


check("build 调换顺序被识别", _bad_build)

print("== cond header 校验与 extensions ==")


def _cond_wrong_format():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "cond_wrongfmt.safetensors")
    save_file({"cond": torch.zeros(1, 4, 8),
               "minimax_token_tags": torch.zeros(4, dtype=torch.long)}, p,
              metadata={"format": "h3.avlatent.v1"})
    try:
        load_cond(store, "cond_wrongfmt.safetensors")
    except ArtifactError as e:
        assert "format" in str(e), str(e)
        return "拒绝"
    raise AssertionError("错误 format 的 cond 被接受")


check("cond 明确错误 format 拒绝", _cond_wrong_format)


def _cond_tags_dtype():
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "cond_baddtype.safetensors")
    save_file({"cond": torch.zeros(1, 4, 8),
               "minimax_token_tags": torch.zeros(4, dtype=torch.float32)}, p,
              metadata={"format": FORMAT_COND})
    try:
        load_cond(store, "cond_baddtype.safetensors")
    except ArtifactError as e:
        assert "整型" in str(e), str(e)
        return "拒绝"
    raise AssertionError("浮点 tags 被接受")


check("cond tags 非整型拒绝", _cond_tags_dtype)


def _cond_1d_tags():
    """真实 ClipProj 返回 1D tags [seq]；必须接受。"""
    from safetensors.torch import save_file
    p = os.path.join(store.stage_dir, "cond_1d.safetensors")
    save_file({"cond": torch.zeros(1, 9, 16),
               "minimax_token_tags": torch.zeros(9, dtype=torch.long)}, p,
              metadata={"format": FORMAT_COND})
    t, md = load_cond(store, "cond_1d.safetensors")
    assert tuple(t["minimax_token_tags"].shape) == (9,)
    return "1D tags 接受"


check("cond 1D tags（真实 ClipProj）可读", _cond_1d_tags)


def _cond_ext_roundtrip():
    ext = ("custom_block",)
    info = save_cond(store, dict(cond_ok, custom_block=torch.ones(1, 2)),
                     {"mode": "ref2va"}, filename="cond_ext.safetensors",
                     extensions=ext)
    md = store.get(info["id"])["metadata"]
    assert md.get("extensions") == '["custom_block"]', md
    t, _ = load_cond(store, "cond_ext.safetensors", extensions=ext)
    assert "custom_block" in t
    try:
        load_cond(store, "cond_ext.safetensors")     # 不声明扩展 → 明确拒绝
    except ArtifactError as e:
        assert "未支持" in str(e), str(e)
        return "声明的扩展被真正保存"
    raise AssertionError("扩展字段未被保存")


check("extensions 声明后真正保存并可读回", _cond_ext_roundtrip)


def _custom_split_used():
    """自定义 split_container 必须真的被调用（之前被忽略）。"""
    called = []

    class Box(object):
        def __init__(self, a, b):
            self.first = a
            self.second = b

    def my_split(container):
        called.append(1)
        return container.first, container.second, "video,audio"

    box = Box(torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 4))
    tensors, extra = split_av_latent({"samples": box}, split_container=my_split)
    assert called, "split_container 未被调用"
    assert extra["container_order"] == "video,audio"
    return "ok"


check("自定义 split_container 被真正使用", _custom_split_used)

print()
shutil.rmtree(TMP, ignore_errors=True)
if fails:
    print("FAILED %d: %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL OK (fixture=%s, torch=%s)" % (FIXTURE_KIND, torch.__version__))
