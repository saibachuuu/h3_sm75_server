"""h3_sm75_server 回归自检（从 tests/ 目录直接跑）。

跑法（需要有 torch 的解释器；本地可用 ~/comfyui/.venv）：
  cd h3_sm75_server/tests && python _selftest_bridge.py

背景：2026-10-11 的两次线上启动失败（self._install_attention
找不到 / bundle 缺 keep_ratio）都是"静态 import+compileall
挡不住、真执行才炸"的类型，所以这些脚本必须真实执行到关键
路径，不能只 import。
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser("~/comfyui"))
os.chdir(os.path.expanduser("~/comfyui"))

# --- 造一个最小 ModelPatcher 替身 ---


class Attn:
    heads = 56
    head_dim = 128


class Block:
    attn = Attn()


class DM:
    blocks = [Block() for _ in range(50)]


class Model:
    diffusion_model = DM()


class Patcher:
    def __init__(self):
        self.model = Model()
        self.model_options = {}
        self._cbs = []

    def clone(self):
        import copy as _copy
        n = Patcher()
        n.model = self.model
        n.model_options = _copy.deepcopy(self.model_options)
        return n

    def add_callback_with_key(self, call_type, key, cb):
        self._cbs.append((call_type, key))


class Args:
    veda_bundle = r"D:\minimax_h3_r2va_veda_preview_fp8.safetensors"
    veda_keep_tiles = 0
    veda_ref_keep_tiles = -1
    veda_keep = 0.0
    veda_dense_layers = ""
    veda_dense_steps = ""
    veda_debug = False


from dit_svc import veda_star7  # noqa: E402
import dit_svc.veda_attn_star7 as vs7  # noqa: E402

_real_load_bundle = veda_star7.load_bundle
import veda.comfy_patch as vcp  # noqa: E402
_orig_apply = vcp.apply

print("== 1. check_model_dims: 维度匹配 ==")
unet = Patcher()
loaded = types.SimpleNamespace(num_layers=50, num_heads=56, head_dim=128)
veda_star7.check_model_dims(unet, loaded, log=print)
print("   OK")

print("== 2. check_model_dims: 维度不匹配必须抛错 ==")
bad = types.SimpleNamespace(num_layers=48, num_heads=56, head_dim=128)
try:
    veda_star7.check_model_dims(unet, bad, log=lambda s: None)
    print("   FAIL: 没抛错")
    raise SystemExit(1)
except RuntimeError as e:
    print("   OK RuntimeError:", str(e)[:80])

print("== 3. install() 的完整编排（打桩 load_bundle + apply） ==")
calls = []


def fake_load(args, log=print):
    calls.append(("load_bundle", args.veda_bundle))
    bundle = types.SimpleNamespace(
        path=args.veda_bundle, num_layers=50, num_heads=56, head_dim=128,
        plans=types.SimpleNamespace(plans={"16x9_t102": 1}),
        metadata={"target_budget_value": 32.0})

    class _B:
        def __init__(self, b):
            self.bundle = b
            self.num_layers = b.num_layers
            self.num_heads = b.num_heads
            self.head_dim = b.head_dim
            self.generated = types.SimpleNamespace(
                describe=lambda: "32 tiles", tiles=32.0, ratio=None)
            self.reference = types.SimpleNamespace(
                describe=lambda: "32 tiles", tiles=32.0, ratio=None)
            self.budget_source = "target/ref_budget_value (tiles)"

    return _B(bundle)


def fake_apply(model, bundle, settings, node_id):
    calls.append(("apply", settings.generated.describe(), settings.reference.describe(),
                  bool(settings.dense_layers), settings.verbose))
    patched = model.clone()
    installed = []

    def _install(opts):
        installed.append(sorted(opts.keys()))

    patch = types.SimpleNamespace(
        run=types.SimpleNamespace(
            calls={"sparse": 0}, evaluations=0, failed=None, video=None),
        _engines={},
        install=_install)
    patched.add_callback_with_key("on_prepare_state", "veda", lambda *a: None)
    patched.add_callback_with_key("on_cleanup", "veda", lambda *a: None)
    calls.append(("clone+callbacks", patched is not model))
    calls.append(("patch.install", installed or ["<pending>"]))
    return patched, patch


veda_star7.load_bundle = fake_load
# resolve_backend 需要 GPU（DLL + self_test），测试桩跳过；签名与生产一致
# （教训：stub 必须兼容 kwargs，否则就是"测试与生产不一致地绿"）
veda_star7.resolve_backend = lambda torch, log=print, args=None: types.SimpleNamespace(
    backend=types.SimpleNamespace(chunk_share=16, display="stub"),
    report="stub")
vs7.BACKEND_INFO.update(resolved=True, ok=True,
                        display="Star7 VEDA INT8 Tensor Core (SM75)",
                        report="star7-cuda-int8: ok")


def _apply_stub(model, bundle, settings, node_id):
    return fake_apply(model, bundle, settings, node_id)


veda_star7.__dict__["_stub_apply"] = _apply_stub
import dit_svc.veda_star7 as vs  # noqa: E402
# install() 内部是 `import veda.comfy_patch as veda_comfy_patch`，
# 所以要在 veda.comfy_patch 模块上打桩才有意义。
vcp.apply = _apply_stub

# install() 返回 (patch, info) 二元组；runtime 的解包必须一致（第三次线上事故教训）
handle, info = vs.install(unet, Args(), torch=None, log=print)
assert hasattr(handle, "run") and hasattr(handle, "install"), \
    "handle 应是 VedaPatch（有 run 统计与 install 方法）"
assert info["keep_tiles"] == 32.0
print("   calls:", calls)
print("   info :", info)
assert calls[0][0] == "load_bundle"
assert calls[1][0] == "apply" and calls[1][1] == "90% sparse"   # settings 来自 CLI 参数
assert any(c[0] == "clone+callbacks" for c in calls)
assert handle is not None
assert info["keep_tiles"] == 32
print("== 4. install() 在 bundle 缺失时必须抛错 ==")
# 解除第 3 步的全局打桩，验证真实的"文件不存在"守卫
vcp.apply = _orig_apply
veda_star7.load_bundle = _real_load_bundle
missing = Args()
missing.veda_bundle = r"Z:\nope\nothere.safetensors"
try:
    veda_star7.install(unet, missing, torch=None, log=lambda s: None)
    print("   FAIL: 没抛错")
    raise SystemExit(1)
except RuntimeError as e:
    print("   OK RuntimeError:", str(e)[:70])

print()
print("ALL OK")
