"""attn.install 请求校验与安装/切换/还原状态机（fake 模块，无 torch/Comfy）。

覆盖：
* validate_attn_install_request 的严格校验（mode/veda 参数/未知字段）；
* dense 档（sage）安装：override 装上 unet + model_shifted，runtime.attn 更新；
* veda 档安装：回调注册在原 unet、model_shifted 双保险、veda_patch 发布；
* veda 安装失败：报 ATTN_INSTALL_FAILED 且**还原原档**（不静默回退 int8）；
* bundle 未配置：安装前即拒绝，原档不动；
* mode=none：完整还原 factory（函数引用 identity、override 键摘除、回调摘除）；
* Runtime.operation_admission 对 attn.install 的外壳门槛。
"""
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dit_svc import api_validation as AV                     # noqa: E402
from dit_svc import attn_service, veda_star7                 # noqa: E402
import dit_svc.runtime as runtime_mod                        # noqa: E402
from dit_svc.operations import OperationError               # noqa: E402


# ------------------------------------------------------------------ fake comfy

class _FakeCallbacksMP:
    ON_PREPARE_STATE = "on_prepare_state"
    ON_CLEANUP = "on_cleanup"


def _install_fake_comfy():
    fake_comfy = types.ModuleType("comfy")
    fake_ext = types.ModuleType("comfy.patcher_extension")
    fake_ext.CallbacksMP = _FakeCallbacksMP
    fake_comfy.patcher_extension = fake_ext
    sys.modules.setdefault("comfy", fake_comfy)
    sys.modules.setdefault("comfy.patcher_extension", fake_ext)


_install_fake_comfy()


# ------------------------------------------------------------------ fake 设施

class FakeAttention:
    def __init__(self):
        self.optimized_attention = "orig_attn_fn"
        self.attention_pytorch = "fn_pytorch"
        self.attention_sub_quad = "fn_sub_quad"
        self.attention_comfy_kitchen_int8 = "fn_int8"
        self.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = True


class FakePatcher:
    def __init__(self):
        self.model_options = {}
        self.callbacks = {}

    def add_callback_with_key(self, call_type, key, fn):
        self.callbacks[(call_type, key)] = fn

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.pop((call_type, key), None)


def make_runtime(veda_path="D:/fake_veda.safetensors"):
    fake_backend = SimpleNamespace(
        config={"path": veda_path} if veda_path else None,
        configured=lambda: bool(veda_path))
    runtime = SimpleNamespace(
        attention=FakeAttention(),
        minimax_model=SimpleNamespace(optimized_attention="orig_mm_fn"),
        comfy_kitchen=SimpleNamespace(int8_attention_is_available=lambda d: True),
        unet=FakePatcher(),
        model_shifted=FakePatcher(),
        rt=SimpleNamespace(
            TURING_UTILS_DIR="X",
            _wrap_attn_mem_probe=lambda fn, tag: "wrapped:%s" % tag,
            install_tuned_sub_quad_chunking=lambda att: None),
        args=SimpleNamespace(),
        weights=SimpleNamespace(backends={"veda.bundle": fake_backend}),
        attn="none", attn_override=None, veda_patch=None,
        _attn_factory=None, _attn_installed_request=None)
    return runtime


def _override_identity(rt_mod_state, mode):
    """fake dense installer：同一 mode 恒返回同一 override 对象（便于断言）。"""
    cache = rt_mod_state.setdefault("cache", {})
    if mode not in cache:
        cache[mode] = SimpleNamespace(
            turing_utils_attention_implementation="turing_%s" % mode)
    return cache[mode]


_DENSE_STATE = {}


def _fake_dense_install(rt, attention, minimax_model, comfy_kitchen, mode,
                        args=None, unet=None):
    attention.optimized_attention = "installed_%s" % mode
    minimax_model.optimized_attention = "installed_%s" % mode
    if mode in ("sage", "w8a8"):
        return _override_identity(_DENSE_STATE, mode)
    return None


def _fake_veda_install_ok(unet, args, torch=None, log=print):
    unet.add_callback_with_key(
        _FakeCallbacksMP.ON_PREPARE_STATE, veda_star7.CALLBACK_KEY,
        lambda mp, ts, mo: None)
    unet.add_callback_with_key(
        _FakeCallbacksMP.ON_CLEANUP, veda_star7.CALLBACK_KEY, lambda mp: None)
    patch = SimpleNamespace(
        install=lambda topts: topts.update(veda_installed=True))
    return patch, {"bundle": "fake.safetensors", "plans": 50}


def _fake_veda_install_boom(unet, args, torch=None, log=print):
    raise RuntimeError("SM75 kernel self_test 未过")


class _Patches:
    """统一 monkeypatch：dense 走 runtime._install_attention，veda 走模块属性。"""

    def __init__(self, veda_fn):
        self._saved_dense = runtime_mod._install_attention
        self._saved_veda = veda_star7.install
        runtime_mod._install_attention = _fake_dense_install
        veda_star7.install = veda_fn

    def restore(self):
        runtime_mod._install_attention = self._saved_dense
        veda_star7.install = self._saved_veda


# ------------------------------------------------------------------ 校验

class ValidationTests(unittest.TestCase):
    def test_modes_and_defaults(self):
        for mode in ("int8", "sdpa", "sage", "w8a8", "none"):
            req = AV.validate_attn_install_request({"mode": mode})
            self.assertEqual(req["mode"], mode)
            self.assertEqual(req["veda"]["warmup"], "probe")
            self.assertEqual(req["veda"]["keep_tiles"], 0)
            self.assertEqual(req["veda"]["tile_precision"], "fp32")
        req = AV.validate_attn_install_request({"mode": "veda"})
        self.assertEqual(req["veda"]["chunk_share"], 16)
        self.assertIsNone(req["request_id"])

    def test_unknown_field_rejected(self):
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request({"mode": "sage", "bogus": 1})

    def test_bad_mode_rejected(self):
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request({"mode": "flash"})

    def test_veda_params_only_with_veda_mode(self):
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "sage", "veda": {"keep_tiles": 32}})

    def test_veda_ranges(self):
        req = AV.validate_attn_install_request({
            "mode": "veda",
            "veda": {"keep_tiles": 32, "ref_keep_tiles": -1, "keep_ratio": 0.25,
                     "dense_layers": "0,1,47-49", "dense_steps": "3",
                     "tile_precision": "fp16", "chunk_share": 8,
                     "warmup": "tier1", "debug": True}})
        self.assertEqual(req["veda"]["dense_layers"], "0,1,47-49")
        self.assertEqual(req["veda"]["keep_ratio"], 0.25)
        self.assertTrue(req["veda"]["debug"])
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"keep_tiles": 9999}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"keep_ratio": 1.5}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"dense_layers": "5-3"}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"dense_layers": "64"}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"warmup": "tier9"}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"tile_precision": "fp8"}})
        with self.assertRaises(AV.ApiError):
            AV.validate_attn_install_request(
                {"mode": "veda", "veda": {"debug": "yes"}})


# ------------------------------------------------------------------ 状态机

class InstallTests(unittest.TestCase):
    def setUp(self):
        # fake torch：_install_veda 只用 torch.device("cuda:0") 构造对象，
        # 不触 CUDA；fake 掉避免真实 torch import 拖慢 stdlib 自测。
        self._saved_torch = sys.modules.get("torch")
        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda value: "device:%s" % value
        sys.modules["torch"] = fake_torch
        self.runtime = make_runtime()
        self.patches = _Patches(_fake_veda_install_ok)

    def tearDown(self):
        self.patches.restore()
        if self._saved_torch is not None:
            sys.modules["torch"] = self._saved_torch
        else:
            sys.modules.pop("torch", None)

    def _topts(self, patcher):
        return patcher.model_options.setdefault("transformer_options", {})

    def test_sage_install(self):
        result = attn_service.install(self.runtime, {"mode": "sage", "veda": {}})
        self.assertEqual(result["mode"], "sage")
        self.assertEqual(self.runtime.attn, "sage")
        override = _override_identity(_DENSE_STATE, "sage")
        self.assertIs(self._topts(self.runtime.unet)["optimized_attention_override"],
                      override)
        self.assertIs(
            self._topts(self.runtime.model_shifted)["optimized_attention_override"],
            override)
        self.assertIs(self.runtime.attn_override, override)
        self.assertEqual(self.runtime._attn_installed_request["mode"], "sage")

    def test_int8_install_no_override(self):
        result = attn_service.install(self.runtime, {"mode": "int8", "veda": {}})
        self.assertEqual(result["mode"], "int8")
        self.assertIsNone(self.runtime.attn_override)
        self.assertNotIn("optimized_attention_override",
                         self._topts(self.runtime.unet))

    def test_veda_install(self):
        req = {"mode": "veda",
               "veda": {"warmup": "off", "keep_tiles": 32}}
        result = attn_service.install(self.runtime, req)
        self.assertEqual(result["mode"], "veda")
        self.assertEqual(result["veda_info"]["bundle"], "fake.safetensors")
        self.assertIsNotNone(self.runtime.veda_patch)
        self.assertIn((_FakeCallbacksMP.ON_PREPARE_STATE,
                       veda_star7.CALLBACK_KEY), self.runtime.unet.callbacks)
        self.assertTrue(self._topts(self.runtime.model_shifted).get("veda_installed"))
        # dense fallback 底座被设置（fake rt._wrap_attn_mem_probe 的产物）
        self.assertEqual(self.runtime.attention.optimized_attention,
                         "wrapped:veda-dense")

    def test_veda_failure_restores_previous_mode(self):
        attn_service.install(self.runtime, {"mode": "sage", "veda": {}})
        sage_override = self.runtime.attn_override
        self.patches.restore()
        boom = _Patches(_fake_veda_install_boom)
        try:
            with self.assertRaises(OperationError) as ctx:
                attn_service.install(self.runtime, {"mode": "veda", "veda": {}})
        finally:
            boom.restore()
        err = ctx.exception
        self.assertEqual(err.code, "ATTN_INSTALL_FAILED")
        self.assertEqual(err.details["restored_mode"], "sage")
        self.assertEqual(self.runtime.attn, "sage")
        self.assertIs(self.runtime.attn_override, sage_override)
        # veda 从未装上：无 veda 回调残留在原 unet 上
        self.assertNotIn((_FakeCallbacksMP.ON_PREPARE_STATE,
                          veda_star7.CALLBACK_KEY), self.runtime.unet.callbacks)
        self.assertNotIn((_FakeCallbacksMP.ON_CLEANUP,
                          veda_star7.CALLBACK_KEY), self.runtime.unet.callbacks)

    def test_missing_bundle_keeps_previous_mode(self):
        self.runtime.weights.backends["veda.bundle"] = SimpleNamespace(
            config=None, configured=lambda: False)
        attn_service.install(self.runtime, {"mode": "sage", "veda": {}})
        with self.assertRaises(OperationError) as ctx:
            attn_service.install(self.runtime, {"mode": "veda", "veda": {}})
        self.assertEqual(ctx.exception.code, "ATTN_NOT_READY")
        self.assertEqual(self.runtime.attn, "sage")

    def test_none_restores_factory(self):
        self.runtime.attention.optimized_attention = "orig_attn_fn"
        attn_service.install(self.runtime, {"mode": "sage", "veda": {}})
        attn_service.install(self.runtime, {"mode": "veda",
                                           "veda": {"warmup": "off"}})
        result = attn_service.install(self.runtime, {"mode": "none", "veda": {}})
        self.assertEqual(result["mode"], "none")
        self.assertEqual(result["previous_mode"], "veda")
        self.assertEqual(self.runtime.attn, "none")
        self.assertIs(self.runtime.attention.optimized_attention, "orig_attn_fn")
        self.assertEqual(self.runtime.minimax_model.optimized_attention,
                         "orig_mm_fn")
        self.assertNotIn("optimized_attention_override",
                         self._topts(self.runtime.unet))
        self.assertNotIn("optimized_attention_override",
                         self._topts(self.runtime.model_shifted))
        self.assertEqual(self.runtime.unet.callbacks, {})
        self.assertIsNone(self.runtime.attn_override)
        self.assertIsNone(self.runtime.veda_patch)
        self.assertIsNone(self.runtime._attn_installed_request)

    def test_shell_missing_raises_not_ready(self):
        self.runtime.unet = None
        with self.assertRaises(OperationError) as ctx:
            attn_service.install(self.runtime, {"mode": "sdpa", "veda": {}})
        self.assertEqual(ctx.exception.code, "ATTN_NOT_READY")

    def test_switch_sage_to_sdpa(self):
        attn_service.install(self.runtime, {"mode": "sage", "veda": {}})
        result = attn_service.install(self.runtime, {"mode": "sdpa", "veda": {}})
        self.assertEqual(result["previous_mode"], "sage")
        self.assertEqual(self.runtime.attn, "sdpa")
        self.assertIsNone(self.runtime.attn_override)
        self.assertNotIn("optimized_attention_override",
                         self._topts(self.runtime.unet))
        self.assertEqual(self.runtime.attention.optimized_attention,
                         "installed_sdpa")


class AdmissionTests(unittest.TestCase):
    def test_attn_install_needs_shell(self):
        from dit_svc.runtime import Runtime
        runtime = Runtime(SimpleNamespace(attn="none"), None)
        runtime.env_ready = True
        runtime.device_healthy = True
        with self.assertRaises(OperationError) as ctx:
            runtime.operation_admission("attn.install", {})
        self.assertEqual(ctx.exception.code, "ATTN_NOT_READY")
        runtime.unet = object()
        runtime.model_shifted = object()
        runtime.operation_admission("attn.install", {})  # 通过


def main():
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(ValidationTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(InstallTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(AdmissionTests),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("RESULT %s" % ("passed" if result.wasSuccessful() else "failed"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
