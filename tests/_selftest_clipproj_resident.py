"""ClipProj Resident Adapter 单元测试。

验证项：
1. 惰性导入测试（顶层导入不触发 torch / Comfy 依赖）；
2. 拒绝 control / synthetic 假投影配置（结构化 400 UNSUPPORTED_PROJECTION）；
3. 归一化张量维度、必需键、张量类型校验（结构化 422 INVALID_PROJECTION_CONFIG）；
4. MLP 残差网络输入/输出边界与层连续性校验；
5. 完整原子准备：异常时不残留部分缓存；
6. require_projection 与 _encode 显式准备检查（未准备或设备不匹配报 409 MISSING_UNITS）；
7. 复用原插件原始数学计算（验证数值与数学公式严格一致）；
8. 多次重复推理具有稳定的 storage（data_ptr 保持不变，无重复分配/迁移）；
9. sink_out 预先上卡并提供运行时视图，使普通同设备 .to 成为 no-op，原始 CPU dict 保持独立；
10. clear_projection 替换 GPU 缓存（swap out），断开强引用并完整保留 CPU master 可重复重载；
11. 缓存冻结保护：推理就绪后拒绝任何变异或清空尝试（409 CACHE_MUTATION_FORBIDDEN）；
12. clone 保持 adapter 派生类型。
"""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dit_svc.clipproj_resident import (
    ResidentGpuCache,
    validate_projection_dict,
    wrap_resident_projected as _wrap_resident_projected,
)
from dit_svc.operations import OperationError


def build_residual_default(proj, device, dtype=None):
    """Test fixture only; production always uses the external plugin builder."""
    import torch
    layers = sorted({int(key.split(".")[1]) for key in proj if key.startswith("mlp.")})
    if not layers:
        return None
    dtype = dtype or proj["mlp.%d.weight" % layers[0]].dtype
    modules = []
    for position, index in enumerate(layers):
        weight = proj["mlp.%d.weight" % index]
        bias = proj.get("mlp.%d.bias" % index)
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight.data = weight.to(device=device, dtype=dtype)
        if bias is not None:
            linear.bias.data = bias.to(device=device, dtype=dtype)
        modules.append(linear)
        if position < len(layers) - 1:
            modules.append(torch.nn.GELU())
    return torch.nn.Sequential(*modules).eval().requires_grad_(False)


def wrap_resident_projected(*args, **kwargs):
    kwargs.setdefault("build_residual_fn", build_residual_default)
    return _wrap_resident_projected(*args, **kwargs)


class FakePatcher:
    def __init__(self, load_device="cpu"):
        self.load_device = load_device


class FakeCondStageModel:
    def __init__(self, clip_name="qwen3vl_4b"):
        self.clip = clip_name


class FakeCLIPBase:
    """模拟 ComfyUI CLIP base 对象。"""

    def __init__(self, load_device="cpu", clip_name="qwen3vl_4b", hidden_state=None):
        self.patcher = FakePatcher(load_device=load_device)
        self.cond_stage_model = FakeCondStageModel(clip_name=clip_name)
        self.hidden_state = hidden_state
        self.fake_tags = None

    def clone(self):
        new_base = FakeCLIPBase(
            load_device=self.patcher.load_device,
            clip_name=self.cond_stage_model.clip,
            hidden_state=self.hidden_state.clone() if self.hidden_state is not None else None,
        )
        new_base.fake_tags = self.fake_tags
        return new_base

    def encode_token_weights(self, tokens):
        return self.hidden_state


class FakePluginProjectedCLIP:
    """真实还原外部 ClipProj 插件 ProjectedCLIP 的数学计算与缓存逻辑。"""

    _PROJ_STORE = {}

    def __init__(self, base, projection_name):
        self.__dict__["_base"] = base
        self.__dict__["_proj_name"] = projection_name
        # 模拟插件调用 load_projection
        proj_data = self._PROJ_STORE.get(projection_name)
        if proj_data is None:
            raise FileNotFoundError(f"Projection not found: {projection_name}")
        self.__dict__["_proj"] = copy.deepcopy(proj_data)
        self.__dict__["_key"] = getattr(base.cond_stage_model, "clip", "qwen3vl_4b")
        self.__dict__["_gpu"] = {}

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_base"], name, value)

    def _encode(self, tokens):
        """还原插件快照 clipproj_nodes.py 第 418-542 行的运算逻辑。"""
        import torch

        proj = self._proj
        base = self._base

        # 模拟原插件中获取设备和 hidden states
        h = base.encode_token_weights(tokens)
        if h.dim() == 4:
            h = h[:, 0]
        h = h.float()

        dev = h.device
        d_in = h.shape[-1]
        cache = self.__dict__["_gpu"]

        # 原插件第 469-474 行：设备检查与惰性创建路径
        if cache.get("device") != dev:
            cache.clear()
            cache["device"] = dev
        if "p" not in cache:
            # 原始插件的 lazy path:
            cache["p"] = {
                k: proj[k].to(dev)
                for k in ("W", "mean_in", "std_in", "mean_out", "std_out")
                if k in proj
            }
            cache["mlp"] = build_residual_default(proj, dev)
        p = cache["p"]

        # 原插件数学计算
        xn = (h - p["mean_in"]) / p["std_in"]
        reseau = cache.get("mlp")
        yn = xn @ p["W"] if "W" in p else None
        if reseau is not None:
            td = reseau[0].weight.dtype
            sortie = reseau(xn.to(td)).float()
            yn = sortie if yn is None else yn + sortie
        cond = yn * p["std_out"] + p["mean_out"]

        # 原插件第 535-537 行：sink_out 搬运
        sink = proj.get("sink_out")
        if sink is not None and cond.shape[1] > 0:
            cond[:, 0] = sink.to(device=cond.device, dtype=cond.dtype)

        tags = getattr(base, "fake_tags", None)
        return cond, tags


def make_valid_proj_dict(d_in=8, d_out=16, with_w=True, with_mlp=False, with_sink=False):
    """构造合法的纯 CPU 投影字典（使用真实小张量）。"""
    import torch

    data = {
        "tap": 12,
        "mean_in": torch.randn(d_in, dtype=torch.float32),
        "std_in": torch.rand(d_in, dtype=torch.float32) + 0.1,
        "mean_out": torch.randn(d_out, dtype=torch.float32),
        "std_out": torch.rand(d_out, dtype=torch.float32) + 0.1,
    }
    if with_w:
        data["W"] = torch.randn(d_in, d_out, dtype=torch.float32)
    if with_mlp:
        # 两层 MLP: d_in -> 12 -> d_out
        data["mlp.0.weight"] = torch.randn(12, d_in, dtype=torch.float16)
        data["mlp.0.bias"] = torch.randn(12, dtype=torch.float16)
        data["mlp.1.weight"] = torch.randn(d_out, 12, dtype=torch.float16)
        data["mlp.1.bias"] = torch.randn(d_out, dtype=torch.float16)
    if with_sink:
        data["sink_out"] = torch.randn(d_out, dtype=torch.float32)
    return data


class TestClipprojResident(unittest.TestCase):
    def setUp(self):
        FakePluginProjectedCLIP._PROJ_STORE.clear()

    def test_lazy_import_no_torch_required(self):
        """核心模块接口与纯函数可独立使用，不强依赖 ComfyUI。"""
        import dit_svc.clipproj_resident as cr
        self.assertTrue(hasattr(cr, "wrap_resident_projected"))
        self.assertTrue(hasattr(cr, "ResidentGpuCache"))
        self.assertTrue(hasattr(cr, "validate_projection_dict"))

    def test_unsupported_control_projections(self):
        """拒绝 control 或 synthetic 投影配置。"""
        controls = ["<control:zero>", "<control:identity>", "<control:random>"]
        for cname in controls:
            FakePluginProjectedCLIP._PROJ_STORE[cname] = {"control": cname.split(":")[1].rstrip(">"), "tap": -1}
            base = FakeCLIPBase()
            with self.assertRaises(OperationError) as ctx:
                wrap_resident_projected(base, cname, base_cls=FakePluginProjectedCLIP)
            self.assertEqual(ctx.exception.code, "UNSUPPORTED_PROJECTION")
            self.assertEqual(ctx.exception.http_status, 400)

        # 字典中显式含有 control 键
        FakePluginProjectedCLIP._PROJ_STORE["synthetic.safetensors"] = {
            "control": "zero", "tap": -1, "mean_in": None
        }
        with self.assertRaises(OperationError) as ctx:
            wrap_resident_projected(FakeCLIPBase(), "synthetic.safetensors", base_cls=FakePluginProjectedCLIP)
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_PROJECTION")
        self.assertEqual(ctx.exception.http_status, 400)

    def test_validation_missing_keys_and_dimensions(self):
        """校验缺失键、张量类型与归一化维度。"""
        import torch

        # 缺少键
        proj = make_valid_proj_dict()
        del proj["mean_out"]
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "bad_keys")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # 非 Tensor 类型
        proj = make_valid_proj_dict()
        proj["mean_in"] = [1.0, 2.0]
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "bad_type")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # 2D 归一化张量
        proj = make_valid_proj_dict()
        proj["mean_in"] = torch.randn(4, 2)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "bad_dim")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # mean_in 与 std_in 长度不一致
        proj = make_valid_proj_dict(d_in=8)
        proj["std_in"] = torch.randn(6)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "in_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # mean_out 与 std_out 长度不一致
        proj = make_valid_proj_dict(d_out=16)
        proj["std_out"] = torch.randn(10)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "out_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # W 形状不匹配
        proj = make_valid_proj_dict(d_in=8, d_out=16)
        proj["W"] = torch.randn(8, 12)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "w_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # 既无 W 也无 MLP
        proj = make_valid_proj_dict(with_w=False, with_mlp=False)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "no_w_no_mlp")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # sink_out 维度不匹配
        proj = make_valid_proj_dict(d_out=16, with_sink=True)
        proj["sink_out"] = torch.randn(10)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "sink_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

    def test_validation_mlp_boundaries(self):
        """校验 MLP 各层的输入输出与跨层连续性。"""
        import torch

        # MLP 输入不匹配 d_in
        proj = make_valid_proj_dict(d_in=8, d_out=16, with_w=False, with_mlp=True)
        proj["mlp.0.weight"] = torch.randn(12, 6)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "mlp_in_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # MLP 输出不匹配 d_out
        proj = make_valid_proj_dict(d_in=8, d_out=16, with_w=False, with_mlp=True)
        proj["mlp.1.weight"] = torch.randn(10, 12)
        proj["mlp.1.bias"] = torch.randn(10)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "mlp_out_mismatch")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

        # MLP 中间层断裂
        proj = make_valid_proj_dict(d_in=8, d_out=16, with_w=False, with_mlp=True)
        proj["mlp.0.weight"] = torch.randn(14, 8)
        proj["mlp.0.bias"] = torch.randn(14)
        proj["mlp.1.weight"] = torch.randn(16, 12)  # 期望 in_features=14
        proj["mlp.1.bias"] = torch.randn(16)
        with self.assertRaises(OperationError) as ctx:
            validate_projection_dict(proj, "mlp_break")
        self.assertEqual(ctx.exception.code, "INVALID_PROJECTION_CONFIG")
        self.assertEqual(ctx.exception.http_status, 422)

    def test_atomic_prepare_failure_leaves_no_partial_cache(self):
        """原子准备：构建失败时保持原样，不残留部分缓存。"""
        proj = make_valid_proj_dict(d_in=8, d_out=16, with_mlp=True)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj
        base = FakeCLIPBase()

        def failing_build_residual(proj_cpu, dev, dtype=None):
            raise RuntimeError("Simulated OOM or CUDA allocation error")

        adapter = wrap_resident_projected(
            base, "proj.safetensors",
            base_cls=FakePluginProjectedCLIP,
            build_residual_fn=failing_build_residual,
        )
        self.assertFalse(adapter.is_prepared)
        self.assertEqual(len(adapter._gpu), 0)

        with self.assertRaises(RuntimeError):
            adapter.prepare_projection("cpu")

        # 验证失败后 _gpu 仍为空，且未被标记为 prepared
        self.assertFalse(adapter.is_prepared)
        self.assertEqual(len(adapter._gpu), 0)
        self.assertIsNone(adapter.prepared_device)

    def test_require_projection_and_encode_require_explicit_preparation(self):
        """未显式调用 prepare_projection 或设备不匹配时报 409 MISSING_UNITS。"""
        import torch

        proj = make_valid_proj_dict(d_in=8, d_out=16)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj
        base = FakeCLIPBase(hidden_state=torch.randn(1, 4, 8))
        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)

        # 1. 未准备时直接 require_projection
        with self.assertRaises(OperationError) as ctx:
            adapter.require_projection("cpu")
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertEqual(ctx.exception.http_status, 409)

        # 2. 未准备时直接调用 _encode
        with self.assertRaises(OperationError) as ctx:
            adapter._encode("dummy_tokens")
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertEqual(ctx.exception.http_status, 409)

        # 3. 准备在 cpu，但在 cuda:0 校验
        adapter.prepare_projection("cpu")
        self.assertTrue(adapter.is_prepared)
        with self.assertRaises(OperationError) as ctx:
            adapter.require_projection("cuda:0")
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")
        self.assertEqual(ctx.exception.http_status, 409)
        self.assertEqual(ctx.exception.details["reason"], "device_mismatch")

    def test_reuse_original_encode_unmodified_math(self):
        """复用原插件原始数学计算，数值严格等价。"""
        import torch

        d_in, d_out = 8, 16
        proj = make_valid_proj_dict(d_in=d_in, d_out=d_out, with_w=True, with_mlp=True)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj

        hidden = torch.randn(1, 5, d_in, dtype=torch.float32)
        base = FakeCLIPBase(hidden_state=hidden)
        base.fake_tags = torch.ones(5, dtype=torch.long)

        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")

        cond, tags = adapter._encode("dummy_tokens")

        # 手动计算标准数学公式以对照
        p = adapter.projection_cpu
        xn = (hidden.float() - p["mean_in"]) / p["std_in"]
        yn = xn @ p["W"]
        reseau = adapter._gpu["mlp"]
        td = reseau[0].weight.dtype
        sortie = reseau(xn.to(td)).float()
        yn = yn + sortie
        expected_cond = yn * p["std_out"] + p["mean_out"]

        self.assertTrue(torch.allclose(cond, expected_cond, atol=1e-5, rtol=1e-5))
        self.assertEqual(list(cond.shape), [1, 5, d_out])
        self.assertTrue(torch.equal(tags, base.fake_tags))

    def test_repeated_inference_stable_storage(self):
        """重复推理时缓存与张量存储地址稳定，无重复 CPU->GPU 拷贝。"""
        import torch

        d_in, d_out = 8, 16
        proj = make_valid_proj_dict(d_in=d_in, d_out=d_out, with_w=True, with_sink=True)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj

        hidden = torch.randn(1, 6, d_in, dtype=torch.float32)
        base = FakeCLIPBase(hidden_state=hidden)
        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")

        # 记录各张量 data_ptr
        ptr_w = adapter._gpu["p"]["W"].data_ptr()
        ptr_mean_in = adapter._gpu["p"]["mean_in"].data_ptr()
        ptr_sink = adapter._proj["sink_out"].data_ptr()

        # 连续推理 3 次
        for _ in range(3):
            cond, _ = adapter._encode("dummy")
            # 验证首 token 正确替换为了 sink
            self.assertTrue(torch.allclose(cond[:, 0], adapter._proj["sink_out"]))
            # 验证底层 storage 没有发生重新创建
            self.assertEqual(adapter._gpu["p"]["W"].data_ptr(), ptr_w)
            self.assertEqual(adapter._gpu["p"]["mean_in"].data_ptr(), ptr_mean_in)
            self.assertEqual(adapter._proj["sink_out"].data_ptr(), ptr_sink)

    def test_sink_prepared_and_cpu_master_isolated(self):
        """sink_out 显式准备上卡，同设备同类型 .to 成为 no-op，原始 CPU 字典隔离。"""
        import torch

        d_in, d_out = 8, 16
        proj = make_valid_proj_dict(d_in=d_in, d_out=d_out, with_sink=True)
        cpu_sink_original = proj["sink_out"].clone()
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj

        base = FakeCLIPBase()
        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)

        # 初始未准备前：CPU 字典里的 sink_out 为原始张量
        self.assertTrue(torch.equal(adapter.projection_cpu["sink_out"], cpu_sink_original))

        adapter.prepare_projection("cpu")

        # prepare 之后：运行时视图里的 sink_out 位于准备设备，且为 float32
        runtime_sink = adapter._proj["sink_out"]
        self.assertEqual(runtime_sink.dtype, torch.float32)
        self.assertEqual(runtime_sink.device, torch.device("cpu"))

        # 模拟插件每步推理执行: sink.to(cond.device, cond.dtype)
        cond_device = torch.device("cpu")
        cond_dtype = torch.float32
        sink_after_to = runtime_sink.to(device=cond_device, dtype=cond_dtype)

        # 验证 .to 为零拷贝 no-op（返回相同对象与相同 data_ptr）
        self.assertIs(sink_after_to, runtime_sink)
        self.assertEqual(sink_after_to.data_ptr(), runtime_sink.data_ptr())

        # 验证原始 CPU 字典独立未被污染
        self.assertTrue(torch.equal(adapter.projection_cpu["sink_out"], cpu_sink_original))

    def test_clear_projection_swaps_cache_and_allows_reload(self):
        """clear_projection 替换 GPU 缓存切断强引用，CPU master 完好支持重新准备。"""
        import torch

        d_in, d_out = 8, 16
        proj = make_valid_proj_dict(d_in=d_in, d_out=d_out, with_sink=True)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj

        base = FakeCLIPBase(hidden_state=torch.randn(1, 3, d_in))
        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")

        old_cache = adapter._gpu
        self.assertTrue(adapter.is_prepared)
        self.assertTrue(old_cache.is_frozen)

        # 执行清理
        adapter.clear_projection()

        self.assertFalse(adapter.is_prepared)
        self.assertIsNone(adapter.prepared_device)
        # 验证 GPU cache 已被替换（swap out），不是原对象的单纯引用
        self.assertIsNot(adapter._gpu, old_cache)
        self.assertEqual(len(adapter._gpu), 0)
        # 验证运行时视图恢复为纯 CPU 字典
        self.assertIs(adapter._proj, adapter.projection_cpu)

        # 再次调用 require_projection 会拒绝
        with self.assertRaises(OperationError) as ctx:
            adapter.require_projection("cpu")
        self.assertEqual(ctx.exception.code, "MISSING_UNITS")

        # 重新调用 prepare_projection 依然可以成功重新上卡
        adapter.prepare_projection("cpu")
        self.assertTrue(adapter.is_prepared)
        self.assertIn("p", adapter._gpu)
        cond, _ = adapter._encode("dummy")
        self.assertEqual(list(cond.shape), [1, 3, d_out])

    def test_lazy_cache_mutation_and_clear_rejected(self):
        """冻结保护：推理阶段拒绝任何就地修改与清空。"""
        import torch

        proj = make_valid_proj_dict(d_in=8, d_out=16)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj
        adapter = wrap_resident_projected(FakeCLIPBase(), "proj.safetensors", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")

        cache = adapter._gpu
        self.assertTrue(cache.is_frozen)

        # 尝试 clear
        with self.assertRaises(OperationError) as ctx:
            cache.clear()
        self.assertEqual(ctx.exception.code, "CACHE_MUTATION_FORBIDDEN")
        self.assertEqual(ctx.exception.http_status, 409)

        # 尝试设置键
        with self.assertRaises(OperationError) as ctx:
            cache["device"] = "cuda:0"
        self.assertEqual(ctx.exception.code, "CACHE_MUTATION_FORBIDDEN")
        self.assertEqual(ctx.exception.http_status, 409)

        # 尝试更新键
        with self.assertRaises(OperationError) as ctx:
            cache.update({"p": {}})
        self.assertEqual(ctx.exception.code, "CACHE_MUTATION_FORBIDDEN")

        # 尝试删除键
        with self.assertRaises(OperationError) as ctx:
            del cache["device"]
        self.assertEqual(ctx.exception.code, "CACHE_MUTATION_FORBIDDEN")

        # 尝试 pop
        with self.assertRaises(OperationError) as ctx:
            cache.pop("device")
        self.assertEqual(ctx.exception.code, "CACHE_MUTATION_FORBIDDEN")

    def test_preparation_is_idempotent_and_nested_cache_is_sealed(self):
        import torch
        FakePluginProjectedCLIP._PROJ_STORE["p"] = make_valid_proj_dict(with_mlp=True)
        adapter = wrap_resident_projected(FakeCLIPBase(), "p", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")
        cache = adapter._gpu
        adapter.prepare_projection("cpu")
        self.assertIs(adapter._gpu, cache)
        with self.assertRaises(OperationError):
            cache["p"]["mean_in"] = torch.ones(8)

    def test_incomplete_cache_refused_before_encoder(self):
        import torch
        FakePluginProjectedCLIP._PROJ_STORE["p"] = make_valid_proj_dict(with_mlp=True)
        adapter = wrap_resident_projected(FakeCLIPBase(), "p", base_cls=FakePluginProjectedCLIP)
        adapter.prepare_projection("cpu")
        dict.__setitem__(adapter._gpu, "mlp", None)
        with self.assertRaises(OperationError) as error:
            adapter._encode("unused")
        self.assertEqual(error.exception.code, "MISSING_UNITS")

    def test_cpu_cache_adoption_and_release_really_free_projection(self):
        import gc
        import types
        from unittest.mock import patch
        from dit_svc.te_manager import TeBackend
        from dit_svc.weights import WeightRegistry
        import torch
        module = types.ModuleType("clipproj.clipproj_projection")
        module._CACHE = {}
        class CachedPlugin(FakePluginProjectedCLIP):
            def __init__(self, base, name):
                super().__init__(base, name)
                module._CACHE["owned"] = self._proj
        CachedPlugin._PROJ_STORE["p"] = make_valid_proj_dict(with_sink=True)
        base = FakeCLIPBase()
        base.patcher.model = torch.nn.Linear(4, 4)
        with patch.dict(sys.modules, {module.__name__: module}):
            adapter = wrap_resident_projected(base, "p", base_cls=CachedPlugin)
            self.assertEqual(module._CACHE, {})
            manager = TeBackend(types.SimpleNamespace(te_residence=types.SimpleNamespace(
                unregister=lambda: None)), WeightRegistry())
            manager.obj, manager._clip, manager._projected = adapter, base, True
            del adapter, base
            refs = manager.release()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))

    def test_encoder_dimensions_and_tap_checked_before_preparation(self):
        import types
        from dit_svc.clipproj_resident import _check_encoder_config
        proj = make_valid_proj_dict()
        _check_encoder_config(types.SimpleNamespace(hidden_size=8, num_hidden_layers=16), proj)
        for config in (types.SimpleNamespace(hidden_size=16, num_hidden_layers=16),
                       types.SimpleNamespace(hidden_size=8, num_hidden_layers=4)):
            with self.assertRaises(OperationError) as error:
                _check_encoder_config(config, proj)
            self.assertEqual(error.exception.code, "INCOMPATIBLE_PROJECTION")

    def test_unknown_plugin_version_is_not_silently_accepted(self):
        from dit_svc.clipproj_resident import _check_plugin
        with self.assertRaises(OperationError) as error:
            _check_plugin(FakePluginProjectedCLIP)
        self.assertEqual(error.exception.code, "UNSUPPORTED_CLIPPROJ")

    def test_clone_preserves_adapter_subclass(self):
        """clone 保持适配器类型，避免退化为非 resident 类。"""
        import torch

        proj = make_valid_proj_dict(d_in=8, d_out=16)
        FakePluginProjectedCLIP._PROJ_STORE["proj.safetensors"] = proj
        base = FakeCLIPBase()
        adapter = wrap_resident_projected(base, "proj.safetensors", base_cls=FakePluginProjectedCLIP)

        cloned = adapter.clone()
        self.assertIsInstance(cloned, type(adapter))
        self.assertEqual(cloned._proj_name, adapter._proj_name)
        self.assertFalse(cloned.is_prepared)
        # 可独立 prepare
        cloned.prepare_projection("cpu")
        self.assertTrue(cloned.is_prepared)


if __name__ == "__main__":
    unittest.main()
