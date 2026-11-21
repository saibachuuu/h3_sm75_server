"""纯 stdlib 自检：H3 skeleton -> BaseModel -> 双 ModelPatcher 的元数据包装。

不 import torch、不初始化 CUDA、不执行 forward。所有 Comfy 依赖都是 fake，
fake BaseModel 一旦发现私有 disable 开关没有生效，就会记录“重建大模型”次数
并直接失败。
"""

import os
import sys
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dit_svc.h3_patcher import (
    H3PatcherPair,
    H3_AUDIO_SHIFT,
    H3_VIDEO_SHIFT,
    invalidate_patcher_sizes,
    wrap_h3_skeleton,
)

BF16 = object()
FP32 = object()
LOAD = object()
OFFLOAD = object()


class FakeSkeleton:
    def __init__(self, dtype=BF16):
        self.dtype = dtype
        self.eval_calls = 0
        self.grad_calls = []

    def eval(self):
        self.eval_calls += 1
        return self

    def requires_grad_(self, flag):
        self.grad_calls.append(flag)
        return self


class FakeConfig:
    created = []

    def __init__(self, unet_config):
        self.unet_config = dict(unet_config)
        self.manual_cast_dtype = None
        self.quant_config = "class-default"
        self.sampling_settings = {"shift": 12.0, "audio_shift": 3.0}
        self.memory_usage_factor = 0.114
        self.set_calls = []

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.set_calls.append((dtype, manual_cast_dtype))
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


def make_config(unet_config):
    FakeConfig.created.append(dict(unet_config))
    return FakeConfig(unet_config)


class FakeBase:
    instances = 0
    big_model_creations = 0
    last = None

    def __init__(self, model_config, model_type=None, device=None):
        type(self).instances += 1
        self.model_config = model_config
        self.model_type = model_type
        self.device = device
        self.manual_cast_dtype = model_config.manual_cast_dtype
        if not model_config.unet_config.get("disable_unet_model_creation", False):
            type(self).big_model_creations += 1
            raise AssertionError("private disable flag 未生效：试图创建第二份大模型")
        # 真实 BaseModel 在 disable 开关生效时不会创建 diffusion_model 属性；
        # 这里故意不设置，确保包装器必须自己绑定 skeleton。
        self.model_sampling = "fake-cpu-schedule"
        self.eval_calls = 0
        type(self).last = self

    def eval(self):
        self.eval_calls += 1
        return self


class FakePatcher:
    def __init__(self, model, load_device, offload_device):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = 777
        self.model_options = {"transformer_options": {}}


class FakeSigmaShift:
    calls = []

    @classmethod
    def execute(cls, model, shift_video, shift_audio):
        cls.calls.append((model, shift_video, shift_audio))
        clone = FakePatcher(model.model, model.load_device, model.offload_device)
        clone.model_options = {
            "transformer_options": {
                "minimax_h3_sigma_shift_video": shift_video,
                "minimax_h3_sigma_shift_audio": shift_audio,
            }
        }
        return SimpleNamespace(args=(clone,))


class TupleSigmaShift:
    calls = []

    @classmethod
    def execute(cls, model, shift_video, shift_audio):
        cls.calls.append((model, shift_video, shift_audio))
        return (FakePatcher(model.model, model.load_device, model.offload_device),)


class ExplodingTorch:
    """任何属性访问都失败，用来证明顶层没有碰 CUDA / dtype / device。"""

    def __init__(self):
        self.accessed = []

    def __getattr__(self, name):
        self.accessed.append(name)
        raise AssertionError("不应访问 torch_mod.%s" % name)


SUPPORTED = SimpleNamespace(MiniMaxH3=make_config)
BASE_MOD = SimpleNamespace(MiniMaxH3=FakeBase)
PATCHER_MOD = SimpleNamespace(ModelPatcher=FakePatcher)


def reset():
    FakeConfig.created = []
    FakeBase.instances = 0
    FakeBase.big_model_creations = 0
    FakeBase.last = None
    FakeSigmaShift.calls = []
    TupleSigmaShift.calls = []


def make_kwargs(torch_mod, **overrides):
    kwargs = dict(
        torch_mod=torch_mod,
        supported_models_mod=SUPPORTED,
        model_base_mod=BASE_MOD,
        model_patcher_mod=PATCHER_MOD,
        sigma_shift_node=FakeSigmaShift,
        load_device=LOAD,
        offload_device=OFFLOAD,
        inference_dtype=BF16,
    )
    kwargs.update(overrides)
    return kwargs


class H3PatcherTests(unittest.TestCase):
    def test_metadata_only_wrap_and_shared_identity(self):
        reset()
        torch_mod = ExplodingTorch()
        skeleton = FakeSkeleton(BF16)
        input_config = {
            "image_model": "minimax_h3",
            "keep": 7,
            "disable_unet_model_creation": False,
        }
        original = dict(input_config)

        pair = wrap_h3_skeleton(
            skeleton,
            input_config,
            **make_kwargs(torch_mod, quant_config={"bits": 4}),
        )

        self.assertIsInstance(pair, H3PatcherPair)
        unet, model_shifted = pair
        self.assertIs(unet, pair.unet)
        self.assertIs(model_shifted, pair.model_shifted)
        self.assertIsNot(unet, model_shifted)
        self.assertIs(unet.model, model_shifted.model)
        self.assertIs(unet.model.diffusion_model, skeleton)
        self.assertIs(model_shifted.model.diffusion_model, skeleton)
        self.assertIs(unet.load_device, LOAD)
        self.assertIs(unet.offload_device, OFFLOAD)
        self.assertIs(model_shifted.load_device, LOAD)
        self.assertIs(model_shifted.offload_device, OFFLOAD)

        self.assertEqual(FakeBase.instances, 1)
        self.assertEqual(FakeBase.big_model_creations, 0)
        self.assertEqual(input_config, original)
        self.assertIsNot(FakeConfig.created[0], input_config)
        self.assertIs(FakeConfig.created[0]["disable_unet_model_creation"], True)

        config = unet.model.model_config
        self.assertEqual(config.set_calls, [(BF16, None)])
        self.assertIs(config.unet_config["dtype"], BF16)
        self.assertIs(config.manual_cast_dtype, None)
        self.assertEqual(config.quant_config, {"bits": 4})
        self.assertNotIn("disable_unet_model_creation", config.unet_config)

        self.assertGreaterEqual(skeleton.eval_calls, 1)
        self.assertEqual(skeleton.grad_calls, [False])
        self.assertGreaterEqual(unet.model.eval_calls, 1)

        self.assertEqual(len(FakeSigmaShift.calls), 1)
        called, video, audio = FakeSigmaShift.calls[0]
        self.assertIs(called, unet)
        self.assertEqual(video, H3_VIDEO_SHIFT)
        self.assertEqual(audio, H3_AUDIO_SHIFT)
        options = model_shifted.model_options["transformer_options"]
        self.assertEqual(options["minimax_h3_sigma_shift_video"], 12.0)
        self.assertEqual(options["minimax_h3_sigma_shift_audio"], 3.0)
        self.assertEqual(torch_mod.accessed, [])

    def test_manual_cast_is_passed_through(self):
        reset()
        pair = wrap_h3_skeleton(
            FakeSkeleton(BF16),
            {"image_model": "minimax_h3"},
            **make_kwargs(ExplodingTorch(), manual_cast_dtype=FP32),
        )
        config = pair.unet.model.model_config
        self.assertEqual(config.set_calls, [(BF16, FP32)])
        self.assertIs(config.manual_cast_dtype, FP32)
        self.assertIs(pair.unet.model.manual_cast_dtype, FP32)

    def test_tuple_node_output_is_accepted(self):
        reset()
        pair = wrap_h3_skeleton(
            FakeSkeleton(BF16),
            {"image_model": "minimax_h3"},
            **make_kwargs(ExplodingTorch(), sigma_shift_node=TupleSigmaShift),
        )
        self.assertIs(pair.model_shifted.model, pair.unet.model)
        self.assertEqual(len(TupleSigmaShift.calls), 1)
        self.assertEqual(TupleSigmaShift.calls[0][1:], (12.0, 3.0))

    def test_dtype_mismatch_fails_before_any_config_or_model(self):
        reset()
        input_config = {"image_model": "minimax_h3", "keep": 9}
        original = dict(input_config)
        with self.assertRaises(ValueError):
            wrap_h3_skeleton(
                FakeSkeleton(FP32),
                input_config,
                **make_kwargs(ExplodingTorch(), inference_dtype=BF16),
            )
        self.assertEqual(FakeConfig.created, [])
        self.assertEqual(FakeBase.instances, 0)
        self.assertEqual(input_config, original)

    def test_missing_skeleton_dtype_fails(self):
        reset()
        with self.assertRaises(ValueError):
            wrap_h3_skeleton(
                object(),
                {"image_model": "minimax_h3"},
                **make_kwargs(ExplodingTorch()),
            )
        self.assertEqual(FakeConfig.created, [])
        self.assertEqual(FakeBase.instances, 0)

    def test_constructor_exception_does_not_pollute_input_config(self):
        reset()

        class BoomBase:
            calls = 0

            def __init__(self, model_config, model_type=None, device=None):
                type(self).calls += 1
                if not model_config.unet_config.get(
                    "disable_unet_model_creation", False
                ):
                    raise AssertionError("private disable flag 未生效")
                raise RuntimeError("boom")

        input_config = {"image_model": "minimax_h3", "keep": 11}
        original = dict(input_config)
        with self.assertRaises(RuntimeError):
            wrap_h3_skeleton(
                FakeSkeleton(BF16),
                input_config,
                **make_kwargs(
                    ExplodingTorch(),
                    model_base_mod=SimpleNamespace(MiniMaxH3=BoomBase),
                ),
            )
        self.assertEqual(BoomBase.calls, 1)
        self.assertEqual(input_config, original)
        self.assertNotIn("disable_unet_model_creation", input_config)

    def test_invalidate_patcher_sizes_and_caller_memory(self):
        reset()
        pair = wrap_h3_skeleton(
            FakeSkeleton(BF16),
            {"image_model": "minimax_h3"},
            **make_kwargs(ExplodingTorch(), quant_config={"bits": 4}),
        )
        unet, shifted = pair
        base = unet.model
        base.model_loaded_weight_memory = 123
        unet.size = 999
        shifted.size = 999

        touched = invalidate_patcher_sizes(
            unet, shifted, model_loaded_weight_memory=4096
        )
        self.assertEqual(touched, (unet, shifted))
        self.assertEqual(unet.size, 0)
        self.assertEqual(shifted.size, 0)
        self.assertEqual(base.model_loaded_weight_memory, 4096)

        # 不传字节账时不改动现有值，只清 size。
        unet.size = 321
        shifted.size = 654
        invalidate_patcher_sizes(unet)
        self.assertEqual(unet.size, 0)
        self.assertEqual(shifted.size, 654)
        self.assertEqual(base.model_loaded_weight_memory, 4096)

        unet.size = 123
        with self.assertRaises(ValueError):
            invalidate_patcher_sizes(unet, model_loaded_weight_memory=-1)
        self.assertEqual(unet.size, 123)

    def test_invalidate_without_patchers(self):
        self.assertEqual(invalidate_patcher_sizes(), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
