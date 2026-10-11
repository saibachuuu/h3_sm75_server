"""h3_sm75_server 回归自检（从 tests/ 目录直接跑）。

跑法（需要有 torch 的解释器；本地可用 ~/comfyui/.venv）：
  cd h3_sm75_server/tests && python _selftest_bundle_compat.py

背景：2026-10-11 的两次线上启动失败（self._install_attention
找不到 / bundle 缺 keep_ratio）都是"静态 import+compileall
挡不住、真执行才炸"的类型，所以这些脚本必须真实执行到关键
路径，不能只 import。
"""

import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/comfyui"))
os.chdir(os.path.expanduser("~/comfyui"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from veda import bundle_compat  # noqa: E402
from veda.core import bundle as veda_bundle  # noqa: E402

LAYERS, HEADS, DIM = 2, 4, 8
SHAPE = (HEADS, 3 * DIM, DIM)
fails = []


def check(name, fn):
    """跑一个检查项；返回 False / 抛错都算失败（教训：bool 结果必须断言）。"""
    try:
        value = fn()
        if value is False:
            raise AssertionError("返回 False")
        print("  OK   %-42s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-42s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


def make_bundle(tmp, *, format_="miowtion-veda-predictor-v1", dtype="float32",
                keep_ratio=None, target_tiles=None, ref_tiles=None):
    """写一个微型 bundle 并返回路径。"""
    meta = {"format": format_, "num_layers": str(LAYERS),
            "num_heads": str(HEADS), "head_dim": str(DIM), "dtype": dtype,
            "step": "1"}
    if keep_ratio is not None:
        meta["keep_ratio"] = keep_ratio
    if target_tiles is not None:
        meta["target_budget_kind"] = "tiles"
        meta["target_budget_value"] = str(target_tiles)
    if ref_tiles is not None:
        meta["ref_budget_kind"] = "tiles"
        meta["ref_budget_value"] = str(ref_tiles)
    plan = {"geometry": "1x1_t2", "grid": [2, 2, 2],
            "shapes": ["1x8x16"],                       # t*h*w 必须 = 128
            "head_shape": [[0] * HEADS] * LAYERS}
    meta["plans"] = json.dumps({"1x1_t2": plan})

    g = torch.Generator().manual_seed(7)
    tensors = {}
    expect = {}
    for i in range(LAYERS):
        for name in ("proj_q", "proj_k"):
            key = "layers.%d.%s" % (i, name)
            if dtype == "float8_e4m3fn":
                raw = torch.rand(SHAPE, generator=g) * 4
                scale = torch.rand(HEADS, generator=g) + 0.5
                stored = raw.to(torch.float8_e4m3fn)   # 存储即截断
                tensors[key] = stored
                tensors[key + ".__scale"] = scale
                # 期望 = 加载路径同款运算：stored.float() * scale -> bf16
                expect[key] = (stored.float() * scale.view(-1, 1, 1)).to(torch.bfloat16)
            else:
                raw = torch.rand(SHAPE, generator=g)
                tensors[key] = raw
                expect[key] = raw.to(torch.bfloat16)
    path = os.path.join(tmp, "synthetic.safetensors")
    save_file(tensors, path, metadata=meta)
    return path, expect


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="veda_bundle_test_")
    print("== 合成 bundle 测试（%s）==" % tmp)

    print("1) r2va 风格（缺 keep_ratio / tiles 预算 / F32）")
    path, expect = make_bundle(tmp, target_tiles=32, ref_tiles=16)
    check("load 不抛错", lambda: bundle_compat.load_bundle_lenient(path).num_layers)
    loaded = bundle_compat.load(path)
    check("budget 来源", lambda: loaded.budget_source)
    check("generated tiles", lambda: loaded.generated.tiles)
    check("reference tiles", lambda: loaded.reference.tiles)
    check("proj_q[0] 形状", lambda: tuple(loaded.bundle.proj_q[0].shape))
    check("proj_q[0] 数值=raw.bf16",
          lambda: bool(torch.equal(loaded.bundle.proj_q[0],
                                   expect["layers.0.proj_q"])))
    check("proj_k[1] 数值=raw.bf16",
          lambda: bool(torch.equal(loaded.bundle.proj_k[1],
                                   expect["layers.1.proj_k"])))
    check("describe 含 budget",
          lambda: "tiles" in loaded.describe())

    print("2) r2va 风格 fp8（__scale 反量化）")
    path, expect = make_bundle(tmp, dtype="float8_e4m3fn", target_tiles=32)
    loaded = bundle_compat.load(path)
    check("fp8 proj_q[0] 反量化数值",
          lambda: bool(torch.equal(loaded.bundle.proj_q[0],
                                   expect["layers.0.proj_q"])))
    check("fp8 proj_k[1] 反量化数值",
          lambda: bool(torch.equal(loaded.bundle.proj_k[1],
                                   expect["layers.1.proj_k"])))
    check("reference 跟随 generated", lambda: loaded.reference.tiles)

    print("3) 原生风格（有 keep_ratio，无 tiles）")
    path, _ = make_bundle(tmp, keep_ratio="0.1")
    loaded = bundle_compat.load(path)
    check("走原生路径", lambda: loaded.budget_source)
    check("generated ratio", lambda: round(loaded.generated.ratio, 4))

    print("4) format 不对必须失败")
    path, _ = make_bundle(tmp, format_="some-other-format")
    check("BundleError", lambda: _expect_bundle_error(path))

    print()
    if fails:
        print("FAILED: %s" % fails)
        return 1
    print("ALL OK")
    return 0


def _expect_bundle_error(path):
    try:
        bundle_compat.load_bundle_lenient(path)
    except veda_bundle.BundleError as e:
        return "BundleError: %s" % str(e)[:60]
    raise AssertionError("没有抛 BundleError")


if __name__ == "__main__":
    raise SystemExit(main())
