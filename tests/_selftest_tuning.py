"""自检：veda/server_tuning.py 的调参层（fp32/fp16 切换、chunk_share、恢复）。

CPU 可跑逻辑部分（patch/恢复/校验/报错）；fp16 数值精度部分需要 GPU
（sm89 本地可跑；CPU 上 fp16 bmm 多数构建不支持，自动跳过并说明）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "comfyui"))
os.chdir(os.path.join(os.path.expanduser("~"), "comfyui"))

import torch  # noqa: E402

from veda import server_tuning  # noqa: E402

fails = []


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("返回 False")
        print("  OK   %-44s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-44s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


def _expect_value_error(name):
    try:
        server_tuning.set_tile_precision(name)
    except ValueError as e:
        return "ValueError: %s" % str(e)[:50]
    raise AssertionError("没抛")


def _expect_share_error(backend, value):
    try:
        server_tuning.set_chunk_share(backend, value)
    except ValueError as e:
        return "ValueError: %s" % str(e)[:50]
    raise AssertionError("没抛")


print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")


print("== 1. tile_precision 切换与恢复 ==")
from veda.core import predictor  # noqa: E402
ORIG = predictor.tile_logits

check("初始 fp32 = 上游函数",
      lambda: server_tuning.set_tile_precision("fp32") == "fp32"
      and predictor.tile_logits is ORIG)
check("切 fp16 生效（被替换）",
      lambda: server_tuning.set_tile_precision("fp16") == "fp16"
      and predictor.tile_logits is not ORIG)
check("回 fp32 恢复原函数",
      lambda: server_tuning.set_tile_precision("fp32") == "fp32"
      and predictor.tile_logits is ORIG)
check("无效值抛 ValueError", lambda: _expect_value_error("bf16"))
check("state()", lambda: server_tuning.state())

print("== 2. fp16 tile_logits 输出契约 ==")
server_tuning.set_tile_precision("fp16", log=lambda s: None)


def _contract():
    heads, n_q, n_k, d = 2, 5, 7, 8
    fq = torch.randn(heads, n_q, 3 * d)
    fk = torch.randn(heads, n_k, 3 * d)
    pq = torch.randn(heads, 3 * d, d)
    pk = torch.randn(heads, 3 * d, d)
    out = predictor.tile_logits(fq, fk, pq, pk)
    assert out.dtype == torch.float32, out.dtype
    assert tuple(out.shape) == (heads, n_q, n_k), out.shape
    assert torch.isfinite(out.float()).all()
    ref = ORIG(fq, fk, pq, pk)
    scale = out.float().abs().max().clamp_min(1e-9)
    rel = (out.float() - ref.float()).abs().max() / scale
    return "dtype=fp32 shape=%s 相对误差=%.4f" % (tuple(out.shape), float(rel))


def _contract_gpu():
    heads, n_q, n_k, d = 2, 40, 40, 128
    fq = torch.randn(heads, n_q, 3 * d, device="cuda")
    fk = torch.randn(heads, n_k, 3 * d, device="cuda")
    pq = torch.rand(heads, 3 * d, d, device="cuda", dtype=torch.bfloat16)
    pk = torch.rand(heads, 3 * d, d, device="cuda", dtype=torch.bfloat16)
    lo = predictor.tile_logits(fq, fk, pq, pk)          # fp16 版
    server_tuning.set_tile_precision("fp32", log=lambda s: None)
    hi = predictor.tile_logits(fq, fk, pq, pk)          # 上游 fp32 版
    server_tuning.set_tile_precision("fp16", log=lambda s: None)
    # 排序一致性：每行的 top-k 集合是否一致（打分只用相对排序）
    k = min(8, n_k)
    a = lo.topk(k, dim=-1).indices.sort(-1).values
    b = hi.topk(k, dim=-1).indices.sort(-1).values
    agree = (a == b).float().mean().item()
    return "top%d 排序一致率=%.3f" % (k, agree)


check("fp16 输出 fp32 + 有限（CPU）", _contract)
if torch.cuda.is_available():
    check("fp16 vs fp32 top-k 一致率（GPU）", _contract_gpu)
else:
    print("  SKIP fp16 GPU 数值比对（无 CUDA）")

print("== 3. chunk_share ==")


class _FakeBackend:
    chunk_share = 16


fb = _FakeBackend()
check("设 8 生效", lambda: server_tuning.set_chunk_share(fb, 8) == 8
      and fb.chunk_share == 8)
check("设 1 生效", lambda: server_tuning.set_chunk_share(fb, 1) == 1)
check("backend=None 不炸", lambda: server_tuning.set_chunk_share(None, 4) == 4)
check("无效值抛 ValueError", lambda: _expect_share_error(fb, 0))

print("== 4. configure 一站式 ==")
check("configure(fp16, 8)", lambda: server_tuning.configure(
    "fp16", 8, backend=fb, log=lambda s: None) ==
    {"tile_precision": "fp16", "chunk_share": 8})
check("收尾恢复上游",
      lambda: server_tuning.set_tile_precision("fp32") == "fp32"
      and predictor.tile_logits is ORIG)
