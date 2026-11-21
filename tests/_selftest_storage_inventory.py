"""自检 P0：真实 storage inventory（views 去重 / buffers / 量化内部 / dict 投影）。

需要 torch（本机 2.8.0+cpu）。只用 CPU 张量；CUDA 相关数字在本机
无法验证，见 docs/p0_weights.md「未验证事项」。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

fails = []


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("returned False")
        print("  OK   %-46s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-46s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


import torch  # noqa: E402
from torch import nn  # noqa: E402

from dit_svc import weights as W  # noqa: E402
from dit_svc.model_adapters import FakeQuantTensor  # noqa: E402


def t_views_dedup():
    base = torch.randn(8, 8)  # 64 * 4 = 256 B
    v1 = base.view(-1)
    v2 = base[2:6]
    inv = W.storage_inventory({"a": base, "b": v1, "c": v2}, owner="u")
    s = inv.to_summary()
    assert s["num_storages"] == 1, s
    assert s["total_cpu"] == 256, s
    assert s["shared"] == 1, s
    return "1 storage, %d B" % s["total_cpu"]


def t_module_buffers_and_params():
    m = nn.Linear(16, 16, bias=True)
    m2 = nn.Linear(16, 16, bias=True)
    inv = W.storage_inventory({"m": m, "m2": m2}, owner="u")
    s = inv.to_summary()
    # Linear(16,16): weight 256*4=1024 + bias 16*4=64 = 1088/unit
    assert s["total_cpu"] == 1088 * 2, s
    assert s["num_storages"] == 4, s
    return "module walk %d B / %d storages" % (s["total_cpu"], s["num_storages"])


def t_quant_internals_expanded():
    qdata = torch.randint(0, 100, (32,), dtype=torch.int8)   # 32 B
    scale = torch.ones(4, dtype=torch.float32)               # 16 B
    q = FakeQuantTensor(qdata, scale)
    plain = torch.randn(4)  # 16 B
    inv = W.storage_inventory({"q": q, "p": plain}, owner="u")
    s = inv.to_summary()
    assert s["total_cpu"] == 32 + 16 + 16, s
    assert s["num_storages"] == 3, s
    kinds = sorted(o for e in s["entries"] for o in e["owners"])
    assert any("_qdata" in k for k in kinds), kinds
    return "quant expanded %d B" % s["total_cpu"]


def t_dict_projection_and_alias_shared_group():
    proj_w = torch.randn(8, 8)  # 256 B
    proj = {"weight": proj_w, "alias": proj_w}  # 同 storage 别名
    other = torch.randn(8, 8)  # 256 B
    inv = W.storage_inventory({"proj": proj, "other": other}, owner="te.proj")
    s = inv.to_summary()
    assert s["total_cpu"] == 512, s
    assert s["num_storages"] == 2, s
    return "alias dedup %d B" % s["total_cpu"]


def t_meta_has_no_storage():
    with torch.device("meta"):
        t = torch.randn(8, 8)
    inv = W.storage_inventory({"m": t}, owner="u")
    s = inv.to_summary()
    assert s["total_cpu"] == 0 and s["meta_tensors"] == 1, s
    assert s["uninspectable"] == [], s
    return "meta counted, no bytes"


class _RealQuantWrapper(torch.Tensor):
    """真实 Tensor-subclass 量化 fixture（对标 comfy-kitchen QuantizedTensor）。

    wrapper 自身无 storage（untyped_storage 直接抛）；
    物理账只能从内部 _qdata（I8）+ _scale（F32 per-row）来。
    逻辑 dtype/shape（F32/[N]）不得用于字节账。
    """

    @staticmethod
    def __new__(cls, qdata, scale):
        return torch.Tensor._make_wrapper_subclass(
            cls, qdata.shape, dtype=torch.float32, requires_grad=False)

    def __init__(self, qdata, scale):
        self._qdata = qdata
        self._scale = scale

    def __tensor_flatten__(self):
        return (["_qdata", "_scale"], None)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError("fixture dispatch: %s" % func)

    def __repr__(self):
        return "_RealQuantWrapper()"


def t_real_subclass_wrapper_not_meta():
    # 远端 probe 20 真实形态：wrapper device cpu、非 meta，但自身无 storage。
    # 旧 walk 误计 meta_tensors=1 且漏内部 385MB 量级账；新 walk 必须展开。
    qdata = torch.randint(0, 100, (32,), dtype=torch.int8)  # 32 B
    scale = torch.ones(4, dtype=torch.float32)             # 16 B
    w = _RealQuantWrapper(qdata, scale)
    assert w.device.type == "cpu" and not w.is_meta
    assert w.dtype == torch.float32  # 逻辑 dtype：8*4=32B，不得用于账
    inv = W.storage_inventory({"w": w}, owner="u")
    s = inv.to_summary()
    assert s["meta_tensors"] == 0, s  # wrapper 自身绝不计 meta
    assert s["uninspectable"] == [], s
    assert s["total_cpu"] == 32 + 16, s  # 物理账，非逻辑 32B
    assert s["num_storages"] == 2, s
    kinds = sorted(o for e in s["entries"] for o in e["owners"])
    assert any("_qdata" in k for k in kinds), kinds
    assert any("_scale" in k for k in kinds), kinds
    return "wrapper transparent: %d B, meta=0" % s["total_cpu"]


def t_wrapper_view_dedup_and_mixed():
    qdata = torch.randint(0, 100, (32,), dtype=torch.int8)
    scale = torch.ones(4, dtype=torch.float32)
    w = _RealQuantWrapper(qdata, scale)
    v = qdata.view(4, 8)  # 同 storage 视图
    inv = W.storage_inventory({"w": w, "v": v}, owner="u")
    s = inv.to_summary()
    assert s["total_cpu"] == 32 + 16, s  # qdata 只计一次
    assert s["num_storages"] == 2, s
    # mixed：meta qdata + CPU scale -> scale 实账 + meta 计 1，wrapper 仍透明
    mq = torch.empty(32, dtype=torch.int8, device="meta")
    w2 = _RealQuantWrapper(mq, torch.ones(4))
    inv2 = W.storage_inventory({"w": w2}, owner="u")
    s2 = inv2.to_summary()
    assert s2["total_cpu"] == 16, s2
    assert s2["meta_tensors"] == 1, s2  # 真 meta 的底层 qdata
    assert s2["uninspectable"] == [], s2
    # purge  analog：全 meta 内部 -> 零账
    w3 = _RealQuantWrapper(torch.empty(32, dtype=torch.int8, device="meta"),
                           torch.empty(4, device="meta"))
    s3 = W.storage_inventory({"w": w3}, owner="u").to_summary()
    assert s3["total_cpu"] == 0 and s3["meta_tensors"] == 2, s3
    assert s3["uninspectable"] == [], s3
    return "dedup+mixed+purge ok"


def t_uninspectable_fail_closed():
    class Opaque(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_wrapper_subclass(
                cls, (4,), dtype=torch.float32, requires_grad=False)

        def __init__(self):
            pass  # 无内部张量、无 storage

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise NotImplementedError("opaque")

    o = Opaque()
    inv = W.storage_inventory({"o": o}, owner="u")
    s = inv.to_summary()
    assert s["meta_tensors"] == 0, s  # 非 meta 不得谎报 meta
    assert len(s["uninspectable"]) == 1, s
    return "opaque -> uninspectable, meta=0"


print("== _selftest_storage_inventory ==")
check("views share one storage", t_views_dedup)
check("module params+buffers walked", t_module_buffers_and_params)
check("quant internals expanded", t_quant_internals_expanded)
check("dict projection alias dedup", t_dict_projection_and_alias_shared_group)
check("meta tensor has no storage", t_meta_has_no_storage)
check("real subclass wrapper transparent", t_real_subclass_wrapper_not_meta)
check("wrapper dedup + mixed + purge", t_wrapper_view_dedup_and_mixed)
check("uninspectable fail-closed", t_uninspectable_fail_closed)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
