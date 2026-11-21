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
    return "meta counted, no bytes"


print("== _selftest_storage_inventory ==")
check("views share one storage", t_views_dedup)
check("module params+buffers walked", t_module_buffers_and_params)
check("quant internals expanded", t_quant_internals_expanded)
check("dict projection alias dedup", t_dict_projection_and_alias_shared_group)
check("meta tensor has no storage", t_meta_has_no_storage)

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
