"""h3_sm75_server 回归自检（从 tests/ 目录直接跑）。

跑法（需要有 torch 的解释器；本地可用 ~/comfyui/.venv）：
  cd h3_sm75_server/tests && python _selftest_star7_port.py

背景：2026-10-11 的两次线上启动失败（self._install_attention
找不到 / bundle 缺 keep_ratio）都是"静态 import+compileall
挡不住、真执行才炸"的类型，所以这些脚本必须真实执行到关键
路径，不能只 import。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "comfyui"))

os.chdir(os.path.join(os.path.expanduser("~"), "comfyui"))

fails = []


def check(name, fn):
    try:
        v = fn()
        print("  OK   %-44s %s" % (name, v))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-44s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


print("== A. vendored veda package ==")
check("import veda", lambda: __import__("veda").__version__)
check("veda.core.engine.VedaEngine",
      lambda: __import__("veda.core.engine", fromlist=["x"]).VedaEngine.__name__)
check("veda.core.tiling.TILE_SIZE",
      lambda: __import__("veda.core.tiling", fromlist=["x"]).TILE_SIZE)
check("veda.core.selection.Budget",
      lambda: __import__("veda.core.selection", fromlist=["x"]).Budget.tiles)
check("veda.backends.torch_gather",
      lambda: __import__("veda.backends.torch_gather", fromlist=["x"]).TorchGatherBackend.name)
check("veda.kernels.native runtime.Kernel",
      lambda: __import__("veda.kernels.native.runtime", fromlist=["x"]).Kernel.__name__)
check("veda.kernels.native ROOT",
      lambda: str(__import__("veda.kernels.native.runtime", fromlist=["x"]).ROOT))
check("veda.comfy_patch.apply",
      lambda: __import__("veda.comfy_patch", fromlist=["x"]).apply.__name__)
check("veda.settings.VedaSettings",
      lambda: __import__("veda.settings", fromlist=["x"]).VedaSettings.__name__)
check("veda.hardware.describe",
      lambda: __import__("veda.hardware", fromlist=["x"]).describe.__name__)
check("veda.bundle_compat.load",
      lambda: __import__("veda.bundle_compat", fromlist=["x"]).load.__name__)

print("== B. backends.candidates on sm75 (device patched) ==")


class _Info:
    kind = "cuda"
    family = "cuda"
    cc = (7, 5)


check("candidates(sm75)",
      lambda: __import__("veda.backends", fromlist=["candidates"]).candidates(_Info()))


class _Info80(_Info):
    cc = (8, 9)


check("candidates(sm89)",
      lambda: __import__("veda.backends", fromlist=["candidates"]).candidates(_Info80()))

print("== C. bridge settings mapping ==")


class A:
    veda_keep_tiles = 0
    veda_ref_keep_tiles = -1
    veda_keep = 0.0
    veda_dense_layers = ""
    veda_dense_steps = ""
    veda_debug = False


def desc(a):
    from dit_svc import veda_star7
    s = veda_star7.settings_from_args(a)
    return "gen=%s ref=%s L=%s S=%s verbose=%s" % (
        s.generated.describe(), s.reference.describe(),
        sorted(s.dense_layers), sorted(s.dense_steps), s.verbose)


check("default", lambda: desc(A()))


class A2(A):
    veda_keep_tiles = 32


class A3(A):
    veda_dense_layers = "0,1,49"
    veda_dense_steps = "0"


class A4(A):
    veda_keep = 0.15


check("tiles=32", lambda: desc(A2()))
check("dense L/S", lambda: desc(A3()))
check("keep=0.15", lambda: desc(A4()))

print("== D. 探针契约（正版 veda 工况） ==")
import importlib  # noqa: E402
vs7 = importlib.import_module("dit_svc.veda_attn_star7")

check("status_report(None)", lambda: vs7.status_report(None))
check("WARMUP_TIERS_VEDA", lambda: vs7.WARMUP_TIERS_VEDA)


class _Stats:
    def kept_fraction(self):
        return 0.12

    def compute_fraction(self):
        return 0.15


class _Backend:
    display = "Star7 VEDA INT8 Tensor Core (SM75)"


class _Engine:
    backend = _Backend()
    stats = _Stats()
    chunking = {}


class _Run:
    def __init__(self):
        self.calls = {"sparse": 50, "other attention": 6}
        self.evaluations = 1
        self.failed = None
        self.video = "1344x768"


class _FakePatch:
    def __init__(self):
        self.run = _Run()
        self._engines = {"cuda:0": _Engine()}
        self.cleanup_calls = 0

    def on_cleanup(self):
        """真实 star7 行为：清零 run。"""
        self.cleanup_calls += 1
        self.run = _Run()


vs7.reset_stats()
fake = _FakePatch()


def _desc_snap(s):
    return "sparse=%s dense=%s compute=%s backend=%s" % (
        s.get("sparse"), s.get("dense"), s.get("compute_fraction"),
        s.get("backend"))


def _desc_status(r):
    c = r["cumulative"]
    return "active=%s backend=%s runs=%d sparse=%d compute_last=%s" % (
        r["active"], r.get("backend"), c["runs"], c["sparse"],
        c.get("compute_fraction_last"))


check("run_snapshot(非空 patch)", lambda: _desc_snap(vs7.run_snapshot(fake)))

check("wrap_cleanup", lambda: vs7.wrap_cleanup(fake, log=lambda s: None))

# 模拟 comfy 采样结束触发 cleanup：先抓累计，再被清零
def _run_and_cleanup():
    fake.on_cleanup()
    return (fake.cleanup_calls, dict(vs7.CUMULATIVE["reasons"]),
            vs7.CUMULATIVE["sparse"], vs7.CUMULATIVE["runs"])


check("cleanup 抓取+清零", lambda: _run_and_cleanup())
check("累计后 run 已清零", lambda: (fake.run.calls, vs7.CUMULATIVE["sparse_per_run"]))
check("status_report(有 patch)", lambda: _desc_status(vs7.status_report(fake)))

check("install_logger", lambda: vs7.install_logger(log=lambda s: None))

print()
if fails:
    print("FAILED: %s" % fails)
    raise SystemExit(1)
print("ALL OK")
