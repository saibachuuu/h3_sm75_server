"""Star7 VEDA 移植 bundle 兼容层。

远端 D 盘的 bundle（minimax_h3_r2va_veda_preview_fp8.safetensors）实测
（_bundle_probe.py 纯 stdlib 解析）是 Miowtion 标准 predictor 格式：
  format = miowtion-veda-predictor-v1   ← star7 原生，一致
  num_layers=50 num_heads=56 head_dim=128 dtype=float8_e4m3fn
  200 张量 = 100x layers.{i}.proj_{q,k}[56,384,128] + 100x .__scale[56]
  plans JSON 12 个（16x9_t102 / 1x1_t* / 4x3_t* ...）
与 star7 `load_bundle` 的期望**逐项一致**，只差一处：

  **没有 `keep_ratio` 元数据**（训练默认保留率），而是双预算绝对块数：
    target_budget_kind=tiles / target_budget_value=32.0
    ref_budget_kind=tiles     / ref_budget_value=32.0
  ——即"每个 query tile 保留 32 个 target key tile / 32 个 reference key tile"。
  这正是 star7 `selection.Budget(tiles=...)` 的语义，可无损映射。

而 star7 的 `load_bundle` 第 100 行硬读 `metadata['keep_ratio']`，缺键直接
BundleError —— 2026-10-11 线上首启正是炸在这里（DiT 加载 164s 之后）。

本模块的处理：
  * metadata 有 keep_ratio      -> 完全走 star7 原生 load_bundle（零分叉）；
  * metadata 缺 keep_ratio      -> `_load_lenient()`：照抄 star7 的张量处理
    （fp8 + per-head __scale 反量化、plans 严格校验），keep_ratio 填占位值。
    占位安全性已 grep 确认：core 里只有 `PredictorBundle.describe()` 显示用
    keep_ratio，engine/plans/selection/predictor/h3_layout 都不用它；
    真实预算一律走 budget_for() 读 tiles 元数据。
    ※ `_load_lenient` 与 `core/bundle.py:load_bundle` 的张量段必须同步，
      上游改动时一起改。

budget 映射（budget_for）优先级：
  target/ref_budget_value(tiles) > keep_ratio 元数据 > 显式 fallback_keep
"""
from __future__ import annotations

import dataclasses
import json
import os

import torch
from safetensors import safe_open

from veda.core import bundle as veda_bundle
from veda.core import plans as veda_plans
from veda.core import selection as veda_selection

_SCALE_SUFFIX = veda_bundle._SCALE_SUFFIX      # '.__scale'（与上游同源）
_STORAGE = dict(veda_bundle._STORAGE)          # dtype 名 -> torch dtype


def _load_lenient(path: str, metadata: dict) -> veda_bundle.PredictorBundle:
    """缺 `keep_ratio` 元数据时的宽松加载（r2va finetune bundle）。

    张量处理与 `core/bundle.py:load_bundle` 逐行同源：fp8 + per-head __scale
    反量化、shape 严格校验、多余张量报错、plans 逐条校验层数/头数。
    差别只有两点：keep_ratio 用占位值；错误消息带"（宽松加载）"。
    """

    def _err(msg):
        return veda_bundle.BundleError(
            "%s: %s（bundle_compat 宽松加载）" % (os.path.basename(path), msg))

    try:
        num_layers = int(metadata["num_layers"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        plan_json = json.loads(metadata["plans"])
    except (KeyError, ValueError) as error:
        raise _err("incomplete metadata (%s)" % (error,)) from error
    stored = metadata.get("dtype", veda_bundle._LEGACY_DTYPE)
    if stored not in _STORAGE:
        raise _err("unknown storage dtype %r" % (stored,))
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    shape = (num_heads, 3 * head_dim, head_dim)
    proj_q, proj_k = [], []
    for layer in range(num_layers):
        for name, out in (("proj_q", proj_q), ("proj_k", proj_k)):
            key = "layers.%d.%s" % (layer, name)
            value = tensors.pop(key, None)
            if value is None or tuple(value.shape) != shape:
                raise _err("%s missing or not %s" % (key, list(shape)))
            if stored == "float8_e4m3fn":
                scale = tensors.pop(key + _SCALE_SUFFIX, None)
                if scale is None or tuple(scale.shape) != (num_heads,):
                    raise _err("%s has no per-head fp8 scale" % (key,))
                value = value.float() * scale.float().view(-1, 1, 1)
            out.append(value.to(torch.bfloat16).contiguous())
    if tensors:
        raise _err("unexpected tensors %s" % (sorted(tensors)[:4],))
    table = veda_plans.PlanTable(
        [veda_plans.TilePlan.from_json(p) for p in plan_json.values()])
    for plan in table.plans.values():
        if plan.num_layers != num_layers or plan.num_heads != num_heads:
            raise _err("plan %s is %dx%d, predictor is %dx%d"
                       % (plan.name, plan.num_layers, plan.num_heads,
                          num_layers, num_heads))
    keep_ratio = float(metadata.get("keep_ratio", 0) or 0)
    if not (0.0 < keep_ratio <= 1.0):
        keep_ratio = 0.1     # 占位：只影响 describe() 显示；真实预算见 budget_for()
    return veda_bundle.PredictorBundle(
        path=path, num_layers=num_layers, num_heads=num_heads,
        head_dim=head_dim, keep_ratio=keep_ratio, plans=table,
        proj_q=proj_q, proj_k=proj_k, metadata=metadata)


def _budget_from_metadata(metadata: dict) -> veda_selection.Budget | None:
    """从 target budget 元数据构造 Budget（tiles 绝对块数）；缺失或不认返回 None。"""
    kind = metadata.get("target_budget_kind")
    value = metadata.get("target_budget_value")
    if kind == "tiles" and value is not None:
        try:
            tiles = float(value)
        except (TypeError, ValueError):
            return None
        if tiles > 0:
            return veda_selection.Budget(tiles=tiles)
    return None


def _ratio_budget(metadata: dict, fallback_keep: float | None
                  ) -> veda_selection.Budget | None:
    """keep_ratio 元数据 -> ratio Budget；否则用调用方给的 fallback_keep。"""
    keep = None
    if "keep_ratio" in metadata:
        try:
            keep = float(metadata["keep_ratio"])
        except (TypeError, ValueError):
            keep = None
    if keep is None and fallback_keep:
        keep = float(fallback_keep)
    if keep is None or not (0.0 < keep <= 1.0):
        return None
    return veda_selection.Budget(ratio=keep)


def budget_for(bundle: veda_bundle.PredictorBundle,
               fallback_keep: float | None = None
               ) -> tuple[veda_selection.Budget, veda_selection.Budget, str]:
    """(generated, reference, 说明)。

    优先绝对块数（r2va finetune bundle 的 target/ref_budget_value），
    否则退回 keep ratio。
    """
    metadata = bundle.metadata or {}
    generated = _budget_from_metadata(metadata)
    kind = metadata.get("ref_budget_kind")
    value = metadata.get("ref_budget_value")
    reference = None
    if kind == "tiles" and value is not None:
        try:
            tiles = float(value)
        except (TypeError, ValueError):
            tiles = 0.0
        if tiles > 0:
            reference = veda_selection.Budget(tiles=tiles)
    how = "target/ref_budget_value (tiles)"
    if generated is None:
        generated = _ratio_budget(metadata, fallback_keep)
        how = "keep_ratio"
    if generated is None:
        raise ValueError(
            "bundle 既无 target_budget_kind=tiles 也无合法 keep_ratio；"
            "budget 元数据=%r" % (sorted(metadata),))
    if reference is None:
        reference = generated
    return generated, reference, how


def load_bundle_lenient(path: str) -> veda_bundle.PredictorBundle:
    """先试 star7 原生加载；缺 keep_ratio 时走宽松路径。

    只对"元数据缺 keep_ratio"这一类差异降级，其它 BundleError 原样抛出
    （format 不对、张量 shape 不对等必须失败）。
    """
    metadata = veda_bundle.read_metadata(path)   # format 校验（star7 函数）
    if "keep_ratio" in metadata:
        return veda_bundle.load_bundle(path)
    return _load_lenient(path, metadata)


@dataclasses.dataclass
class LoadedBundle:
    """加载结果：star7 bundle + 派生预算。"""

    bundle: veda_bundle.PredictorBundle
    generated: veda_selection.Budget
    reference: veda_selection.Budget
    budget_source: str

    @property
    def num_layers(self) -> int:
        return self.bundle.num_layers

    @property
    def num_heads(self) -> int:
        return self.bundle.num_heads

    @property
    def head_dim(self) -> int:
        return self.bundle.head_dim

    def describe(self) -> str:
        b = self.bundle
        return ("%s | %dL x %dH x d%d | budget %s: generated %s, reference %s"
                % (b.path.split("\\")[-1].split("/")[-1], b.num_layers,
                   b.num_heads, b.head_dim, self.budget_source,
                   veda_selection.Budget.describe(self.generated),
                   veda_selection.Budget.describe(self.reference)))


def load(path: str, fallback_keep: float | None = None) -> LoadedBundle:
    """加载 bundle 并按元数据推导 generated/reference 预算。"""
    bundle = load_bundle_lenient(path)
    generated, reference, how = budget_for(bundle, fallback_keep)
    return LoadedBundle(bundle=bundle, generated=generated,
                        reference=reference, budget_source=how)
