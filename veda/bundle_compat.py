"""Star7 VEDA 移植 bundle 兼容层。

远端 D 盘的 bundle（minimax_h3_r2va_veda_preview_fp8.safetensors）是
Miowtion 标准 predictor 格式（format=miowtion-veda-predictor-v1，
fp8 + per-head __scale，layers.{i}.proj_{q,k}[56,384,128] + __scale[56]，
plans 12 个 tile plan），但与 star7 的原生 `load_bundle` 有三处差异：

  1. 没有 `keep_ratio` 元数据（训练默认保留率），而是双预算绝对块数：
     `target_budget_kind=tiles / target_budget_value=32.0`、
     `ref_budget_kind=tiles / ref_budget_value=32.0`
     ——即"每个 query tile 保留 32 个 target key tile / 32 个 reference key tile"。
     这正是 star7 `selection.Budget(tiles=...)` 的语义，故可无损映射。
  2. `plans` 的 JSON 是 {geometry: {...}} 的 dict（star7 直接吃 list(plans.values())）。✓
  3. `format` 键与 star7 FORMAT 常量一致。✓

因此这里只补 (1)：从 budget 元数据推导 Budget，其余完全走 star7 原生加载路径。
budget 元数据缺失时退回 ratio 预算（keep_ratio 元数据或显式传入值）。
"""
from __future__ import annotations

import dataclasses

from veda.core import bundle as veda_bundle
from veda.core import selection as veda_selection


def _budget_from_metadata(metadata: dict) -> veda_selection.Budget | None:
    """从 target/ref budget 元数据构造 Budget；缺失或不认返回 None。"""
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

    优先绝对块数（r2va finetune bundle 的 target_budget_value/ref_budget_value），
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
    bundle = veda_bundle.load_bundle(path)
    generated, reference, how = budget_for(bundle, fallback_keep)
    return LoadedBundle(bundle=bundle, generated=generated,
                        reference=reference, budget_source=how)
