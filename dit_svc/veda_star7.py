"""Star7 VEDA 稀疏 attention 在 h3_sm75_server 的接入层（路线 B 主体）。

与最小 fork（_clone_svdint4 / comfyui_turing_utils veda 适配层）的关系：
  fork 的 Veda 需要 configure() 返回的**整个 patched ModelPatcher**（override +
  layout provider + schedule + config 全装在 clone 上），server 却只从 clone 里
  抠出 override 函数、再手工塞回 self.unet / model_shifted 的 model_options ——
  于是 fork 侧依赖的伴生组件全部丢失，gate 永远 dense（详见
  6acb10c030e522a0cbe68b22_veda_sm75_dense_根因与改造方案.md 的根因一/二/三）。

  star7 的 veda 没有这个问题：它的 gate 只读 comfy core **每次都发布**的
  `transformer_options["minimax_h3_layout"]`（comfy/ldm/minimax/model.py:624）
  与 `transformer_options["block_index"]`（同文件:755），不依赖任何外部
  wrapper / schedule；`apply()` 返回的 clone 上同时带着 override 与
  ON_PREPARE_STATE / ON_CLEANUP 回调，且 comfy 的 `wrap_attn`
  （comfy/ldm/modules/attention.py:200-238）保证 override 优先于
  checkpoint 钉死的 preferred_attention。所以在本 server 里只要把
  `apply()` 返回的 patched 对象**当作模型本体继续用**即可。

本模块职责：
  1. 解析 CLI 参数 -> star7 VedaSettings（sparsity / dense layers / steps）；
  2. 加载 bundle（veda/bundle_compat.py，兼容 r2va finetune 的 tiles 预算）；
  3. 对 ModelPatcher 安装 veda patch，并把 patch 对象登记到 runtime 上；
  4. 提供 probe / status 钩子，让 /api/status 能看 sparse_attend_calls。
"""
from __future__ import annotations

import os
import sys


def _ensure_import_path(here: str) -> None:
    """让 `import veda` 解析到 h3_sm75_server/veda（server 根目录）。"""
    if here not in sys.path:
        sys.path.insert(0, here)


def settings_from_args(args, log=print):
    """CLI -> (VedaSettings, bundle兼容说明)。veda/ 包须已可 import。"""
    import veda.settings as veda_settings
    from veda.core import selection

    # r2va 的 bundle 用"绝对 tiles 预算"训练（target/ref_budget_value=32），
    # 与 --veda-keep 的比例语义天然正交：这里以"每 query tile 保留的 key
    # tile 数"表达，默认 32（=bundle 训练值），0 表示用 bundle 自带预算。
    keep_tiles = int(getattr(args, "veda_keep_tiles", 0) or 0)
    ref_keep_tiles = int(getattr(args, "veda_ref_keep_tiles", -1) or -1)
    keep_ratio = float(getattr(args, "veda_keep", 0.0) or 0.0)

    def _budget(tiles: int, ratio: float) -> selection.Budget:
        if tiles > 0:
            return selection.Budget(tiles=tiles)
        if ratio > 0.0 and ratio <= 1.0:
            return selection.Budget(ratio=ratio)
        return selection.Budget.sparsity(90.0)

    generated = _budget(keep_tiles, keep_ratio)
    reference = _budget(ref_keep_tiles if ref_keep_tiles >= 0 else keep_tiles,
                        keep_ratio)
    dense_layers = _parse_int_list(getattr(args, "veda_dense_layers", "") or "")
    dense_steps = _parse_int_list(getattr(args, "veda_dense_steps", "") or "")
    settings = veda_settings.VedaSettings(
        generated=generated, reference=reference,
        dense_layers=dense_layers, dense_steps=dense_steps,
        verbose=bool(getattr(args, "veda_debug", False)))
    return settings


def _parse_int_list(text: str):
    """"0,1,47-49" -> frozenset({0,1,47,48,49})（0 基，含端点）。"""
    try:
        from veda.settings import parse_index_list
        return parse_index_list(text, "list")
    except Exception:
        return frozenset()


def load_bundle(args, log=print):
    """加载 bundle 并推导 (generated, reference) 预算。返回 LoadedBundle。"""
    from veda import bundle_compat

    path = getattr(args, "veda_bundle", None)
    if not path or not os.path.isfile(path):
        raise RuntimeError("--attn veda 需要 --veda-bundle 指向本地 predictor "
                           "bundle (.safetensors)，未找到: %r" % (path,))
    loaded = bundle_compat.load(path)
    log("[veda] bundle: %s" % loaded.describe())
    return loaded


def check_model_dims(unet, loaded, log=print) -> None:
    """DiT blocks 形状 vs bundle 维度（50L x 56H x d128 才允许装）。"""
    blocks = unet.model.diffusion_model.blocks
    shape = (len(blocks), blocks[0].attn.heads, blocks[0].attn.head_dim)
    want = (loaded.num_layers, loaded.num_heads, loaded.head_dim)
    if shape != want:
        raise RuntimeError(
            "Veda bundle 维度 %r 与当前 DiT %r 不匹配" % (want, shape))
    log("[veda] DiT 形状核对 OK: %dL x %dH x d%d" % shape)


def install(unet, args, torch, log=print, probe_log=None):
    """安装 veda 到 unet（就地装 override + 回调，不替换 unet 本体）。

    返回 (patched, patch, info)。patched 与 unet.model 是同一对象，只是
    model_options 里多了 override 与回调；调用方应继续用 unet（server 的
    runtime 常驻引用、_unhook_dit、_AttnWindowProfiler 全都按原对象工作）。

    失败一律抛异常，由 _install_attention 捕获回退 int8。
    """
    import veda.comfy_patch as veda_comfy_patch

    settings = settings_from_args(args, log=log)
    loaded = load_bundle(args, log=log)
    check_model_dims(unet, loaded, log=log)
    patched, patch = veda_comfy_patch.apply(unet, loaded.bundle, settings, None)
    # apply() 已 clone；但 server 全程用原 unet 干活（常驻、unhook、profiler
    # 都引用它），所以把 clone 上的 patch 就地装回原对象，保持单一对象。
    patch.install(unet.model_options.setdefault("transformer_options", {}))
    info = {
        "bundle": os.path.basename(loaded.bundle.path),
        "budget": "%s (generated %s / reference %s)" % (
            loaded.budget_source,
            loaded.generated.describe(), loaded.reference.describe()),
        "plans": [p for p in loaded.bundle.plans.plans.keys()],
        "keep_tiles": (loaded.generated.tiles if loaded.generated.tiles is not None
                       else "ratio=%.4f" % loaded.generated.ratio),
    }
    log("[veda] override installed: implementation=star7-cuda-int8(sm75) | %s"
        % info["budget"])
    return patched, patch, info
