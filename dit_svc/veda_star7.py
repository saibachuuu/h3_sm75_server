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
  wrapper / schedule。

本模块职责：
  1. **启动即自检**：backends.resolve() 加载 SM75 DLL + 数值 self_test，
     失败直接抛（回退 int8）——"待命"状态必须先知道 kernel 可用；
  2. 把 star7 的诊断 logger 接到 stdout（每 run 的 Attention computed % 摘要）；
  3. CLI 参数 -> star7 VedaSettings（sparsity / dense layers / steps）；
  4. 加载 bundle（veda/bundle_compat.py，兼容 r2va finetune 的 tiles 预算）；
  5. **回调注册到原 unet**（不是 apply() 返回的 clone——server 全程用原
     unet 干活；clone 链经 ModelPatcher.clone() 自动继承 callbacks，见
     comfy/model_patcher.py:431-505）；
  6. 包住 patch.on_cleanup，在 run 清零前抓取统计（否则任务一结束 /api/status
     就归零，WebUI 永远看不到真数字）。
"""
from __future__ import annotations

import os
import sys

CALLBACK_KEY = "veda_sparse_attention"


def _ensure_import_path(here: str) -> None:
    """让 `import veda` 解析到 h3_sm75_server/veda（server 根目录）。"""
    if here not in sys.path:
        sys.path.insert(0, here)


def resolve_backend(torch, log=print) -> dict:
    """启动即加载 SM75 稀疏后端并做数值自检（fail fast）。

    star7 的 backends.resolve() 会：加载 DLL（sha256 + ABI 校验）-> base.self_test()
    （含"kernel 是否忽略了 block mask"的检查）。这一步放在服务启动时，
    而不是等第一个真任务的第一次稀疏调用才暴露失败。
    """
    from veda import backends
    from dit_svc import veda_attn_star7

    info = hardware_device_info(torch)
    device = torch.device("cuda:0")
    resolution = backends.resolve(device, notify=lambda s: log("[veda-backend] %s" % s))
    report = resolution.report()
    veda_attn_star7.BACKEND_INFO.update(
        resolved=True,
        ok=resolution.backend is not None,
        display=(resolution.backend.display if resolution.backend is not None else None),
        report=report)
    log("[veda-backend] device=%s cc=%s" % (info.label, info.cc))
    log("[veda-backend] %s" % report)
    if resolution.backend is None:
        raise RuntimeError("SM75 稀疏后端不可用（self_test 未过）: %s" % report)
    log("[veda-backend] self_test passed: %s" % resolution.backend.display)
    return resolution


def hardware_device_info(torch):
    from veda import hardware
    return hardware.describe(torch.device("cuda:0"))


def settings_from_args(args, log=print):
    """CLI -> star7 VedaSettings（sparsity / dense layers / steps）。"""
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
        if 0.0 < ratio <= 1.0:
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
    if keep_tiles or keep_ratio or getattr(args, "veda_debug", False):
        log("[veda] settings: generated=%s reference=%s dense_layers=%s dense_steps=%s verbose=%s"
            % (settings.describe(), settings.describe_full_attention() or "-",
               sorted(dense_layers), sorted(dense_steps), settings.verbose))
    return settings


def _parse_int_list(text: str):
    """"0,1,47-49" -> frozenset({0,1,47,48,49})（0 基，含端点）。"""
    if not text.strip():
        return frozenset()
    from veda.settings import parse_index_list
    return parse_index_list(text, "list")


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


def _register_callbacks(unet, patch, log=print) -> dict:
    """把 star7 的两个回调注册到**原 unet**（不是 apply 返回的 clone）。

    关键：apply() 返回的 clone 带着 callbacks，但 server 全程用原 unet 干活
    （常驻引用 / _unhook_dit / profiler / SigmaShift 从它 clone），所以那份
    clone 是死的 —— 回调必须装在原 unet 上，之后的 clone 链才会继承：
      * ON_PREPARE_STATE：comfy/samplers.py:258,396 每步触发，
        patch.install() 幂等（current in self.installed），保证 override
        永远在栈顶（后装的其它节点顶不掉它）；
      * ON_CLEANUP：comfy/samplers.py:1267 的 finally 必触发，
        star7 在这里打 Attention computed % 摘要 + 清零 run；
        bridge 在清零前抓取累计统计（veda_attn_star7.wrap_cleanup）。
    """
    import comfy.patcher_extension
    from dit_svc import veda_attn_star7

    ct = comfy.patcher_extension.CallbacksMP
    added = {}
    # 同 key 会 append（model_patcher.py:1325-1328），先清掉防重复注册
    for call_type in (ct.ON_PREPARE_STATE, ct.ON_CLEANUP):
        try:
            unet.remove_callbacks_with_key(call_type, CALLBACK_KEY)
        except Exception:  # noqa: BLE001  老 comfy 可能没有该方法
            pass

    def _on_prepare(model_patcher, timestep, model_options):
        patch.install(model_options["transformer_options"])

    def _on_cleanup(model_patcher):
        patch.on_cleanup()

    unet.add_callback_with_key(ct.ON_PREPARE_STATE, CALLBACK_KEY, _on_prepare)
    unet.add_callback_with_key(ct.ON_CLEANUP, CALLBACK_KEY, _on_cleanup)
    added["on_prepare_state"] = True
    added["on_cleanup"] = True

    if veda_attn_star7.wrap_cleanup(patch, log=log):
        added["stats_wrap"] = True
    return added


def install(unet, args, torch, log=print, probe_log=None):
    """安装 veda：backend 自检 -> bundle -> override 就地装到 unet + 回调。

    返回 (patch, info)；patch 即 runtime.veda_patch（供统计读取）。
    失败一律抛异常，由 _install_attention 捕获回退 int8。
    """
    import veda.comfy_patch as veda_comfy_patch
    from dit_svc import veda_attn_star7

    # 0) star7 诊断接出（每个 run 的 Attention computed % 摘要靠它可见）
    veda_attn_star7.install_logger(log=log)

    # 1) SM75 后端：加载 DLL + 数值 self_test，启动即 fail fast
    resolve_backend(torch, log=log)

    # 2) 参数与 bundle
    settings = settings_from_args(args, log=log)
    loaded = load_bundle(args, log=log)
    check_model_dims(unet, loaded, log=log)

    # 3) override（apply 内 clone 装一份；下面在原 unet 上再装一次，
    #    让 server 全程用的这个对象带上 override）
    patched, patch = veda_comfy_patch.apply(unet, loaded.bundle, settings, None)
    patch.install(unet.model_options.setdefault("transformer_options", {}))
    callbacks = _register_callbacks(unet, patch, log=log)
    log("[veda] override installed on unet.model_options + callbacks=%s"
        % sorted(callbacks))
    # apply 返回的 clone 我们不用（server 用原 unet）；patch 与 callbacks 已就位
    del patched

    info = {
        "bundle": os.path.basename(loaded.bundle.path),
        "budget": "%s (generated %s / reference %s)" % (
            loaded.budget_source,
            loaded.generated.describe(), loaded.reference.describe()),
        "plans": len(loaded.bundle.plans.plans),
        "keep_tiles": (loaded.generated.tiles if loaded.generated.tiles is not None
                       else "ratio=%.4f" % loaded.generated.ratio),
        "backend": veda_attn_star7.BACKEND_INFO.get("display"),
        "callbacks": sorted(callbacks),
    }
    log("[veda] active: implementation=star7-cuda-int8(sm75) | %s" % info["budget"])
    return patch, info
