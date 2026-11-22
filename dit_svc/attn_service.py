"""运行时 attention 档位安装/切换（attn.install operation 的执行体）。

老仓库在启动 CLI 里一次性装档（``runtime._install_attention``，veda 失败
静默回退 int8）。本模块把同一条已验证路径改造成显式 API 语义：

* 单 worker 内执行，与权重四动作/推理串行（operation_admission 把关）；
* 首次安装前快照 factory 状态（``attention.optimized_attention`` /
  ``minimax_model.optimized_attention`` / unet 与 model_shifted 的
  ``transformer_options["optimized_attention_override"]``），卸载与切换都
  还原到 factory，不把上一档的 override 错当原点；
* veda 安装失败**报错并尝试还原原档**（重装失败才落 factory，并在
  details 里如实报告），绝不静默换成 int8；
* sage/w8a8 的 dense 回退底座、veda 的 dense fallback + model_shifted
  双保险 + 回调注册（ON_PREPARE_STATE / ON_CLEANUP）照搬 runtime._load()
  与 veda_star7.install 的既有行为，本模块不发明新的安装方式。

安装要求 DiT 采样外壳就绪（``runtime.unet`` / ``runtime.model_shifted``
非空，即 dit 全部 unit 至少在 RAM——由 dit_weights._ensure_patchers 保证）；
attention 只改 model_options 与模块函数引用，不搬权重，meta/CPU 骨架
也能读 blocks 维度做 veda 校验。
"""
import time

from dit_svc.operations import OperationError, json_or_marker
from dit_svc.veda_star7 import CALLBACK_KEY

DENSE_MODES = ("int8", "sdpa", "sage", "w8a8")


# ------------------------------------------------------------------ factory

def _record_override(patcher):
    """factory 快照里 unet/model_shifted 的 override 现值；None = 原本无键。"""
    if patcher is None:
        return None
    topts = (getattr(patcher, "model_options", None) or {}).get(
        "transformer_options") or {}
    if "optimized_attention_override" in topts:
        return ("value", topts["optimized_attention_override"])
    return None


def _ensure_factory(runtime):
    """首次安装前记录 factory 状态；此后切换/卸载一律还原到这份快照。"""
    if getattr(runtime, "_attn_factory", None) is not None:
        return
    runtime._attn_factory = {
        "attention_fn": runtime.attention.optimized_attention,
        "minimax_fn": runtime.minimax_model.optimized_attention,
        "unet_override": _record_override(runtime.unet),
        "shifted_override": _record_override(runtime.model_shifted),
    }
    print("[attention] factory 快照已记录", flush=True)


def _restore_override(patcher, record):
    if patcher is None:
        return
    topts = patcher.model_options.setdefault("transformer_options", {})
    if record is None:
        topts.pop("optimized_attention_override", None)
    else:
        topts["optimized_attention_override"] = record[1]


def _remove_veda_callbacks(unet):
    """摘掉 veda_star7 注册在原 unet 上的两个回调（无 veda 档时无害）。"""
    if unet is None:
        return
    try:
        import comfy.patcher_extension
        ct = comfy.patcher_extension.CallbacksMP
    except ImportError:
        return
    for call_type in (ct.ON_PREPARE_STATE, ct.ON_CLEANUP):
        try:
            unet.remove_callbacks_with_key(call_type, CALLBACK_KEY)
        except Exception:  # noqa: BLE001  老 comfy 无该方法时静默跳过
            pass


def _uninstall_to_factory(runtime):
    """把 attention/调用点/回调全部还原到 factory 快照，runtime.attn 置 none。"""
    factory = getattr(runtime, "_attn_factory", None)
    if factory is None:
        return
    runtime.attention.optimized_attention = factory["attention_fn"]
    runtime.minimax_model.optimized_attention = factory["minimax_fn"]
    _restore_override(runtime.unet, factory["unet_override"])
    _restore_override(runtime.model_shifted, factory["shifted_override"])
    _remove_veda_callbacks(runtime.unet)
    runtime.attn_override = None
    runtime.veda_patch = None
    runtime.attn = "none"
    runtime._attn_installed_request = None


# ------------------------------------------------------------------ veda 档

def _veda_bundle_path(runtime):
    backend = runtime.weights.backends.get("veda.bundle")
    config = backend.config if backend is not None and backend.configured() else None
    path = (config or {}).get("path")
    if not path:
        raise OperationError(
            "ATTN_NOT_READY",
            "veda.bundle recipe 未配置（--defaults-file models.veda.bundle.path），"
            "无法安装 veda 档", 409, False)
    return path


def _veda_args(runtime, veda, bundle_path):
    """veda dict 缺键时按 validate 的默认值补齐（防御式：不经 HTTP 的调用者）。"""
    from types import SimpleNamespace
    return SimpleNamespace(
        veda_bundle=bundle_path,
        veda_keep_tiles=veda.get("keep_tiles", 0),
        veda_ref_keep_tiles=veda.get("ref_keep_tiles", -1),
        veda_keep=veda.get("keep_ratio") or 0.0,
        veda_dense_layers=veda.get("dense_layers", ""),
        veda_dense_steps=veda.get("dense_steps", ""),
        veda_tile_precision=veda.get("tile_precision", "fp32"),
        veda_chunk_share=veda.get("chunk_share", 16),
        veda_debug=veda.get("debug", False))


def _install_veda(runtime, veda, operation=None):
    """安装 veda：backend 自检 + bundle 维度校验 + override + 回调 + warmup。

    与 runtime._install_attention 的 veda 分支等价，但失败不静默回退
    int8——异常原样上抛，由 install() 统一还原原档并报错。
    """
    import torch
    from dit_svc import veda_star7
    from dit_svc.runtime import _VedaWarmupShim

    rt = runtime.rt
    attention = runtime.attention
    minimax_model = runtime.minimax_model
    comfy_kitchen = runtime.comfy_kitchen
    unet = runtime.unet
    device = torch.device("cuda:0")
    int8_ok = (attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
               and comfy_kitchen.int8_attention_is_available(device))
    bundle_path = _veda_bundle_path(runtime)
    args_proxy = _veda_args(runtime, veda, bundle_path)

    handle, veda_info = veda_star7.install(
        unet, args_proxy, torch=torch,
        log=lambda s: print(s, flush=True))

    # dense 回退底座（Veda 拒绝的调用走 int8 而不是裸 SDPA——sm75+bf16 会
    # Math 模式 OOM）；override 仍在 model_options 里，H3 前向照样经
    # wrap_attn 咨询它，这里换的只是"没 override 时的默认"。
    if int8_ok:
        attention.optimized_attention = rt._wrap_attn_mem_probe(
            attention.attention_comfy_kitchen_int8, "veda-dense")
    else:
        attention.optimized_attention = attention.attention_sub_quad
        rt.install_tuned_sub_quad_chunking(attention)
    minimax_model.optimized_attention = attention.optimized_attention

    # model_shifted 双保险（老 _load() 同款；install 幂等：current in installed）
    shifted = runtime.model_shifted
    if shifted is not None:
        handle.install(shifted.model_options.setdefault("transformer_options", {}))
        print("[attention] veda override installed on model_shifted.model_options",
              flush=True)

    fallback = (attention.attention_comfy_kitchen_int8 if int8_ok
                else attention.attention_sub_quad)
    level = veda.get("warmup", "probe")
    if level != "off":
        # warmup 失败视为安装失败（不静默回退）：原档由 install() 恢复。
        _VedaWarmupShim().run(torch, unet, handle, level, fallback)
    return {"mode": "veda",
            "implementation": "veda(sparse) + int8 dense fallback",
            "veda_info": veda_info,
            "override": None,
            "veda_patch": handle}


# ------------------------------------------------------------------ dense 档

def _install_dense(runtime, mode, operation=None):
    """int8/sdpa/sage/w8a8：直接复用已验证的 runtime._install_attention。"""
    from dit_svc.runtime import _install_attention
    override = _install_attention(
        runtime.rt, runtime.attention, runtime.minimax_model,
        runtime.comfy_kitchen, mode, args=runtime.args, unet=runtime.unet)
    if override is not None:
        # sage/w8a8：override 装到 unet 与 model_shifted 的 transformer_options
        #（老 _load() 同款；真正的 H3 调用点只经 wrap_attn 咨询这个键）。
        runtime.unet.model_options.setdefault("transformer_options", {})[
            "optimized_attention_override"] = override
        shifted = runtime.model_shifted
        if shifted is not None:
            shifted.model_options.setdefault("transformer_options", {})[
                "optimized_attention_override"] = override
        print("[attention] %s override installed on unet + model_shifted" % mode,
              flush=True)
        implementation = getattr(override, "turing_utils_attention_implementation",
                                  "?")
    else:
        implementation = mode
    return {"mode": mode, "implementation": implementation,
            "override": override, "veda_patch": None}


# ------------------------------------------------------------------ 编排

def _apply_mode(runtime, mode, veda, operation=None):
    if mode == "veda":
        return _install_veda(runtime, veda, operation)
    return _install_dense(runtime, mode, operation)


def _publish(runtime, mode, info, veda):
    runtime.attn = mode
    runtime.attn_override = info.get("override")
    runtime.veda_patch = info.get("veda_patch")
    runtime._attn_installed_request = {"mode": mode, "veda": veda}


def _try_restore(runtime, previous, operation=None):
    """安装失败后还原原档；重装失败落 factory 并如实返回实际档名。"""
    mode = previous.get("mode") if previous else "none"
    veda = previous.get("veda") or {} if previous else {}
    if not mode or mode == "none":
        _uninstall_to_factory(runtime)
        return "none"
    try:
        info = _apply_mode(runtime, mode, veda, operation)
        _publish(runtime, mode, info, veda)
        print("[attention] 失败后已还原原档: %s" % mode, flush=True)
        return mode
    except Exception as exc:  # noqa: BLE001  恢复失败必须落 factory，不能谎报
        print("[attention] 原档 %s 还原失败(%s: %s)，落回 factory"
              % (mode, type(exc).__name__, exc), flush=True)
        _uninstall_to_factory(runtime)
        return "none"


def install(runtime, request, operation=None):
    """attn.install 的唯一执行入口（单 worker 内调用）。"""
    mode = request["mode"]
    veda = request.get("veda") or {}
    started = time.perf_counter()
    if runtime.unet is None or runtime.model_shifted is None:
        raise OperationError(
            "ATTN_NOT_READY",
            "DiT 采样外壳未就绪；先完成 dit 全量 disk2ram（补齐全部 unit）",
            409, False,
            {"hint": "POST /api/weights/dit/disk2ram {\"mode\":\"all\"}"})
    if mode == "veda":
        # 提前读 recipe（无副作用）：配置缺失时连卸载都不发生，原档原样保留。
        _veda_bundle_path(runtime)

    _ensure_factory(runtime)
    previous = dict(getattr(runtime, "_attn_installed_request", None)
                    or {"mode": "none", "veda": {}})
    _uninstall_to_factory(runtime)

    if mode == "none":
        elapsed = round(time.perf_counter() - started, 3)
        print("[attention] 已卸载到 factory in %.2fs" % elapsed, flush=True)
        return {"mode": "none", "implementation": "factory",
                "previous_mode": previous.get("mode"), "elapsed_s": elapsed}

    try:
        info = _apply_mode(runtime, mode, veda, operation)
    except Exception as exc:
        restored = _try_restore(runtime, previous, operation)
        if isinstance(exc, OperationError):
            err = exc
            err.details = dict(err.details or {})
        else:
            err = OperationError(
                "ATTN_INSTALL_FAILED",
                "%s 档安装失败: %s: %s" % (mode, type(exc).__name__, exc),
                500, True)
            err.details = {}
        err.details.update({"requested_mode": mode, "restored_mode": restored})
        raise err from exc

    _publish(runtime, mode, info, veda)
    elapsed = round(time.perf_counter() - started, 3)
    print("[attention] installed mode=%s in %.2fs" % (mode, elapsed), flush=True)
    result = {"mode": mode, "implementation": info.get("implementation"),
              "previous_mode": previous.get("mode"), "elapsed_s": elapsed}
    if mode == "veda":
        result["veda_info"] = json_or_marker(info.get("veda_info"), "veda_info")
    return result
