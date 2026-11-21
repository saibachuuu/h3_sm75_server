"""把已经构造好的 MiniMax H3 DiT skeleton 包装成可采样的 ComfyUI 模型。

这里刻意只做 **metadata / 采样外壳**：

* 不读取 checkpoint，不调用 ``load_models_gpu``，不构造第二份大模型。
* 不执行 forward，不初始化 CUDA，也不猜测原 SM75 权重的 dtype。
* 只使用调用方注入的 ``torch_mod`` / Comfy 模块 / SigmaShift 节点。

真实 ComfyUI 的 ``BaseModel.__init__`` 在
``model_config.unet_config["disable_unet_model_creation"] is True`` 时会跳过
``unet_model(**unet_config)``，但仍会建立 CPU 上的 model_sampling schedule。
因此本模块给 ``MiniMaxH3`` config 的私有副本加上这个开关，创建 BaseModel 后
立刻移除开关，再把已有的 skeleton 绑到 ``diffusion_model`` 上。传入的
``unet_config`` 字典永远不会被修改（包括构造抛异常时）。
"""

import operator
from typing import NamedTuple

H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0
_PRIVATE_DISABLE_KEY = "disable_unet_model_creation"

__all__ = [
    "H3PatcherPair",
    "H3_VIDEO_SHIFT",
    "H3_AUDIO_SHIFT",
    "invalidate_patcher_sizes",
    "wrap_h3_skeleton",
]


class H3PatcherPair(NamedTuple):
    """明确命名的成对返回值，避免 ``unet`` / ``model_shifted`` 混用。"""

    unet: object
    model_shifted: object


def _skeleton_dtype(skeleton):
    dtype = getattr(skeleton, "dtype", None)
    if dtype is None:
        getter = getattr(skeleton, "get_dtype", None)
        if callable(getter):
            dtype = getter()
    return dtype


def _drop_private_flag(config):
    """从 config 的 unet_config 副本移除内部开关，不碰调用方字典。"""
    if config is None:
        return
    mapping = getattr(config, "unet_config", None)
    if mapping is None:
        return
    try:
        mapping.pop(_PRIVATE_DISABLE_KEY, None)
    except Exception:
        # 如果注入的 config 使用了不可变 mapping，不掩盖真正的构造错误。
        pass


def _node_output_args(output):
    """兼容 io.NodeOutput(.args) 与普通 tuple/list 的节点返回值。"""
    args = getattr(output, "args", None)
    if args is not None:
        if isinstance(args, tuple):
            return args
        if isinstance(args, list):
            return tuple(args)
        return (args,)
    if isinstance(output, tuple):
        return output
    if isinstance(output, list):
        return tuple(output)
    raise RuntimeError(
        "SigmaShift 节点返回类型不可识别：%r；期望 NodeOutput.args 或 tuple/list"
        % (type(output),)
    )


def wrap_h3_skeleton(
    skeleton,
    unet_config,
    *,
    torch_mod,
    supported_models_mod,
    model_base_mod,
    model_patcher_mod,
    sigma_shift_node,
    load_device,
    offload_device,
    inference_dtype,
    manual_cast_dtype=None,
    quant_config=None,
):
    """把已存在的 H3 skeleton 包装为 ``(unet, model_shifted)``。

    参数
    ----
    skeleton:
        已经构造好的 ``comfy.ldm.minimax.model.MiniMaxH3Model``（或其他带
        ``dtype`` / ``eval`` / ``requires_grad_`` 的兼容对象）。本函数不会重建它。
    unet_config:
        H3 的 unet config 字典。函数内部复制；调用方字典保持原样。
    torch_mod:
        显式注入的 torch 依赖。仅作为依赖边界保留，本函数不会访问 CUDA 或做 cast。
    supported_models_mod:
        提供 ``MiniMaxH3`` config 类的 Comfy ``supported_models`` 模块。
    model_base_mod:
        提供 ``MiniMaxH3`` BaseModel 类的 Comfy ``model_base`` 模块。
    model_patcher_mod:
        提供 ``ModelPatcher`` 的 Comfy ``model_patcher`` 模块。
    sigma_shift_node:
        提供 ``execute(model, 12.0, 3.0)`` 的 SigmaShift 节点类。
    load_device / offload_device:
        调用方明确的 ModelPatcher 设备，不在这里推断。
    inference_dtype:
        调用方明确声明的 skeleton 权重 dtype。必须与 ``skeleton.dtype`` 一致；
        本函数不会为了“修正”它而转换权重。
    manual_cast_dtype:
        可选的计算时 manual-cast dtype，原样传给 ``set_inference_dtype``。
    quant_config:
        可选量化配置，设置到 config 上供后续权重生命周期使用。

    返回
    ----
    ``H3PatcherPair(unet=..., model_shifted=...)``。两者共享同一个 BaseModel
    和同一个 ``diffusion_model`` skeleton；``model_shifted`` 是 SigmaShift 的
    clone，带有固定 12.0 / 3.0 的 video/audio shift。
    """

    if skeleton is None:
        raise ValueError("skeleton 不能为空")
    if inference_dtype is None:
        raise ValueError("inference_dtype 必须由调用方明确传入，不能猜测")
    if load_device is None or offload_device is None:
        raise ValueError("load_device / offload_device 必须由调用方明确传入")

    skeleton_dtype = _skeleton_dtype(skeleton)
    if skeleton_dtype is None:
        raise ValueError("skeleton 没有可读取的 dtype；拒绝隐式猜测权重 dtype")
    if skeleton_dtype != inference_dtype:
        raise ValueError(
            "skeleton dtype %r 与调用方 inference_dtype %r 不一致；"
            "本函数不会隐式转换 skeleton 权重"
            % (skeleton_dtype, inference_dtype)
        )

    # 只改私有副本。即使 supported_models_mod 没有 copy，也不会污染调用方字典。
    private_config = dict(unet_config or {})
    private_config[_PRIVATE_DISABLE_KEY] = True
    config = supported_models_mod.MiniMaxH3(private_config)

    base = None
    try:
        config.set_inference_dtype(inference_dtype, manual_cast_dtype)
        if quant_config is not None:
            config.quant_config = quant_config

        # device=None：不在这里创建/移动任何模块；ModelPatcher 会把 BaseModel
        # 的 device 补成 offload_device。sampling schedule 留在 CPU 即可。
        base = model_base_mod.MiniMaxH3(config, device=None)
    finally:
        # 构造成功或失败都清理私有开关；调用方 dict 从未被写入过。
        _drop_private_flag(config)
        _drop_private_flag(getattr(base, "model_config", None))
        private_config.pop(_PRIVATE_DISABLE_KEY, None)

    # 已有 skeleton 的初始化状态在这里落实；不做 .to()/cast。
    if hasattr(skeleton, "eval"):
        skeleton.eval()
    if hasattr(skeleton, "requires_grad_"):
        skeleton.requires_grad_(False)

    # 私有 disable 开关生效时，真实 BaseModel 根本不会创建 diffusion_model
    # 属性；这里必须直接绑定，不能先要求该属性已存在。
    base.diffusion_model = skeleton
    if hasattr(base, "eval"):
        base.eval()

    unet = model_patcher_mod.ModelPatcher(base, load_device, offload_device)
    if getattr(unet, "model", None) is not base:
        raise RuntimeError("ModelPatcher 没有保留传入的 BaseModel")
    if getattr(base, "diffusion_model", None) is not skeleton:
        raise RuntimeError("BaseModel 的 diffusion_model 不是传入的 skeleton")

    shifted_output = sigma_shift_node.execute(
        unet, H3_VIDEO_SHIFT, H3_AUDIO_SHIFT
    )
    shifted_args = _node_output_args(shifted_output)
    if len(shifted_args) != 1:
        raise RuntimeError(
            "SigmaShift 应只返回一个 ModelPatcher，实际返回 %d 个"
            % (len(shifted_args),)
        )

    model_shifted = shifted_args[0]
    if model_shifted is unet:
        raise RuntimeError("SigmaShift 没有 clone，返回了原 ModelPatcher")
    if getattr(model_shifted, "model", None) is not base:
        raise RuntimeError("SigmaShift clone 没有共享同一个 BaseModel")
    if getattr(model_shifted.model, "diffusion_model", None) is not skeleton:
        raise RuntimeError("SigmaShift clone 的 diffusion_model 不是同一 skeleton")

    return H3PatcherPair(unet=unet, model_shifted=model_shifted)


def invalidate_patcher_sizes(*patchers, model_loaded_weight_memory=None):
    """权重生命周期后清掉 ModelPatcher 的 ``size`` 缓存。

    ``model_loaded_weight_memory`` 是 **可选** 的调用方实账：如果传入，就原样
    写到共享 model 的 ``model_loaded_weight_memory``（字节数），两个 patcher
    会看到同一个值。不传时保持现有字段不变，只清 size 缓存。本函数绝不自行
    扫描模块或计算/猜测 CUDA 占用。

    返回被处理的 patcher 元组，便于调用方继续使用。
    """

    memory = None
    if model_loaded_weight_memory is not None:
        try:
            memory = operator.index(model_loaded_weight_memory)
        except TypeError as exc:
            raise ValueError("model_loaded_weight_memory 必须是整数字节数") from exc
        if memory < 0:
            raise ValueError("model_loaded_weight_memory 不能为负数")

    touched = []
    for patcher in patchers:
        if patcher is None:
            continue
        patcher.size = 0
        if memory is not None:
            model = getattr(patcher, "model", None)
            if model is None:
                raise ValueError("patcher 缺少 model，无法写 model_loaded_weight_memory")
            model.model_loaded_weight_memory = memory
        touched.append(patcher)
    return tuple(touched)
