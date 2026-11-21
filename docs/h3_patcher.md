# H3 skeleton patcher 包装

`dit_svc/h3_patcher.py` 把**已经构造好的** MiniMax H3 DiT skeleton 接到
ComfyUI 的采样外壳上。它只做 metadata / model_sampling / ModelPatcher
包装，不读取权重文件，不重建 DiT，不执行 forward，也不初始化 CUDA。

## 为什么需要这一层

真实 ComfyUI 源码（approved 源码快照
`explicit_api_sources_20261121/comfy/model_base.py` 171-208 行）里，
`BaseModel.__init__` 只有在下面条件成立时才构造 `diffusion_model`：

```python
if not unet_config.get("disable_unet_model_creation", False):
    ...
    self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
```

`model_base.MiniMaxH3` 默认使用 `ModelType.FLOW_AV`，最终走
`BaseModel`；`supported_models.MiniMaxH3` 的 sampling 默认是
video shift 12.0 / audio shift 3.0。P0 已经直接构造出了
`comfy.ldm.minimax.model.MiniMaxH3Model` skeleton，但 skeleton 本身不是
可采样的 `BaseModel`，也没有 `ModelPatcher` 和固定的 SigmaShift clone。

本模块因此：

1. 在 config 的**私有副本**里设置
   `disable_unet_model_creation=True`；
2. 创建 `model_base.MiniMaxH3(config)`，跳过第二次大模型构造，但保留
   正常的 CPU sampling schedule；
3. 立刻移除私有开关，绑定传入的 skeleton；
4. 用 `ModelPatcher(model, load_device, offload_device)` 建立轻量 patcher；
5. 调用真实 `MiniMaxH3SigmaShift.execute(unet, 12.0, 3.0)`，得到共享同一
   BaseModel / diffusion_model 的 `model_shifted` clone。

传入的 `unet_config` 字典不会被修改；构造抛异常时也不会留下私有开关。

## API

```python
from dit_svc.h3_patcher import (
    H3PatcherPair,
    H3_VIDEO_SHIFT,
    H3_AUDIO_SHIFT,
    invalidate_patcher_sizes,
    wrap_h3_skeleton,
)

pair = wrap_h3_skeleton(
    skeleton,
    unet_config,
    torch_mod=torch,
    supported_models_mod=comfy.supported_models,
    model_base_mod=comfy.model_base,
    model_patcher_mod=comfy.model_patcher,
    sigma_shift_node=MiniMaxH3SigmaShift,
    load_device=device,
    offload_device=device,
    inference_dtype=torch.bfloat16,
    manual_cast_dtype=None,
    quant_config=None,
)

unet = pair.unet
model_shifted = pair.model_shifted
```

返回值是 `H3PatcherPair`（`NamedTuple`），字段名明确；同时也可以按普通
二元 tuple 解包：

```python
unet, model_shifted = pair
```

### 参数约束

| 参数 | 约束 |
|---|---|
| `skeleton` | 已存在的 H3 DiT skeleton，必须有可读 `dtype` |
| `unet_config` | mapping；内部复制，调用方原对象不变 |
| `torch_mod` | 显式注入；本模块不会访问 `cuda`、不会做 dtype/device cast |
| `supported_models_mod` | 需有 `MiniMaxH3` config 类 |
| `model_base_mod` | 需有 `MiniMaxH3` BaseModel 类 |
| `model_patcher_mod` | 需有 `ModelPatcher` |
| `sigma_shift_node` | 需有 `execute(model, 12.0, 3.0)` |
| `load_device` / `offload_device` | 调用方明确传入，不由本模块猜测 |
| `inference_dtype` | 必须与 `skeleton.dtype` 相等，否则 `ValueError` |
| `manual_cast_dtype` | 原样传给 `set_inference_dtype` |
| `quant_config` | 非 `None` 时设置到 config，供后续权重生命周期使用 |

### dtype 策略

本模块**不猜测**原 SM75 权重的 dtype，也不把 skeleton 转成另一个 dtype：

* `inference_dtype` 只传给 `config.set_inference_dtype(inference_dtype,
  manual_cast_dtype)`；
* `skeleton.dtype` 必须等于 `inference_dtype`，否则在任何 config/model
  构造前直接失败；
* 需要 manual-cast 时由调用方同时给出匹配的 `inference_dtype`（存储 dtype）
  与 `manual_cast_dtype`（计算 dtype）；
* 不存在隐式 `.to()`、`.cuda()` 或 `meta` 全局上下文。

skeleton 会被设置为 `eval()` 且 `requires_grad_(False)`。sampling schedule
保持在 CPU 上即可，不会被强制移动到 GPU，也不会被 meta context 变成 meta。

## SigmaShift 与 patcher 共享关系

`wrap_h3_skeleton` 固定调用：

```python
sigma_shift_node.execute(unet, H3_VIDEO_SHIFT, H3_AUDIO_SHIFT)
```

并校验：

* `unet.model is model_shifted.model`；
* 两者 `.model.diffusion_model is skeleton`；
* `model_shifted is not unet`（确实发生了 clone）；
* 节点返回 `NodeOutput.args` 或 tuple/list 中的唯一 ModelPatcher。

真实的 `ModelPatcher.clone()` 会深拷贝 options 并复用同一 `model`，SigmaShift
再把 video/audio shift 写入 clone 的
`transformer_options["minimax_h3_sigma_shift_video/audio"]`。因此
`unet` 和 `model_shifted` 的权重身份完全一致，只有采样 schedule 不同。

## 权重生命周期后的 size 缓存

`ModelPatcher.model_size()` 会缓存 `size`。权重加载、部分卸载、重绑或
generation 变化后，调用方应清掉缓存：

```python
invalidate_patcher_sizes(
    pair.unet,
    pair.model_shifted,
    model_loaded_weight_memory=actual_cuda_bytes,
)
```

* 所有传入 patcher 的 `size` 归零；
* 如果传入 `model_loaded_weight_memory`，原样写到共享 model 的
  `model_loaded_weight_memory`（字节数）；
* 不传该参数时只清 `size`，不修改现有字节账；
* 本函数不扫描模块、不查询 CUDA，实际字节数由调用方按真实账目传入。

## 明确不做的事

* 不 import torch / CUDA（模块顶层只用 stdlib）；
* 不读 safetensors / checkpoint，不调用 `load_models_gpu`；
* 不构造第二份 `MiniMaxH3Model`；
* 不执行 `.attn` 安装、attention override 或 forward；
* 不修改调用方 `unet_config`，也不永久保留私有
  `disable_unet_model_creation` 标记。

## 自检

纯 stdlib、无 torch、无 GPU：

```powershell
python tests/_selftest_h3_patcher.py
```

覆盖：

* fake BaseModel 检查私有 disable 开关；若未生效，记录“重建大模型”次数并失败；
* 两个 patcher 共享同一个 BaseModel / diffusion_model；
* SigmaShift 收到精确的 `(12.0, 3.0)`（NodeOutput 与 tuple 两种返回）；
* dtype 匹配、manual-cast 透传、dtype 不匹配在任何构造前失败；
* 构造异常不会把私有 flag 写进传入的 `unet_config`；
* 不访问注入的 fake torch（因此不会初始化 CUDA）；
* `invalidate_patcher_sizes` 清 size 缓存，并按调用方传入值更新字节账。
