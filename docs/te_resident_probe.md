# TE 驻留推理与规模探针

## 阶段 1：Comfy 权重搬运保护

- 从 `agent/explicit-bindings` 的 `comfy_bindings.py` 复用所有权/执行上下文代码，
  不合并 attention 或 VEDA 变更。
- `TeResidency` 在显式 TE 上卡时调用原生 `LoadedModel.model_load()` 完成加载和
  登记，不使用可能全局驱逐其它模型的 `load_models_gpu()`。
- TE 构造禁用动态卸载；推理中的 `load_models_gpu` 只检查已存在的 TE 登记，
  不执行原加载流程。缺失登记拒绝推理，不补登记。
- TE 执行上下文中的 `free_memory` 禁止驱逐任何在场模型（包括 unowned）。
  helper 线程没有授权；裸函数别名或包装冲突拒绝执行。
- 物理审计读取真实 storage 地址/字节，不只读 registry 设备标签。
  部分驻留 DiT 合法；存储变化或无法审计不能报告成功。
- CUDA workspace OOM 与权重损坏分开：权重未变化时保留已驻留 TE，不自动卸载
  别的模型或缩小输入重试。
- legacy `/api/te/step` 与 `/api/te/probe` 迁入唯一 worker，有界等待保持同步
  响应；超时返回 operation receipt。移除 HTTP 线程中的重复执行与落盘代码。
  `Runtime.te_encode_tokens` 也复用同一 TE 推理实现。

本阶段独立提交时仍保留 ClipProj 原来的首次 encode GPU 副本创建；下面的
阶段 2 已将它移入显式上卡动作。

## 阶段 2：投影显式上卡

- `clipproj_resident.py` 只包装已核对的插件 `_encode`，不复制 encoder 或 MLP
  数学实现。规范化 encoder 源码 SHA256 为
  `f1da7ee8cee419a69e174c062be6f083388ade5612e1b6c1cfb3d013ca15beea`；
  其它版本拒绝，不静默使用兼容性猜测或备用模型。
- `te ram2vram` 一并准备投影系数、原插件 residual MLP 和 float32 sink。
  完整临时缓存验证成功后才发布；重复上卡不重新分配已有投影缓存。
- GPU 投影缓存与内部系数字典只读，推理不能走惰性权重创建路径；保留原
  dtype，CPU 主副本和 CUDA 计算副本分开记账（包括 sink）。
- `encode` 要求 TE 和已配置投影都为 vram。没有独立 ClipProj 的模型只按自身
  unit 检查，不虚构 te.proj。
- `vram2ram` 释放计算副本；`purgefromram` 检查投影 Tensor 的 weakref。
  从插件采用的 CPU 缓存只移除对应条目，避免全局缓存偷偷保留服务权重。
- 原子投影分配失败时报告实际 partial 状态，允许用户显式重试或下卡；
  TE error/mixed 可以通过整组 vram2ram 尝试恢复，不在推理时修复。
- CUDA 编码计时在计算前后同步，避免仅测量异步提交时间。

本地原生 ComfyUI（CPU）`tests/probes/te_native_bookkeeping.py` 已验证真实
LoadedModel/ModelPatcher 加载、登记、resident 检查、卸载与弱引用释放。
## 阶段 3：可调规模 TE 探针 API

先分别调用 TE 的 disk2ram、ram2vram，并等待每个 operation 完成。之后：

```text
POST /api/infer/te_probe
```

```json
{
  "scale": {"text_chars": 1024, "image_width": 512, "image_height": 512, "images": 1},
  "warmup": 1,
  "repeat": 5,
  "seed": 0,
  "min_free_mb": 256,
  "request_id": "te-probe-example-1"
}
```

返回 202 operation 回执，用 `GET /api/operations/<id>` 查询结果。

| 参数 | 默认 | 范围 |
|---|---|---|
| scale.text_chars | 512 | 1–8000 个字符，不是 token 数 |
| scale.image_width / image_height | 512 | 32–1600，32 的倍数 |
| scale.images | 1 | 1–9；当前没有公布 text_only 能力，不接受 0 |
| warmup | 1 | 0–5 次，不计入正式均值 |
| repeat | 3 | 1–20 次 |
| seed | 0 | 0–2^63−1，局部 CPU 图片随机种子，不改变全局采样 RNG |
| min_free_mb | 256 | 0–16384 MiB，执行前最低可用显存余量 |

- 生成指定字符长度的文字和指定大小的随机 RGB 参考图，再走真实 TE/投影计算。
  图片不经画布缩放，但 Qwen 自己仍可能调整输入；不承诺精确的内部 token 数。
- 只要求已配置的 TE 及其投影就绪；DiT/VAE/VEDA 可以缺席、在 CPU、部分或
  完全在 GPU。不加载它们，也不为 TE 卸载它们。
- 整个请求（准备输入、热身、重复编码及最终审计）共用一个驻留上下文，
  不在轮次之间留下 helper 线程可卸载模型的空档。
- 返回 actual_input、warmup/runs、编码耗时、输出条件形状/长度、显存统计和
  residence 存储审计。只返回指标，不保存 cond 或视频 artifact。
- encode_s 在 CUDA 计算前后同步；wall_s 还包含 tokenize、输出转 CPU 和有限值
  检查。每轮重置本服务 PyTorch 的显存峰值统计，workspace_peak_bytes 是超过
  本轮开始分配量的增量；它不是整卡所有进程的总峰值。
- min_free_mb 只是下限检查，不是“保证不 OOM”的估算。实际 OOM 明确失败，
  不缩小规模、不搬其它模型重试。NaN/Inf 输出也不能报告成功。
- request_id 同请求只执行一次，不同请求复用相同 ID 返回 409。

独立客户端只访问已有服务，不启动/停止进程、不加载权重：

```powershell
python tests/probes/te_infer_probe.py --url http://127.0.0.1:8190 `
  --text-chars 1024 --image-width 512 --image-height 512 --images 1 `
  --warmup 1 --repeat 5 --out te_probe_report.json
```

## 验证范围

CPU/fake 自测覆盖四动作、模型/投影缺失、原子准备、真实存储检查、释放、
请求校验、warmup/repeat、局部 RNG、OOM/非有限值、HTTP 202 与幂等、CLI 查询。
本地真实 Comfy CPU 登记/释放探针已通过，支持的 ClipProj 源码摘要已核对。
GPU 真实 TE、显存边界与同输入数值对比仍未验证；本轮没有连接远端。
