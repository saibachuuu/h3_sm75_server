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
CPU/fake 测试不能证明真实 RTX 2080 Ti 上的驻留、数值或性能；远端尚未验证。
