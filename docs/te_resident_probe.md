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

本阶段仍保留 ClipProj 原来的首次 encode GPU 副本创建；完整投影驻留保护在
阶段 2 接入。不要把此阶段独立当作最终规模探针实现。

测试：原有 15 组自测 + bindings 25 个契约场景 + TE residency 8 个 CPU 场景。
CPU/fake 测试不能证明真实 RTX 2080 Ti 上的驻留、数值或性能；远端尚未验证。
