# 运行时 attention 档位 API（attn.install）

老仓库在启动 CLI 里一次性安装 attention（`--attn int8|sage|w8a8|sdpa|veda`，
veda 失败静默回退 int8）。本服务是空壳启动，attention 改为**显式 operation**：

```text
POST /api/attention/install      202 operation（kind=attn.install）
GET  /api/attention             当前档位 + veda 工况快照
```

## 1. 请求

```json
{"mode": "int8|sdpa|sage|w8a8|veda|none",
 "veda": { "keep_tiles": 32, "ref_keep_tiles": -1, "keep_ratio": 0.0,
           "dense_layers": "0,1,47-49", "dense_steps": "3",
           "tile_precision": "fp32", "chunk_share": 16,
           "warmup": "probe", "debug": false },
 "request_id": "attn-1"}
```

| 字段 | 说明 |
|---|---|
| `mode` | 必填；`none` = 卸载回 factory 状态 |
| `veda` | 仅 `mode=veda` 接受；全部有默认值 |
| `request_id` | 幂等，同 payload 复用同一 operation |

veda 参数语义沿老仓库 CLI：

- `keep_tiles`：每 query tile 保留的 key tile 数（默认 0 = bundle 自带预算，即
  r2va 训练值 32）；`ref_keep_tiles` 独立控制参考段，-1 跟随 `keep_tiles`；
  `keep_ratio` 是比例语义（0–1），与 tiles 正交，非 0 时优先。
- `dense_layers` / `dense_steps`：强制走 dense 的层/步（0 基闭区间列表），
  上限 63（DiT 50 层内有效）。
- `tile_precision`：打分精度 `fp32|fp16`；`chunk_share`：tile 分块预算。
- `warmup`：`off|probe|tier1|tier2|tier3`（probe=256 tokens dense 通路验证，
  tier3=24500 全尺寸，仅手动）。**warmup 失败视为安装失败**，不静默回退。
- `debug`：star7 verbose。

## 2. 前置与互斥

- 要求 DiT 采样外壳就绪（`dit` 全部 unit 至少在 RAM；`mode=none` 同样要求，
  因为 factory 快照与还原都以外壳上的 model_options 为对象）。缺失报
  `ATTN_NOT_READY` 409。
- `mode=veda` 要求 `--defaults-file` 配置了 `models.veda.bundle.path`（安装读
  该文件做维度校验与预算推导）；未配置在**卸载原档之前**就拒绝，原档不受影响。
- 单 worker 串行：与权重四动作/推理共享唯一 operation 队列，天然互斥。

## 3. 安装行为（照搬已验证路径）

- `int8` / `sdpa`：`attention.optimized_attention` 与
  `comfy.ldm.minimax.model.optimized_attention` 直接换底座，无 override。
- `sage` / `w8a8`：`make_attention_override`（ComfyUI-TuringUtils，SM75 移植）
  构建的 override 装到 `unet` 与 `model_shifted` 的
  `transformer_options["optimized_attention_override"]`（H3 真实调用点只经
  comfy 分发器 `wrap_attn` 咨询这个键）；dense 回退底座为 int8（不可用时
  tuned sub_quad）。
- `veda`：`veda_star7.install`（backend DLL 自检 → bundle → 维度校验 →
  override 装原 unet → `ON_PREPARE_STATE`/`ON_CLEANUP` 回调注册 → warmup）；
  dense 回退底座为 int8；`model_shifted` 上双保险再 install 一次（幂等）。
  安装成功后 `/api/status.veda_calls` 与 `GET /api/attention` 反映稀疏工况。

## 4. 切换、卸载与失败语义

- 首次安装前快照 **factory 状态**（模块函数原引用 + unet/model_shifted 的
  override 键现值）。任何切换先完整还原到 factory 再装新档，因此
  A→B→A 不会把 A 的 override 错当原点。
- `mode=none`：还原 factory、摘除 veda 回调，`/api/status.attn` 回 `none`。
- **失败不静默换档**（与老仓库启动路径不同）：新档安装抛异常时先尝试**重装
  原档**；重装失败才落 factory，`details.restored_mode` 如实报告实际档位，
  错误码 `ATTN_INSTALL_FAILED` 500（retryable）。
- veda 累计统计跨切换保留（`/api/attention.veda_calls.cumulative`），
  `active` 字段反映当前是否 veda 档。

## 5. GET /api/attention

```json
{"ok": true, "attention": {
  "mode": "veda",
  "installed": {"mode": "veda", "veda": {"keep_tiles": 32, ...}},
  "override_active": false,
  "veda_calls": {"active": true, "backend": "...", "cumulative": {...},
                 "current": {...}}}}
```

`override_active` 仅对 sage/w8a8 为 true；veda 档的入口是回调 +
transformer_options override，`veda_calls.active` 才是 veda 的在位标记。

## 6. 已验证范围与边界

- 本地 fake 自测（`tests/_selftest_attn_api.py`）覆盖校验、安装、切换、
  失败还原、factory 还原、admission；**没有真实 SM75 kernel 验证**。
- 真机验收顺序建议：`sdpa`（无副作用对照）→ `sage` → `veda`（probe warmup）
  → 切换/`none` 还原 → 真任务下 `/api/warmup` 与 `veda_calls`。
- sage/w8a8 历史上标注"全尺寸 0/2 崩溃"，作为调试档照旧暴露；全尺寸稳定
  性问题不属于本 API 的修复范围。
- TE / VAE 推理不经 `optimized_attention_override`（它们走 PyTorch SDPA
  默认路径），本 API 只影响 DiT 采样前向。
