# DiT 采样探针（POST /api/infer/dit_probe）

可调规模的**DiT 主体采样探针**：合成输入、真实采样计算、只返回指标。
与 [te_resident_probe.md](te_resident_probe.md) 的 TE 探针同构，但走的是
DiT 采样链（SamplerCustomAdvanced），**不包含 token refiner 流程**。

```text
POST /api/infer/dit_probe
GET  /api/operations/<id>
```

## 1. 前置

**除 token refiner 外，DiT 其余 unit 必须在显存**；`dit.refiner`
（`token_refiner` + `condition_proj`）允许缺席（`absent`）。

原因：探针的 cond 直接以 DiT 内部宽度（`hidden_size`，本 checkpoint 为 5376）
构造，`MiniMaxH3Model._embed_and_pack` 走
`text_states.shape[-1] == self.hidden_size` 快路径，不会调用
`token_refiner` / `condition_proj`。所以这是一次纯粹的 **body-only** 探针。

TE / VAE / VEDA 均不需要在场。

如果某项 body unit 不在显存，admission 阶段返回 `MISSING_UNITS` 409，
`details.missing` 列出具体 unit id（`dit.refiner` 永不出现在其中）。

真实视频任务（`/api/jobs*`）仍然要求 `dit` 全量 `vram`（含 refiner），
探针的放宽不适用于采样任务。

## 2. 请求

```json
{"scale": {"cond_tokens": 512, "width": 640, "height": 960, "length": 124},
 "steps": 1, "warmup": 1, "repeat": 3, "seed": 0, "min_free_mb": 256}
```

| 字段 | 默认 | 范围/含义 |
|---|---|---|
| scale | 必须给对象 | `{}` 使用下面默认值，未知键 400 |
| scale.cond_tokens | 512 | 16–8000；合成 cond 的序列长度 |
| scale.width / height | 640 / 960 | 256–1600，32 的倍数 |
| scale.length | 124 | 16–400 帧（24 fps 口径） |
| steps | 1 | 1–4（探针用小步数） |
| warmup | 1 | 0–5 次，不计入正式统计 |
| repeat | 3 | 1–20 次 |
| seed | 0 | 0–2^63−1；cond 与噪声共用此种子 |
| min_free_mb | 256 | 每轮采样前的最低可用显存余量 |

## 3. 真实计算路径

```text
http_api._post_dit_probe -> api_validation.validate_dit_probe_request
  -> dit_probe.validate_request
  -> Runtime.operation_admission（除 refiner 外 unit 必须 vram）
  -> QueueManager.submit_operation('infer.dit_probe')
  -> Runtime.run_infer_dit_probe -> dit_probe.run_probe
       合成 cond（局部 RNG，随机 [1, N, hidden_size] bf16 + 全 1 tags）
       -> 直接作为 positive 条件（不调 preprocess_text_embeds）
       -> _empty_av_latent（与视频任务同款空 latent）
       -> RandomNoise / BasicGuider / KSamplerSelect(res_multistep)
          / BasicScheduler(simple) / SamplerCustomAdvanced
       warmup + repeat 轮，逐轮同步 CUDA、重置 peak、显存余量检查
       结束后对比 dit 驻留快照（units 设备/计数/字节）
```

- **不带参考图**：不建 `minimax_refs`，不涉及 VAE；探针自包含。
- **不含 token refiner**：cond 宽度等于 body 宽度，forward 跳过
  `condition_proj` + `token_refiner`；因此 refiner 缺席也能跑。
- tags 全 1（纯文本 token），不构造 vision/ref 段。
- 与 `/api/jobs` 真任务共用同一采样链，不复制采样数学。

## 4. 结果（operation kind = infer.dit_probe）

```text
kind = dit_probe
synthetic_inputs = true
body_only = true
token_refiner = "skipped (body-width cond)"
cond_dim = 5376
scale / steps / seed
warmup = 各次热身统计
runs = 各次正式统计
veda_calls = veda 工况快照（veda 档才非 null；含 cumulative sparse/dense）
summary = {repeat, sample_mean_s, sample_min_s, sample_max_s,
           peak_allocated_bytes}
memory_before / memory_after
elapsed_s
```

每次统计：

```text
iteration, sample_s, steps, frames, latent_finite=true,
peak_allocated_bytes, peak_reserved_bytes, workspace_peak_bytes,
memory_after = {free_bytes, total_bytes, allocated_bytes, reserved_bytes}
```

- `sample_s`：一轮完整采样（noise→guider→sampler→latent），CUDA 同步
  前后计时；`warmup + repeat` 共用一个 refine 后的 cond 与 latent 骨架。
- `latent_finite`：逐帧 finite 检查（整块 float() 在全尺寸下会额外吃显存）。
- 峰值口径同 TE 探针：本进程 PyTorch 分配器统计，不是全卡占用。
- **不写任何 artifact**（mp4/latent/cond 都不落盘），不回调 cond server。

## 5. 失败语义（与 TE 探针一致）

| code | HTTP | 场景 |
|---|---|---|
| `MISSING_UNITS` | 409 | 除 refiner 外的 body unit 不在显存（admission 阶段） |
| `INSUFFICIENT_VRAM` | 409 | 某轮前可用显存低于 min_free_mb |
| `CUDA_OOM` | 409 | 采样真实 OOM，不自动缩小规模/卸载其它模型重试 |
| `NONFINITE_LATENT` | 422 | 采样输出含 NaN/Inf |
| `WEIGHT_MUTATED` | 500 | 采样前后 dit 驻留快照（设备/计数/字节）变化 |

执行阶段失败查 `operation.status=failed` 与 `error.details`
（含 scale/steps/failed_iteration/warmup_completed/runs_completed/memory）。

## 6. 已验证范围

`tests/_selftest_dit_probe.py`（13 用例，fake 采样链 + CPU torch）覆盖：
严格校验、合成 cond 确定性/宽度、probe 不触发 refiner、warmup/repeat 指标、
显存不足前拒绝、非有限 latent、驻留突变 fail-closed、admission 对
refiner 缺席的放行与 body unit 缺失的拒绝、真实任务的严格门槛。
**没有真实 GPU 采样验证**；真机最小验收建议：
`{"scale": {"cond_tokens": 512, "width": 640, "height": 960,
"length": 124}, "steps": 1, "repeat": 1, "min_free_mb": 512}`，
veda 档下对照 `veda_calls.cumulative`。
