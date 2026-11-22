# DiT 采样探针（POST /api/infer/dit_probe）

可调规模的**真实 DiT 采样探针**：合成输入、真实计算路径、只返回指标。
与 [te_resident_probe.md](te_resident_probe.md) 的 TE 探针同构，但走的是
DiT 采样链（token_refiner → SamplerCustomAdvanced），不是 TE 编码。

```text
POST /api/infer/dit_probe
GET  /api/operations/<id>
```

## 1. 前置

`dit` 聚合状态必须为 `vram`（全部 56 unit）。采样前向每一步都会过
`token_refiner.preprocess_text_embeds`，refiner 缺席（partial 布局）在
admission 阶段就拒绝（`MISSING_UNITS` 409），不在运行中途失败。
TE / VAE / VEDA 均不需要在场。

```text
POST /api/weights/dit/disk2ram {"mode":"all"}   # 202 轮询到 done
POST /api/weights/dit/ram2vram {"mode":"all"}   # 202 轮询到 done
```

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
  -> Runtime.operation_admission（dit 必须 vram）
  -> QueueManager.submit_operation('infer.dit_probe')
  -> Runtime.run_infer_dit_probe -> dit_probe.run_probe
       合成 cond（局部 RNG，随机 [1, N, 5120] bf16 + 全 1 tags）
       -> diff_model.preprocess_text_embeds（token_refiner 真实参与）
       -> _empty_av_latent（与视频任务同款空 latent）
       -> RandomNoise / BasicGuider / KSamplerSelect(res_multistep)
          / BasicScheduler(simple) / SamplerCustomAdvanced
       warmup + repeat 轮，逐轮同步 CUDA、重置 peak、显存余量检查
       结束后对比 dit 驻留快照（units 设备/计数/字节）
```

- **不带参考图**：不建 `minimax_refs`，不涉及 VAE；探针自包含。
- cond 宽度固定 5120（本 checkpoint 的 ClipProj 输出维），tags 全 1
  （纯文本 token）。
- 与 `/api/jobs` 真任务共用同一采样链，不复制采样数学。

## 4. 结果（operation kind = infer.dit_probe）

```text
kind = dit_probe
synthetic_inputs = true
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
| `MISSING_UNITS` | 409 | dit 非 vram（admission 阶段） |
| `INSUFFICIENT_VRAM` | 409 | 某轮前可用显存低于 min_free_mb |
| `CUDA_OOM` | 409 | 采样真实 OOM，不自动缩小规模/卸载其它模型重试 |
| `NONFINITE_LATENT` | 422 | 采样输出含 NaN/Inf |
| `WEIGHT_MUTATED` | 500 | 采样前后 dit 驻留快照（设备/计数/字节）变化 |

执行阶段失败查 `operation.status=failed` 与 `error.details`
（含 scale/steps/failed_iteration/warmup_completed/runs_completed/memory）。

## 6. 已验证范围

`tests/_selftest_dit_probe.py`（8 用例，fake 采样链 + CPU torch）覆盖：
严格校验、合成 cond 确定性、warmup/repeat 指标、显存不足前拒绝、非有限
latent、驻留突变 fail-closed、admission/capabilities。**没有真实 GPU 采样
验证**；真机最小验收建议：`{"scale": {"cond_tokens": 512, "width": 640,
"height": 960, "length": 124}, "steps": 1, "repeat": 1, "min_free_mb": 512}`，
veda 档下对照 `veda_calls.cumulative`。
