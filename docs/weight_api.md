# 显式权重 API（统一四动作）

更新时间：2026-11-21。目标：**所有模型权重只能通过 API 管理**，无参数启动是纯空壳；
推理与权重操作共用同一个单 worker 执行器。

## 1. 模型与粒度

| model id | backend | 首版粒度 | unit |
|---|---|---|---|
| `dit` | `DitBackend`（A 的 `H3ComfyAdapter`） | block 级部分物化 | `dit.blocks.N`、`dit.refiner`、`dit.modules.*`、`dit.root` |
| `vae.video` | `VaeBackend`（Comfy VAE） | 整组 | `vae.video.model` |
| `vae.audio` | `VaeBackend`（Comfy VAE） | 整组 | `vae.audio.model` |
| `te` | `TeBackend`（Comfy CLIP + ClipProj） | `te.model` + `te.proj`（同进同出） | 见左 |
| `veda.bundle` | `VedaBackend`（`veda.bundle_compat`） | 整组 | `veda.bundle.model` |

`GET /api/weights` 与 `GET /api/weights/<model>` 返回同一份 registry 摘要：

```json
{
  "ok": true,
  "revision": 3,
  "models": {
    "te": {
      "model_id": "te", "state": "partial|absent|ram|vram|error",
      "generation": 2, "catalog_ready": true, "addressable": true,
      "recipe": {"kind": "4b", "path": "...", "fingerprint": {...}},
      "counts": {"absent": 0, "ram": 1, "vram": 1, "mixed": 0, "error": 0},
      "bytes": {"resident_cpu": 5123456789, "resident_cuda": 5123456789,
                "expected": 5150000000},
      "units": [{"id": "te.model", "state": "vram", "device": "cuda:0",
                 "resident_bytes_cpu": 0, "resident_bytes_cuda": 5150000000,
                 "expected_bytes": 4858559240, "last_error": null}, ...]
    }
  }
}
```

state 是聚合值：含 mixed/error 即 `error`；部分缺失为 `partial`；全部 CPU 为
`ram`；全部 GPU 计算副本就绪为 `vram`。TE 投影可保留 CPU 主副本，CUDA 计算
副本完整就绪后 te.proj 仍为 vram；两端字节分别报告。bytes 是**真实 storage**（量化展开、按 storage 去重），
不是 `numel * element_size`。

## 2. 四动作

```text
POST /api/weights/<model>/<action>
action ∈ disk2ram | ram2vram | vram2ram | purgefromram
```

统一请求体：

```json
{"mode": "all"}                        // 或 {"mode": "units", "units": ["dit.blocks.12"]}
{"model": "te"}                        // 可省；省则用 URL 里的 model；不一致 400
{"dry_run": true}                      // 只读 header/catalog + 资源摘要，不构造/不搬
{"request_id": "client-abc"}           // 可选：同 payload 幂等返回同一个 operation
{"recipe": {"path": "...", "kind": "4b", "proj": "...", "mmproj": "..."}}
                                       // 仅 disk2ram；不传则用 defaults-file
```

返回 **202**：

```json
{"ok": true, "operation_id": "op_...", "status": "queued",
 "operation_url": "/api/operations/op_...", "model": "te", "action": "disk2ram"}
```

`GET /api/operations/<id>` 轮询；`result` 里有 `completed / failed / not_attempted /
dry_run / generation / model_state`，成功动作另带 `audit` 或 `proof`。

### 状态迁移

| 动作 | 合法 → | no-op | 拒绝 |
|---|---|---|---|
| disk2ram | absent → ram | 已 ram/vram | mixed/error；recipe 变更 |
| ram2vram | ram → vram | 已 vram | absent；mixed/error |
| vram2ram | vram → ram | 已 ram/absent | mixed/error |
| purgefromram | ram → absent | 已 absent | 任何 CUDA 驻留（先 vram2ram） |

### TE 投影与登记

`te ram2vram` 同时完成 TE 原生 Comfy 登记，以及投影系数、residual MLP 和
sink 的 GPU 准备，不推理。编码要求全部已配置的 TE unit 为 vram；推理不会
补登记、补建投影、为自己卸载其它模型。投影 CPU 主副本保留供下次上卡。
原子投影准备失败可以留下真实 partial 状态，显式重试或 vram2ram；整组 TE
的 vram2ram 也可尝试恢复 error/mixed，完成后必须重新通过物理审计。
ClipProj encoder 仅支持已核对的源码版本，不匹配返回 `UNSUPPORTED_CLIPPROJ`。

### 关键错误

| code | HTTP | 场景 |
|---|---|---|
| `MODEL_NOT_FOUND` | 404 | 未知 model |
| `<MODEL>_NOT_CONFIGURED` | 409 | defaults-file 没有该模型 |
| `SHARED_UNIT_REQUIRED` | 409 | TE 只选 `te.model` 或 `te.proj` |
| `MISSING_UNITS` | 409 | 上卡/编码所需 unit 不在目标状态，带 `suggested` |
| `INVALID_SELECTOR` | 400 | 未知 unit / 区间越界（不 clamp） |
| `TRANSITION_CONFLICT` | 409 | mixed/error 未恢复就继续 |
| `CUDA_RESIDENT` | 409 | purge 时仍有 CUDA 存储 |
| `PURGE_INCOMPLETE` | 500 | weakref 还活着，拒绝谎报 absent |
| `UNINSPECTABLE_WEIGHTS` | 500 | 量化包装无法审计，fail-closed |
| `WEIGHT_MUTATED` | 500 | 推理期间权重设备被内部流程改变 |

## 3. 显式 TE 条件编码

```text
POST /api/weights/te/disk2ram   {"mode": "all"}   # 4.85GB 读盘，构造在 CPU
POST /api/weights/te/ram2vram   {"mode": "all"}   # 权重上卡（不推理）
POST /api/infer/encode          {"prompt": "...", "image_b64s": ["..."],
                                 "mode": "ref2va", "width": 640, "height": 960}
GET  /api/operations/<id>                          # 轮询到 done
POST /api/weights/te/vram2ram   {"mode": "all"}   # 下卡（清 ClipProj GPU 缓存）
POST /api/weights/te/purgefromram {"mode": "all"} # 释放并证明零存储
```

`/api/infer/encode` 固定 202；`result.artifact` 是 `cond_<hex>` artifact：

```json
{"artifact": {"id": "cond_...", "filename": "cond_te_....safetensors",
              "bytes": 4100000, "sha256": "..."},
 "stats": {"n_tokens": 1234, "cond_shape": [1, 1234, 5120],
           "tokenize_s": 0.1, "encode_s": 4.2, "mode": "ref2va"},
 "audit": {...}}
```

下载：`GET /outputs/stage/<filename>`（支持单段 Range）。

旧 `POST /api/encode` 保留同步形态：仍由同一个 worker 执行，有界等待（1800s），
超时返回已有的 202 回执，**不再自动 VAE 让位/TE 上卡**；TE 未上卡时 409。
旧 `POST /api/te/step`、`/api/te/probe` 保留为 legacy 调试路径，也由同一
worker 执行；有界等待 1800s，超时返回 operation 回执，不再在 HTTP 线程里
并发搬模型或编码。

## 4. 启动与配置

```powershell
python h3_dit_server.py --host 127.0.0.1 --port 8190 `
  --runner-dir C:\Users\sheepserver\Desktop `
  --defaults-file C:\Users\sheepserver\Desktop\h3_explicit_defaults.json
```

`defaults.json` 只描述配方，不触发加载：

```json
{"models": {
  "te":  {"path": "C:\\Users\\sheepserver\\qwen3vl_4b_int8_convrot.safetensors",
          "kind": "4b",
          "proj": "mmh3-4b-ClipProj-celeb-mlp.safetensors"},
  "dit": {"path": "C:\\Users\\sheepserver\\10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"},
  "vae.video": {"path": "D:\\minimax_h3_video_vae_int8_convrot.safetensors"},
  "vae.audio": {"path": "D:\\minimax_h3_audio_vae_fp32.safetensors"},
  "veda.bundle": {"path": "D:\\minimax_h3_r2va_veda_preview_fp8.safetensors"}
}}
```

`te.kind` 可省略：有 `proj` → `4b`；`.gguf` → `gguf`；否则 `checkpoint`。
`te.proj` 可以是 `models/clip_projections` 下的文件名，也可以是绝对路径。

## 5. 已知边界

* DiT 已接采样外壳：全量 `ram2vram` 后用 B 的 `wrap_h3_skeleton` 绑定 BaseModel /
  ModelPatcher + 固定 12/3 SigmaShift；`/api/jobs/cond` 可直接在该 resident DiT
  上采样。partial（未全量）时不会创建外壳。
* VAE / VEDA 首版是整组 unit，尚未按 encoder/decoder/层拆分。
* 量化 VAE（int8_convrot）的 `ram2vram` 不是裸 `.to()`，而是在目标设备重建
  （先释放 CPU 实例）。原因：CPU 构造后 `.to(cuda)` 不同步量化 Linear 的设备
  选择，decode 时 int8 attention 的 q/k/v 仍在 CPU。
* sample 只要求 `dit` 在显存；`vae.video` / `vae.audio` 允许停在 RAM。decode
  阶段由 pipeline 调 `WeightService.prepare_vae()` 显式上卡（量化 VAE 走重建）。
  同时驻留 14GB DiT + 2.7GB VAE 会在 2080Ti 22GB 上 OOM。
* C 分支的 ComfyBindings 已单独接入 TE；attention 与 VEDA 严格 resident
  尚未合并，`/api/status` 的 `attn` 仍为 none。不能把 TE 的验证当作全程序
  所有推理路径的驻留验证。
* `purgefromram` 的跨 unit 事务不承诺全回滚：报告 completed/failed/not_attempted
  和实际 snapshot。
* `/api/te/step`、`/api/te/probe` 已迁入 operation；旧 `/api/jobs*` 仍保留 Job 外形。
