# 文本编码器（TE）原子化 API 文档

本仓库支持把 DiT server 的本地文本编码器（TE）encode 过程拆成单步 API，方便在**远程高密度 GPU**下一颗一颗指令地调试，避免一次性密集操作把 GPU 卡死。

## 一、前置条件

启动 `h3_dit_server.py` 时必须加载本地 TE 之一：

```powershell
# 4B + ClipProj 投影路径（推荐）
python h3_dit_server.py --te-4b D:\qwen3vl_4b_minimax_h3_int8_convrot.safetensors --te-proj my_projection

# 或 GGUF 路径
python h3_dit_server.py --te-gguf D:\qwen3vl_32b_minimax_h3_q2_k.gguf
```

加载成功后 `/api/status` 的 `ready` 为 `true`。

## 二、端点

### GET /api/te/state

返回当前 TE 状态机快照、TE 权重所在 device、VRAM 占用、当前步骤。

**响应示例：**

```json
{
  "ok": true,
  "te_state": {
    "step": "vae_swapped_back",
    "te_device": "cpu",
    "vram_alloc_gb": 18.25,
    "vram_reserved_gb": 19.10,
    "vram_free_gb": 4.15,
    "vram_total_gb": 22.5
  }
}
```

### POST /api/te/step

单步触发一个原子操作。

**请求体：**

```json
{
  "action": "te_load_vram"
}
```

**action 列表：**

| action | 说明 | 额外字段 |
|--------|------|----------|
| `vae_swap_out` | VAE 从 GPU 搬到 CPU | 无 |
| `te_load_vram` | TE 权重 RAM → VRAM，**不推理** | 无 |
| `te_offload_ram` | TE 权重 VRAM → RAM | 无 |
| `preprocess` | 参考图预处理（CPU） | `image_b64s`, `mode`, `width`, `height`, `ref_image_size` |
| `te_encode` | 执行 tokenize + encode | `prompt`, `image_b64s?`, `mode`, `width`, `height`, `ref_image_size` |
| `vae_swap_back` | VAE 从 CPU 回到 GPU | 无 |
| `reset` | 重置状态机，VAE/TE 归位 | 无 |

**约束：**

- 队列非空（有 running/queued 任务）时返回 409。
- 调用 `te_encode` 前通常需要：
  1. `vae_swap_out`
  2. `te_load_vram`
  3. `preprocess`（或把图直接塞进 `te_encode`）
  4. `te_encode`
  5. `te_offload_ram`（可选，防占卡）
  6. `vae_swap_back`

**分步调用示例：**

```bash
# 1. VAE 让位
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"vae_swap_out"}'

# 2. TE 权重上卡（不推理）
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"te_load_vram"}'

# 3. 预处理参考图
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"preprocess","image_b64s":["<base64>"],"mode":"ref2va","width":640,"height":960}'

# 4. 执行 encode
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"te_encode","prompt":"a cat dancing","mode":"ref2va","image_b64s":["<base64>"]}'

# 5. TE 退回
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"te_offload_ram"}'

# 6. VAE 归位
curl -X POST http://127.0.0.1:8190/api/te/step \
  -H "Content-Type: application/json" \
  -d '{"action":"vae_swap_back"}'
```

### POST /api/te/probe

只把 TE 权重反复从 RAM 搬到 VRAM 再退回，**不执行任何推理**。专门测试远程 GPU 是否能承受 TE 上卡/退卡操作。

**请求体：**

```json
{
  "cycles": 3
}
```

**响应示例：**

```json
{
  "ok": true,
  "cycles": 3,
  "results": [
    {"load": {"elapsed_s": 1.23, "device": "cuda:0", "alloc_gb": 21.5},
     "offload": {"elapsed_s": 0.45, "device": "cpu"}}
  ],
  "te_state": {...}
}
```

## 三、完整封装入口

如果只是想要一个调用就完成的 cond encode（VAE 让位 → TE 上卡 → encode → TE 退回 → VAE 归位），仍可使用：

### POST /api/encode

```bash
curl -X POST http://127.0.0.1:8190/api/encode \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat dancing","image_b64s":["<base64>"],"mode":"ref2va","width":640,"height":960,"save_cond":true}'
```

返回的 `stats` 包含 swap_out、encode、swap_back 各段耗时。

## 四、与普通视频任务的关系

加载本地 TE 后，`POST /api/jobs` 也会优先本地 encode，不再回调 cond server：

```text
/api/jobs 取条件优先级：
  1. /api/jobs/cond 直接喂 cond
  2. 本地 TE（如果启动时加载了 --te-4b/--te-gguf）
  3. cond_client.fetch() 回调本机 cond server
```

## 五、故障排查

| 现象 | 原因 | 处置 |
|------|------|------|
| `/api/te/step` 409 | 队列里有生成任务在跑 | 等任务结束或用 `/api/jobs/drop` 取消 |
| `te_load_vram` 报错 | TE 未加载或 GPU 没空间 | 检查启动参数，先 `vae_swap_out` 腾显存 |
| `te_encode` 报错但 TE 没退回 | 异常未捕获 | 调用 `reset` 强制 VAE/TE 归位 |
| GPU 卡死 | 单次密集指令过多 | 用 `/api/te/probe` 先测 cycles=1，逐步增加 |

## 六、设计注意

- 所有 `/api/te/step` 操作都是**串行独占**的，调用方需自行保证不与生成任务并发。
- `reset` 会尽量把 VAE 搬回 GPU、TE 退回 CPU；失败只记日志不抛异常。
- 状态机 `self._te_step_state` 只保存在内存里，服务重启后重置。
