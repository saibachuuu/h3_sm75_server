# h3_sm75_server — MiniMax H3 连续生成双服务器

本机跑文本编码（cond server），远程 2080 Ti 跑 DiT 生成（dit server + WebUI）。
模型常驻、串行队列、手动逐单提交、产物直接 ffmpeg mux 成 **mp4（不落 PNG 序列/wav）**。

## 架构

```
浏览器 ── http://127.0.0.1:8190 ──[ssh -L 8190]──▶ 远程 h3_dit_server.py (:8190)
                                                       │ 需要条件时回调
                                                       ▼
本机 h3_cond_server.py (:8189) ◀──[ssh -R 9310]── 远程访问 127.0.0.1:9310
（Qwen3-VL-32B int4 文本编码器常驻，内容哈希缓存）
```

## 快速开始（三步 + 浏览器）

**① 本机 cond server**（常驻不关；已在跑就跳过）
```powershell
Start-Process C:\Users\88888888\comfyui\.venv\Scripts\python.exe `
  -ArgumentList 'h3_cond_server.py' `
  -WorkingDirectory <本目录> -WindowStyle Hidden
```
冷启动编码器约 15~60 秒，`http://127.0.0.1:8189/health` 可查。

**② 远程 dit server**（模型加载约 210 秒）
```powershell
ssh -p <port> sheepserver@<host> "C:\Users\sheepserver\Desktop\ComfyUI_windows_portable\python_embeded\python.exe -X utf-8 C:\Users\sheepserver\Desktop\h3_sm75_server\h3_dit_server.py 2>&1 | Tee-Object -FilePath C:\Users\sheepserver\Desktop\_dit_server.log"
```

**③ 双向隧道**（保持开着；**必须显式 127.0.0.1，不能写 localhost**——
sshd 转发时 `localhost` 会解析成 IPv6 `::1`，而服务只绑 IPv4，会表现为
"连上就被掐断"）
```powershell
# 本机已有 tunnel.ps1（已预填 host/port/user，直接跑）：
.\tunnel.ps1
# 等价手动命令：
ssh -N -p <port> -L 8190:127.0.0.1:8190 -R 9310:127.0.0.1:8189 sheepserver@<host>
```

**④ 浏览器**打开 `http://127.0.0.1:8190/`

## WebUI 使用

- 提示词 + 参考图必填；默认参数即已验收的单 pass 基线（res_multistep/simple、4 步、
  640×960、124 帧、refine_steps=0）；seed 填 -1 随机；
- 参考图：**ref2va 1–9 张**（顺序即 `<Picture i>` 编号），fl2va 仅 1 张；
  每图独立缩放 + 逐张 VAE 编码 → `minimax_refs` 多块（参考 token 每步参与，多图更慢）；
- 队列串行执行（单 GPU），排队中的任务可取消；
- 画廊直接播放/下载 mp4，"参数"链接是同名 JSON sidecar；
- 页头芯片诊断：`服务失联`=②没跑、`cond: 隧道不通`=③没开、`cond: 加载中`=等 ①。

## 常用服务参数

```text
h3_cond_server.py
  --port 8189  --model D:\qwen3vl_32b_minimax_h3_int4_convrot.safetensors
  --cache-dir ./cond_cache   --no-int4-mma(调试)

h3_dit_server.py
  --port 8190  --cond-url http://127.0.0.1:9310
  --unet D:\10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors
  --attn int8|sage|w8a8|sdpa   ← 服务级；sage/w8a8 仅调试（sage 全尺寸 0/2 崩溃）
  --w4a8_backend turingutils   --mlp_chunk 8192  --runner-dir(自动探测)
```

## API（供脚本化提交）

```text
GET  /api/status                     服务/cond/显存/队列状态
GET  /api/defaults                   表单默认值与可选采样器
GET  /api/jobs                       任务列表
GET  /api/jobs/<id>                  任务详情（含日志尾部）
POST /api/jobs                       JSON {prompt, image_b64, mode, width, height,
                                     length, steps, sampler, scheduler, seed,
                                     refine_steps, ref_image_size, sigmas?} -> {job_id}
POST /api/jobs/drop {id}             取消排队任务
GET  /outputs/<id>.mp4|.json         成果（支持 Range）
```

cond server：`GET /health`、`POST /encode`、`GET /files/<hash>.safetensors`。

## 设计要点

- **模型常驻**：DiT + 双 VAE + token_refiner 均不卸载（refiner 二次 refine 0.01s）；
  实测空闲显存 18.25 GB，解码峰值 20.28/23.62 GB（余量 ~3.3 GB）。
- **cond 内容哈希缓存**：同 prompt+图+参数免重复编码（命中 <0.1s）。
- **cond 文件 slim**：只含文本编码器输出（cond/minimax_token_tags…），不带图像像素；
  参考帧由 dit server 从任务 `image_b64` 本地解码 + 缩放 + VAE 编码
  （目标尺寸 `common.protocol.target_frame_size`，与 cond 侧 Qwen 输入一致）。
- **无 warmup 逻辑**：DiT warmup 已被实测证伪（崩溃跟随全尺寸计算，与"第一次"无关）。
- **ffmpeg 直出 mp4**：rawvideo 管进 + 临时 wav 管进，h264+AAC，mux 完删临时文件。
- **latent 探针**：采样输出含 NaN/Inf 直接判失败，不送解码。
- 依赖 h3-runner 的 `infer_h3_turing.py` / `h3_mlp_chunk.py` 作为库（不修改 runner）。

## 目录结构

```
h3_sm75_server/
├── h3_cond_server.py      # 入口A（本机）
├── h3_dit_server.py       # 入口B（远程）
├── tunnel.ps1             # 本机双向隧道脚本（-L 8190 / -R 9310，已预填 host/port/user）
├── common/protocol.py     # API 约定与参数校验
├── cond_svc/              # 编码服务（Encoder 复用 embed_server.Encoder）
├── dit_svc/               # runtime / pipeline / job_queue / cond_client / http_api
├── webui/index.html       # 单页中文 WebUI（增量轮询，不整页重建）
├── cond_cache/            # 运行时（git 忽略）
└── outputs/               # 运行时（git 忽略）
```

## 故障排查

| 现象 | 原因 | 处置 |
|---|---|---|
| 页面"服务失联" | 远程 dit server 没跑/被杀 | 重启②（注意：停本地 ssh 不一定杀得死远程进程，需远程显式杀） |
| "cond: 隧道不通" | 隧道没开或用了 localhost 写法 | 重开③（显式 127.0.0.1） |
| 提交报"模型加载中" | ②还在加载 ~210s | 等就绪 |
| 任务失败含 ffmpeg rc | mux 出错 | 看任务日志尾部的 ffmpeg stderr |
| 远程端口被占 | 上次服务没退干净 | 远程查 8190 占用并杀进程后重启② |
