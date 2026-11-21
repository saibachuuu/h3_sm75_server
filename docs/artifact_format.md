# artifact 格式与 stage 目录约定

本文档对应 `dit_svc/artifacts.py`（存储层 + 序列化层）与
`dit_svc/api_validation.py`（新协议校验）。接口形状依据
`plan_20261121_152726/03_api_contract.md` §6/§7 与 `04_inference_and_attention.md`
§5/§6。

## 1. 目录布局

```text
<out_dir>/
  stage/
    <uuid>.safetensors             # 唯一文件名，创建时随机生成，不用秒级时间戳
    <uuid>.safetensors.json        # manifest：id/sha256/bytes/metadata/created_at
    .tmp_xxxxx.part                 # 原子写中间文件，正常流程不会残留
```

- artifact ID 形如 `cond_<32位hex>` / `latent_<32位hex>`（`uuid4().hex`），
  与文件名解耦；客户端只拿 ID，文件名不表达任何语义。
- manifest 与文件分开写：文件先原子落地，再写 manifest；重启时只扫描
  `stage/*.json` 重建索引，**不恢复模型驻留、不恢复 operation 状态**。

## 2. 安全路径规则（`validate_stage_name` / `safe_join` / `resolve_any`）

下载与引用解析只接受 stage 内**单层文件名**，以下一律返回 `INVALID_PATH`（400）：

| 输入 | 拒绝原因 |
|---|---|
| `C:\out\stage\x.safetensors` | 绝对路径 / 盘符 |
| `\\\\host\share\x` | UNC |
| `a\b`、`a/b` | 反斜杠 / 目录分隔 |
| `..`、`.` | 精确匹配的穿越名 |
| `%2e%2e%2fx`、`x%00y` | 百分号编码（解码后与原文不同即拒绝） |
| `x.`、`x ` | Win32 会去掉的尾随点/空格 |
| `con`、`lpt1` | Windows 保留设备名 |
| 已存在的 symlink / reparse point | 逐段 `lstat` 检查 |

包含关系判断用 `os.path.realpath` + `os.path.commonpath`，**不做字符串前缀
比较**（防止 `stage_evil` 被 `stage` 前缀骗过）。

绝对本地路径不属于新 infer 协议：`sample`/`decode` 的 `*_file` 必须相对 stage；
checkpoint 的绝对路径是另一个受信配置能力，不在这里放开。

## 3. 原子写与预算

- `atomic_write_bytes` / `ArtifactStore.commit_stream`：同目录 `mkstemp` 唯一
  临时文件 → `write` + `fsync` → `os.replace` 原子改名 → 目录 `fsync`（Windows
  上目录 fsync 尽力而为）。任何异常都删除临时文件，不留半写产物。
- 预算在写前检查：`stage` 配额（默认 32 GiB）+ 磁盘可用量减预留（默认 2 GiB）。
  不足返回 `RESOURCE_EXHAUSTED`（503），`budget()` 同时给出 `fits`/`reasons`
  供 dry-run 报告。
- 引用保护：consumer 期间 `acquire(id)`，清理与 `delete` 会跳过/拒绝被引用的
  artifact（`ACTIVE_BINDING` 409）。`cleanup()` 只删过期且未引用的条目。

## 4. cond 格式 `h3.cond.v1`

必需 tensor key：`cond`、`minimax_token_tags`。
可选：`attention_mask`、`pooled_output`（以及调用方显式声明的 `extensions`）。

- 保存前统一 CPU contiguous，**不改 dtype**（cond 常是 bf16）。
- `minimax_token_tags is None` 时拒绝保存（`INVALID_ARTIFACT` 422）——不能写出
  一个缺必需 key 的"成功"文件。只有已验证的适配器可以传 `synthesize_tags`
  生成等价 tags，并在 metadata 写 `token_tags_source=synthesized`。
- `tags` 的形状必须是 `cond.shape` 的前缀（`[B,T]` vs `[B,T,C]`）。
- metadata strings：`format=h3.cond.v1`、`mode`、`width`、`height`、
  `token_tags_source`，以及调用方补充的 TE recipe fingerprint / 参考图摘要。
- legacy 文件（无 metadata）仍可读；缺必需 key 仍拒绝。

## 5. 音视频 latent 格式 `h3.avlatent.v1`

真实容器（remote `comfy/nested_tensor.py`）：

```python
class NestedTensor:
    def __init__(self, tensors): self.tensors = list(tensors); self.is_nested = True
    def unbind(self): return self.tensors
```

`nodes_minimax_h3.py` 的 `_av_latent` 固定返回
`NestedTensor((video[B,24,T,H//16,W//16], audio[B,32,2,T40]))`。序列化层因此：

1. **不猜 `torch.nested`**，只按 `.tensors` 列表 / `{video,audio}` 映射 / 二元
   序列三种结构取两个 tensor，再用 shape 判别谁是谁（5D 且 C=24 → video；
   4D 且 C=32 且第二维=2 → audio）。判别失败即 422。
2. 记录 `container_order`（`video,audio` 或 `audio,video`）与 `container_type`，
   读回时按原序重建；重建优先用已加载的 Comfy `NestedTensor` 类，找不到才退化成
   tuple。可用 `split_container=` / `build_container=` 注入自定义适配。
3. 保存 `video`、`audio` 两个 key，各自**原 dtype 的 CPU contiguous** tensor，
   不为省事统一 float32。
4. `noise_mask` 若存在：嵌套容器拆成 `noise_mask_video` / `noise_mask_audio`
   （按维度判别，不套 latent 通道规则）；单 tensor 则存 `noise_mask`。
   `batch_index` 只存字符串摘要（不假装是 tensor）。
5. **未知字段直接失败**（`INVALID_ARTIFACT` 422 + `unsupported_fields`），
   不静默丢弃。

metadata strings：

| key | 说明 |
|---|---|
| `format` | `h3.avlatent.v1`（必需） |
| `requested_length` | 用户请求帧数（必需） |
| `native_fps` | 固定 `24`（必需，其它值 400） |
| `width` / `height` | 画布 |
| `effective_latent_frames` | 网格对齐后的 `latent_t`，仅供诊断 |
| `frame_count` | 网格对齐后的真实帧数（`17k+5`） |
| `mode` | `ref2va` / `fl2va` |
| `video_shape` / `audio_shape` | JSON 数组字符串 |
| `dtypes` | JSON 对象字符串 |
| `container_order` / `container_type` | 重建提示 |
| `sampling_params_json` / `checkpoint_fingerprint` / `cond_fingerprint` / `adapter_layout_version` | 调用方补充 |

## 6. 时间轴与裁帧（不要从 latent_t 猜片长）

`temporal_shape` 会把帧数 snap 到 `17k+5` 网格。`artifacts.plan_temporal(length)`
给出与外层节点一致的数值（纯数值复制，不 import `comfy_extras`）：

| length | frame_count | latent_t | audio_t |
|---|---|---|---|
| 124 | 124 | 37 | 207 |
| 100 | 107 | 32 | 178 |
| 16 | 22 | 7 | 37 |

解码裁帧用 `decode_trim(decoded_frames, metadata)`：

- 以 `metadata.requested_length` 为准裁掉尾部 padding；
- 解码帧数少于 `requested_length` → `EXECUTION_FAILED`（500），不静默变短；
- 缺 `requested_length` → `INVALID_ARTIFACT`（422）。

带音频的 MP4 首版只允许 24fps；`api_validation.validate_decode_request` 在解码前
用 `UNSUPPORTED_FPS`（422）拒绝其它 fps，而不是输出可能音画错位的成片。

## 7. 数值探针

`probe_tensor_chunked(tensor, chunk_elems=4Mi)` 有界分块统计 NaN/Inf/min/max，
不额外分配整块 fp32 副本（124 帧整块 fp32 探针会 OOM，沿用旧代码的逐帧口径）。
`probe_av_latent` 汇总各 tensor 并给 `ok=False` 标记。

## 8. 验证边界（必须知道）

| 环境 | 版本 | 已测内容 |
|---|---|---|
| 本机隔离 env | Python 3.10.11 / torch 2.8.0+cpu / safetensors 0.8 | 路径、预算、manifest、容器拆装、cond/latent 读写、探针、时间轴（CPU） |
| 远端目标 | Python 3.12.7 / torch 2.13+cu132 / safetensors 0.7 | **未运行** |

因此：

- 只声明"CPU 上的 safetensors roundtrip 与 NestedTensor 容器拆装通过"，**不声称**
  远端 GPU 上的真实采样→保存→解码链路通过。
- `safetensors` 的 `save_file` / `safe_open` 在 0.7 与 0.8 的 API 形状一致
  （`save_file(tensors, path, metadata=dict[str,str])`、
  `safe_open(path, framework="pt", device="cpu")`），但 metadata 只支持字符串值，
  所以本模块显式把非字符串转成字符串。
- 真实 ComfyUI 的 `comfy.nested_tensor.NestedTensor` 与下载副本一致（提供
  `.tensors`/`.is_nested`/`unbind`）；重建时若该类未加载会退化为 tuple，调用方
  应通过 `build_container=` 注入真实类以保证类型一致。

## 9. 尚未覆盖

- decode 的 ffmpeg/mux 与 WAV 输出不在本模块（属 `dit_svc/infer.py`）。
- `GET /api/artifacts/<id>` 的 HTTP 路由由主代理在 `http_api.py` 接入；
  `ArtifactStore.public()` 已给出稳定摘要形状。
- operation 层错误统一到 `dit_svc/operations.py` 由主代理完成；本模块错误对象
  已带 `code`/`http_status`/`to_error()`，可直接映射。
- `..` 之外的 NTFS 8.3 短名（如 `PROGRA~1`）未单独拒绝；它不构成穿越，但若
  需要严格白名单可在 `validate_stage_name` 增加 `~` 检查（当前只拒绝以 `~`
  开头的名字）。
