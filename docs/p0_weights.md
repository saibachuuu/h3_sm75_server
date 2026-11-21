# P0 权重底座：可行性结论与证据（agent/explicit-weights）

> 本文档是 P0 阶段结论。所有“真实”指远端只读快照与真实 header，
> 本机无 GPU、无真实权重 payload 读取、无 CUDA import。
> 不得把 fake 域自测说成真实 partial 已通过。

## 1. 版本基线

| 项 | 本机（自测环境） | 远端（只读快照，未执行） |
|---|---|---|
| python | 3.10.11 | 3.12.7（portable embedded） |
| torch | 2.8.0+cpu | 2.13.0+cu132 |
| safetensors | 0.8（隔离 venv，仅本 worktree 自测用，未装全局） | 0.7.0 |
| comfy-kitchen | 无 | 0.2.40 |
| ComfyUI | 未提供（`_remote_probe` 为旧快照，不采信） | 5f415e31a（只读快照 `explicit_api_sources_20261121/comfy/`） |
| H3 模型类 | 未提供 | `minimax/model.py` 只读快照（同根） |
| checkpoint | 未提供（无 payload 读取） | 4 个真实 safetensors **完整 header**（见 §3） |
| GPU | 无（2080Ti 在远端，本机未验证任何 CUDA 路径） | 2080Ti 22GB（旧交接记载，未复核） |

CPU torch 2.8 与远端 2.13 的差异已记录：自测只用 CPU 张量语义
（storage 去重、`untyped_storage`、`meta` 设备），不依赖版本特有 API；
`F8_E4M3` dtype 字节数按 1 字节计（header 解析用，不做数值运算）。

## 2. 真实 Comfy 构造链证据（只读，未改快照）

源根：`C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_sources_20261121\`

1. **H3 detect 只需 key 存在性 + ~10 个形状**（`comfy/model_detection.py:401-433`）：
   `video/audio_patch_proj`、`final_layer.video/audio_out`、
   `blocks.0.attn.{q_norm,qkv_proj}`、`blocks.0.mlp.fc1`、
   `condition_proj`、`rope.inv_freq`，二选一 `adaln_t_table`（本 checkpoint
   走此路，省 `time_embedder.*`）或 `time_embedder.proj_in/out`；
   另需 `blocks.{N}.` / `token_refiner.blocks.{N}.` 前缀枚举（`count_blocks`）。
   结论：detect 可用 header-only 形状（`get_shape()`）满足，
   不读权重值。代码常量 `dit_svc/weights.py:H3_DESCRIPTOR_KEYS`。
2. **stock loader 是全量 eager，无 partial 路径**（`comfy/sd.py:2431-2524`）：
   `model_config_from_unet(sd)` → `get_model(new_sd)` → `ModelPatcher(...)`
   → `model.to(offload)` → `load_model_weights(new_sd, assign=dynamic)`。
   `get_model` 对 H3 直接 `model_base.MiniMaxH3(self, device=device)`
   （`supported_models.py:1006`），构造与权重无关。
3. **`strict=False` 是 R4 铁证**（`comfy/model_base.py:373-388`）：
   `load_state_dict(to_load, strict=False, assign=...)`，缺 key 只 warning。
   部分 sd 喂原函数会留下随机初始化参数 → partial 绝不能复用原函数，
   必须走 adapter 的 skeleton + 逐 unit 物化。
4. **无存储骨架可行 plumbing**（`minimax/model.py:476-518`）：
   `MiniMaxH3Model(..., device=None, operations=None)`，全部 Linear 经
   `operations.Linear(..., device=device)`，buffers 用 `torch.empty(...)`，
   `self.blocks = nn.ModuleList([... for _ in range(num_layers)])`
   （索引稳定，不重排；purge 只许换槽位、不许删）。
   默认值与真实 header 一致：`num_layers=50`、`hidden=5376`、
   `token_refiner_num_layers=2`。`device="meta"` 骨架待真实类实测，
   `operations=pick_operations` 量化分支待 `comfy/ops.py` 核对（未读）。
5. **拦截目标签名已确认**（`comfy/model_management.py:939`）：
   `load_models_gpu(models, ...)` 维护全局 `current_loaded_models` 并搬运/
   卸载 clone —— P3 resident 路径必须经 `dit_svc/comfy_bindings.py`
   受控（本 stage 不做，只记位点）。

## 3. 真实 header catalog fixture（4 checkpoint，全覆盖零 unmapped）

header 来源：`remote_environment.json`（`models_constructed=0`、
`cuda_imported_by_probe=false`，纯 header 解析）。

| # | 文件 | keys | catalog units | 常驻字节（header 算术） |
|---|---|---|---|---|
| 0 | DiT `...w4a8_14gb_optimized.safetensors` | 1294 | 58：`dit.blocks.0-49` + `refiner` + `modules.{video,audio}_patch_proj/final_layer/adaln_basis/adaln_mean/adaln_t_table/rope` | 13.998GB（I8 10.597 + BF16 1.688 + F8 1.700 + F32 0.012 + U8 描述符） |
| 1 | video VAE int8 | 850 | 3：`encoder(116)` / `decoder(728)` / `shared(6: quant/post_quant/latents)` | 2.811GB |
| 2 | audio VAE fp32 | 917 | 3：`encoder(110)` / `decoder(777)` / `shared(30: pre_block/dec_in/logs/mean/proj/latents)` | 0.605GB |
| 3 | VEDA fp8 preview | 200 | 50：`layers.{0-49}` = `proj_q/proj_k(F8)` + `__scale(F32)` | 0.275GB |

选择器冒烟（真实 catalog）：`dit.blocks.0-2` → 3 units；
`veda.bundle` → 50 units；`all` = catalog 序。

**量化异构（同 checkpoint 内两套布局，adapter 必须分别处理）：**
`_quantization_metadata.format_version=1.0`，共 200 quant layers；
blocks.0 风格：`I8 qdata + F32 weight_scale([N,1] per-row) + U8 comfy_quant[82B 描述符]`
（例 `qkv_proj I8[21504,5376]`≈115.6MB + scale 86KB）；
blocks.10+ 风格：`F8_E4M3 qdata + weight_codebook + weight_s_channel + weight_s_rel`
（每 unit 26 keys / ~0.23GB，前者 18 keys / ~0.39GB）。
物理账必须按 `qdata+scale/codebook/s_rel` 实算，不得用逻辑 fp 尺寸冒充——
`storage_inventory` 已按底层 storage 去重实现，fake 域验证通过，
真实量化类（`comfy/ops.py` 待读）接入后复验。

**VAE encoder/decoder 独立性未证**：`shared` 暂收 `quant/post_quant`、
`pre_block`、`latents_*` 等；`encoder` 与 `decoder` 是否共享 storage
需真实张量 inventory 实测（P2/P3），现按“可单独选择、独立性 unknown”记录，
不宣称内部任意部分加载。

## 4. P0 结论

通过：
- (a) recipe/header/catalog 全链（标准库可 import，顶层无 torch）；
- (b) 稳定 unit 选择器（id / 组 / 闭区间，无 glob；真实 catalog 验证）；
- (c) 真实 storage inventory（views/buffers/量化内部/dict 别名去重，
  meta 零字节；fake+CPU 真 torch 验证）；
- (d) 严格四动作状态机（no-op 成功语义、CUDA 残留 purge 拒绝、
  recipe 变更拒绝、recover 规则）；
- (e) 真实 H3 构造证据链（§2）：detect 形状集、strict=False 禁区、
  meta-device 骨架 plumbing、ModuleList 槽位稳定。

接口约定（供主代理集成）：`WeightRegistry.snapshot()` /
`model_state()` / `resolve_units(model,mode,units)`；
`apply_disk2ram/ram2vram/vram2ram/purgefromram`（底座编排，见
`dit_svc/model_adapters.py` 的 fake 联动实现）返回可 JSON 摘要；
局部错误类型（`WeightsError` 族）由主代理映射到公共 `OperationError`。

## 5. 未验证事项（不得宣称通过）

1. 真实 GPU 上卡/量化加载/显存数字（本机无 CUDA；fake move 只改逻辑 device）。
2. 真实 `MiniMaxH3Model(device="meta")` 骨架 + `mixed_precision_ops` 量化分支
   （`comfy/ops.py:1256-1397`、`quant_ops.py` 的 consumable 契约未读；
   真实 `comfy-kitchen` QuantizedTensor 内部字段未展开实测——inventory 走
   通用 `__dict__` 展开，需复验）。
3. VAE encoder/decoder 与 shared 的真实 storage 边界（§3 已标 unknown）。
4. TE（Qwen3VL 32B + ClipProj）真实资源边界（本次无 TE header）。
5. `load_models_gpu`/CLIP/VAE/sampler 拦截与 VEDA resident engine（P3/P4）。
6. 远端 torch 2.13 与本机 2.8 的行为差（仅标准张量语义自测，无版本特有 API）。
   已知一条：torch 2.8 禁止跨 device 原地 `set_`/`p.data=` 赋值
   （purge 已改槽位替换，见 `h3_comfy.purge_unit`）。

## 6. 测试（隔离 venv：`explicit_api_env_20261121`，未动全局）

```bat
CD C:\Users\88888888\work-on-comfyui\h3_sm75_server_wt_weights
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_weights_logic.py
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_storage_inventory.py
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_partial_lifecycle.py
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_registry_review.py
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_h3_adapter.py
C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests/_selftest_real_header_catalog.py
```

- `_selftest_weights_logic`：stdlib-clean import、真实 H3 key 命名 catalog
  （7 units/12 keys fake 头）、cover 拒绝 leftover、选择器严格性、
  四动作矩阵、recipe 变更拒绝。ALL OK。
- `_selftest_storage_inventory`：CPU 真 torch —— 视图单 storage、
  module params+buffers、全量化内部展开、dict 别名去重、meta 零字节。ALL OK。
- `_selftest_partial_lifecycle`（P2）：fake 真选 key（access log 作证）、
  未选无 storage、partial purge 保槽位、补回不碰兄弟 storage（data_ptr 比对）、
  CUDA 残留 purge 拒绝、全量 purge 回 absent。ALL OK。
- 真实 header catalog 验证：§3 表格（4 checkpoint，`check_cover` 零 unmapped）。
- `_selftest_registry_review`（主代理 m00424/m00699/m00830 审查回归）：
  same_source 含 kind/options/version、无驻留换源清旧 catalog 且 generation
  单调、驻留换 catalog 拒绝、未知 unit 不得当 absent 通过、四动作遇
  error/mixed 全拒（recover 必经）、recover 必须附 adapter 实测
  release_proof（双零+空 digest，否则 ghost absent 拒绝）、输出深拷贝隔离、
  snapshot 修订号/counts/bytes、header 检验矩阵（32MiB 界/dtype/shape/
  offsets/EOF/重复 key）、签名钉死 dtype+shape+expected（同 key 集换 shape
  驻留拒）、audit 拒 META_MISSING/META_REMAINS、noncompute 分类。ALL OK。
- `_selftest_probe_helpers`：parse 拒 all/空、resolve 后 unit 数+1GiB 源端
  双预算（range 展开 50units 拒）、schema 不符 unverified+exit、staging
  weakref 释放证明（含残留检出）、真 peak RSS、env 无 CUDA 字段。ALL OK。
- `_selftest_h3_adapter`：stub 语义（device 透传、meta 零 storage、strict
  关门、purge 保槽位、registry 锁并发）+ 无 Comfy 报 ComfyUnavailable；
  stub 故意复现真实 bug（adaln_t_table/rope.inv_freq 无 device 分配），
  骨架靠 `torch.device` 上下文兜底并断言两 buffer 落 meta。ALL OK。
- probe 安全（m00830，主代理远端运行，本机只做静态+00 实测）：
  00 仅 importlib.metadata 版本只读，零 torch.cuda.*（已本地实测）；
  10/20 import comfy 前清 argv + 设 cpu/gpu_only，前后断言
  `cuda.is_initialized()==False`；20/21 resolve 后双预算（≤2 units、
  源端合计 ≤1GiB，range 展开拒）；schema 不符 verdict=unverified+exit 2；
  20 purge 前 staging（key_tensors/sub）释放 + weakref 无残留证明；
  报告真 CPU peak（PeakWorkingSetSize/ru_maxrss）而非 rss delta。
  静态核实：快照内无顶层 torch.cuda 调用、无 sage/flash 引用、
  xformers 仅 import 时加载（不执行 attention：无 forward、无 sampler）。
- `_selftest_real_header_catalog`：按远端 header 报告复刻的布局保真合成头
  （双量化伴生键/adaln/rope/refiner/patch/final），catalog 全覆盖、
  单位划分、字节账、选择器、waived 精确忽略。ALL OK。
