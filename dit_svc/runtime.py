"""远程 DiT 运行时：一次性加载常驻 + 内核/attention 安装。

对照 infer_h3_turing.py 的 main() 移植，两处刻意不同：
  1. **不含任何 warmup 逻辑**（DiT 部分的 warmup 已按需求删除）；
  2. token_refiner + condition_proj **常驻不卸载**（服务器逐任务复用；
     VRAM 预算实测：refiner 常驻后采样峰值 ~17.5GB、解码峰值 ~18.6GB < 22.5GB）。

解码与 DiT 常驻的冲突已用 10/9 基线日志排除：runner 的 "DiT release" 实际只
释放 2.8GB（权重被引用驻留），解码本来就在 DiT 权重常驻下跑，峰值 16.70/17.10GB。
"""
import gc
import os
import shutil
import sys
import threading
import time
import traceback

from dit_svc import veda_attn_star7  # noqa: F401  veda NVML/warmup 工具
from dit_svc.weight_service import WeightService
from dit_svc.te_residency import TeResidency
from dit_svc.operations import OperationError


def _all_request():
    """Legacy te_* 端点复用的全量请求；语义与 POST /api/weights/te/all 一致。"""
    return {"action": "all", "models": ["te"], "mode": "all",
            "units": [], "dry_run": False, "request_id": None,
            "recipe": None, "recipes": None, "options": None}


def find_ffmpeg():
    """远程 ffmpeg：优先 imageio_ffmpeg 自带二进制（ComfyUI python_embeded 已带 7.1）。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


class Runtime:
    """Environment availability and compute readiness are separate states."""

    def __init__(self, args, rt):
        self.args = args
        self.rt = rt                    # infer_h3_turing 模块（工具函数来源）
        self.env_ready = False
        self.compute_ready = False
        self.device_healthy = True
        self._env_lock = threading.RLock()
        self.torch = None
        self.mm = None
        self.attention = None
        self.minimax_model = None
        self.comfy_kitchen = None
        self.device = None
        self.error = None
        self.stage = "初始化"
        self.stage_t0 = time.perf_counter()
        self.started_at = time.time()
        # 模型（load 后填充）
        self.unet = None
        self.model_shifted = None
        self.vvae = None
        self.avae = None
        self.diff_model = None
        self.attn_override = None
        self.veda_patch = None      # --attn veda 时：star7 VedaPatch（运行期统计）
        self.ffmpeg = None
        self.attn = getattr(args, "attn", "none")
        # 运行时 attention 档位（attn.install）：factory 快照与最后一次成功安装的请求
        self._attn_factory = None
        self._attn_installed_request = None
        # 分块卸载 / 文本编码器常驻（--offload-dit-blocks / --te-gguf / --te-4b）
        self.offloaded_blocks = []
        self.text_encoder = None
        self.te_proj = None         # ClipProj 包装后的 4B 编码器（--te-4b + --te-proj）
        self.te_load_s = 0.0
        self.te_swaps = 0           # 已发生的 VAE↔TE 交换次数
        self._te_step_state = {}    # 原子化 TE encode 状态机
        # 显式权重服务（stdlib 导入安全，启动不读权重）
        self.weights = WeightService(self)
        self.te = self.weights.backends["te"]
        self.te_residence = TeResidency(self)

    @property
    def ready(self):
        """Legacy compute-ready field; control APIs must instead use env_ready."""
        return self.compute_ready

    @ready.setter
    def ready(self, value):
        self.compute_ready = bool(value)

    def ensure_env(self):
        """Initialize dependencies without constructing any weighted model.

        Quant kernels, decode patches and attention optimization are deferred
        to explicit operations. Failure remains visible to diagnostic APIs.
        """
        with self._env_lock:
            if self.env_ready:
                return
            self.set_stage("初始化 ComfyUI 环境（不加载权重）")
            try:
                import torch
                import comfy.options
                comfy.options.enable_args_parsing(True)
                import comfy.cli_args
                comfy.cli_args.args.gpu_only = True
                import comfy.sd
                import comfy.model_management as mm
                import comfy.ldm.modules.attention as attention
                import comfy.ldm.minimax.model as minimax_model
                import comfy_kitchen
                ffmpeg = find_ffmpeg()
                if not ffmpeg:
                    raise RuntimeError("ffmpeg unavailable; configure it before inference")
                self.torch = torch
                self.mm = mm
                self.attention = attention
                self.minimax_model = minimax_model
                self.comfy_kitchen = comfy_kitchen
                self.device = torch.device("cuda:0")
                self.ffmpeg = ffmpeg
                self.error = None
                self.env_ready = True
                self.set_stage("环境就绪（无权重）")
                # 只登记 recipe/catalog（stat 文件），不读权重、不构造模型
                try:
                    self.weights.prepare()
                except Exception as exc:
                    print("[weights] prepare 失败: %s: %s"
                          % (type(exc).__name__, exc), flush=True)
            except Exception as exc:
                self.env_ready = False
                self.error = "%s: %s" % (type(exc).__name__, exc)
                self.set_stage("环境初始化失败")
                traceback.print_exc()
                raise

    def weights_snapshot(self):
        """A cheap metadata snapshot; startup never opens checkpoint files."""
        if self.weights is None:
            return {"ok": True, "models": {}}
        return self.weights.snapshot()

    def capabilities(self):
        models = self.weights_snapshot()["models"]
        needs = {"encode": ("te",), "te_probe": ("te",), "sample": ("dit",),
                 "dit_probe": ("dit",),
                 "decode": ("vae.video", "vae.audio"),
                 "video_from_cond": ("dit", "vae.video", "vae.audio")}
        result = {}
        for name, required in needs.items():
            missing = []
            for model in required:
                state = models[model]["state"]
                ok = (state == "vram") if name in ("encode", "te_probe",
                                                  "dit_probe") else (
                    state in ("ram", "vram")) if name == "decode" else (
                    state == "vram" if model == "dit" else state in ("ram", "vram"))
                if not ok:
                    missing.append(model)
            result[name] = {"ready": self.env_ready and self.device_healthy and not missing,
                            "missing": missing}
        return result

    def job_admission(self, job):
        """Cheap locked admission hook.

        sample 只需要 DiT 在显存；VAE 允许停在 RAM（decode 前 pipeline 会上卡，
        首版仍是其原有的 `.to(device)`）。缺失的模型仍拒绝，不自动搬运。
        具体 unit 依赖在 P3 推理原语里继续细化。
        """
        from dit_svc.operations import OperationError
        capability = "decode" if job.kind == "vae_probe" else "video_from_cond"
        models = self.weights_snapshot()["models"]
        issues = []
        if job.kind == "vae_probe":
            for model in ("vae.video",):
                if models[model]["state"] not in ("ram", "vram"):
                    issues.append((model, models[model]["state"]))
        else:
            if models["dit"]["state"] != "vram":
                issues.append(("dit", models["dit"]["state"]))
            for model in ("vae.video", "vae.audio"):
                if models[model]["state"] not in ("ram", "vram"):
                    issues.append((model, models[model]["state"]))
        if issues:
            raise OperationError(
                "MISSING_UNITS", "任务所需权重尚未就绪", 409, False,
                {"capability": capability,
                 "missing": [model for model, _ in issues],
                 "states": {model: state for model, state in issues}})

    def require_job_execution(self, job):
        self.job_admission(job)

    def operation_admission(self, kind, payload):
        from dit_svc.operations import OperationError
        if not self.env_ready:
            raise OperationError("ENV_NOT_READY", self.error or "环境尚未就绪", 503, True)
        if not self.device_healthy:
            raise OperationError("DEVICE_UNHEALTHY", "CUDA 设备不可用，需检查或重启", 503, False)
        if kind in ("infer.encode", "infer.te_probe"):
            # 只查状态，不自动上卡；缺资源时给有序建议。
            self.weights.preflight_encode(None)
        if kind == "infer.dit_probe":
            # 采样前向每步都过 token_refiner；partial 布局直接拒绝。
            dit = self.weights_snapshot()["models"].get("dit") or {}
            if dit.get("state") != "vram":
                raise OperationError(
                    "MISSING_UNITS", "dit_probe 需要 dit 全部 unit 在显存",
                    409, False,
                    {"dit": dit.get("state"),
                     "hint": "POST /api/weights/dit/ram2vram {\"mode\":\"all\"}"})
        if kind == "attn.install":
            # 便宜检查（持锁）：DiT 采样外壳必须就绪；安装本体在 worker 里执行。
            if self.unet is None or self.model_shifted is None:
                raise OperationError(
                    "ATTN_NOT_READY",
                    "DiT 采样外壳未就绪；先完成 dit 全量 disk2ram（补齐全部 unit）",
                    409, False,
                    {"hint": "POST /api/weights/dit/disk2ram {\"mode\":\"all\"}"})

    def run_weights_action(self, model_id, action, request, operation=None):
        """weights.<action> 的唯一执行入口（单 worker 调用）。"""
        return self.weights.run_action(model_id, action, request, operation)

    def run_infer_encode(self, request, operation=None):
        """infer.encode 的唯一执行入口（单 worker 调用）。"""
        return self.weights.encode(request, operation)

    def run_infer_te_probe(self, request, operation=None):
        return self.weights.probe(request, operation)

    def run_infer_dit_probe(self, request, operation=None):
        """infer.dit_probe 的唯一执行入口（单 worker 调用）。"""
        from dit_svc import dit_probe
        return dit_probe.run_probe(self, request, operation)

    def run_attn_install(self, request, operation=None):
        """attn.install 的唯一执行入口（单 worker 调用）。"""
        from dit_svc import attn_service
        return attn_service.install(self, request, operation)

    def operation_synchronize(self, operation):
        if self.torch is not None and self.torch.cuda.is_initialized():
            try:
                self.torch.cuda.synchronize()
            except Exception:
                self.device_healthy = False
                raise

    def job_synchronize(self, job):
        self.operation_synchronize(job)

    # ---------- 阶段上报 ----------
    def set_stage(self, name):
        self.stage = name
        self.stage_t0 = time.perf_counter()
        print("[runtime] stage -> %s" % name, flush=True)

    def stage_elapsed(self):
        return time.perf_counter() - self.stage_t0

    def _vram_ledger(self, label, torch):
        """显存账本：分段打印 alloc/reserved/free + 本段增量（上次调用以来）。"""
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free, total = (x / 1e9 for x in torch.cuda.mem_get_info())
        prev = getattr(self, "_ledger_prev", 0.0)
        print("[vram] %-14s alloc=%.2f (+%.2f) reserved=%.2f free=%.2f/%.2f GB"
              % (label, alloc, alloc - prev, reserved, free, total), flush=True)
        self._ledger_prev = alloc

    def _dump_module_dtypes(self, unet):
        """打印 DiT 每个带权模块的实际 dtype/量化标记 + 汇总。

        用途：定位"哪些层没被 w4a8 接管、于是按 bf16 跑"——bt16 在 sm75 上没有
        张量核，PyTorch 会退回 MAGMA 的 fp32 仿算（profile 里那只 236ms/kernel）。
        """
        from collections import Counter
        try:
            dm = unet.model.diffusion_model
        except AttributeError:
            print("[dtype] 拿不到 diffusion_model，跳过", flush=True)
            return
        summary = Counter()
        rows = []
        for name, mod in dm.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or getattr(w, "ndim", 0) < 2:
                continue
            qf = getattr(mod, "quant_format", None) or "-"
            summary[(str(getattr(w, "dtype", "?")), str(qf))] += 1
            rows.append("  %-56s %-16s %-20s quant=%s"
                        % (name, str(getattr(w, "dtype", "?")), tuple(getattr(w, "shape", ())), qf))
        print("[dtype] DiT 带权模块 %d 个，按 (dtype, quant) 汇总：" % len(rows), flush=True)
        for (dt, qf), n in sorted(summary.items(), key=lambda kv: -kv[1]):
            print("[dtype]   %-16s quant=%-12s x%d" % (dt, qf, n), flush=True)
        for line in rows[:240]:
            print("[dtype]" + line, flush=True)
        if len(rows) > 240:
            print("[dtype]   …（只打前 240 行，共 %d）" % len(rows), flush=True)

    def _dit_sub_ledger(self, unet, torch):
        """DiT 内部分项（静态参数量，按顶层子模块分组；与动态 alloc 对照）。"""
        try:
            dm = unet.model.diffusion_model
        except AttributeError:
            return
        rows = []
        total = 0
        for name, mod in dm.named_children():
            n = sum(p.numel() * p.element_size() for p in mod.parameters())
            rows.append((name, n / 1e9))
            total += n
        rows.sort(key=lambda r: -r[1])
        print("[vram] DiT分项(静态%.2fGB): %s" % (
            total / 1e9, " ".join("%s=%.2f" % r for r in rows[:12])), flush=True)

    # ---------- 后台加载 ----------
    def load(self):
        try:
            self._load()
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            self.stage = "加载失败"
            traceback.print_exc()
            raise

    def _load(self):
        rt = self.rt
        args = self.args
        import torch

        # ---- comfy 环境（argv 已由入口清干净） ----
        self.set_stage("初始化 ComfyUI 运行时")
        import comfy.options
        comfy.options.enable_args_parsing(True)
        import comfy.cli_args  # noqa: F401
        if not comfy.cli_args.args.gpu_only:
            comfy.cli_args.args.gpu_only = True
        import comfy.sd
        import comfy.model_management as mm
        import comfy.ldm.modules.attention as attention
        import comfy.ldm.minimax.model as minimax_model
        import comfy_kitchen
        self.torch = torch
        self.mm = mm
        self.attention = attention

        # ---- w4a8 内核 ----
        self.set_stage("安装 w4a8 内核 (%s)" % args.w4a8_backend)
        if args.w4a8_backend == "turingutils":
            rt.install_wjie98_w4a8()
        else:
            rt.unlock_cuda_w4a8_on_sm75()

        # ---- attention 后端（默认 int8；sage/w8a8 为调试档；veda=dev 实验） ----
        # Veda 的 configure() 需要 model 对象（bundle 维度校验 + layout 安装），
        # 故 attention 安装推迟到 DiT 加载后（原先在这里的调用已下移）。
        self.attn_override = None

        # ---- 解码功率节流（PSU di/dt 保护，沿用 runner） ----
        rt.install_decode_power_limiter(tile_batch=1)

        device = torch.device("cuda:0")
        self.device = device

        # ---- 1. DiT 流式加载常驻 ----
        self.set_stage("加载 DiT 权重（流式，约 164s）")
        t0 = time.perf_counter()
        sd, md = rt.stream_load_safetensors_to_dict(args.unet, device="cuda")
        model_options = {"load_device": device, "offload_device": device}
        unet = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=md)
        unet.model.manual_cast_dtype = torch.bfloat16
        unet.offload_device = device
        del sd
        gc.collect()
        torch.cuda.empty_cache()
        # 带 quant_config 的模型恒走 manual-cast：set_inference_dtype 一旦拿到非 None 的
        # manual_cast_dtype，就把模型 dtype 钉成 fp32，于是 model_options["dtype"] 对常驻
        # 精度无效（实测传 bf16 后 15.82GB 纹丝不动）。改为加载后就地降级——checkpoint 里
        # 这些张量本来就是 bf16，334 个张量因此从 3.38GB 回到 1.69GB。量化层是
        # QuantizedTensor 包装类（dtype 也报 fp32），必须跳过：认它的 _qdata 属性，
        # 别依赖 comfy_kitchen 顶层导出（那里没有 QuantizedTensor，会 AttributeError）。
        _squeezed = 0
        for _mod in unet.model.diffusion_model.modules():
            for _name, _p in list(_mod.named_parameters(recurse=False)):
                if _p.dtype != torch.float32 or hasattr(_p, "_qdata"):
                    continue
                _mod._parameters[_name] = torch.nn.Parameter(
                    _p.data.to(torch.bfloat16), requires_grad=False)
                _squeezed += 1
        gc.collect()
        torch.cuda.empty_cache()
        print("[load] non-quant params -> bf16: %d tensors" % _squeezed, flush=True)
        print("[load] DiT loaded in %.1fs, vram: %.2f GB"
              % (time.perf_counter() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)
        self._vram_ledger("DiT权重", torch)
        self._dit_sub_ledger(unet, torch)
        if getattr(args, "dump_module_dtypes", False):
            self._dump_module_dtypes(unet)
        result = _install_attention(
            rt, attention, minimax_model, comfy_kitchen, args.attn,
            args=args, unet=unet)
        if args.attn == "veda":
            # 路线 B：veda_star7.install() 已把 override + ON_PREPARE_STATE /
            # ON_CLEANUP 回调就地装在 unet.model_options 上（不走 clone，
            # 保持单一对象给常驻引用 / _unhook_dit / _AttnWindowProfiler 用）。
            # result 是 veda handle（VedaPatch，不是函数）；None = 已回退 int8。
            self.attn_override = None
            self.veda_patch = result
            print("[attention] veda handle=%s"
                  % ("patch" if result is not None else "None(已回退int8)"), flush=True)
        else:
            self.attn_override = result
            if self.attn_override is not None:
                unet.model_options.setdefault("transformer_options", {})[
                    "optimized_attention_override"] = self.attn_override
                print("[attention] %s override installed on unet.model_options" % args.attn, flush=True)

        # ---- 2. 两个 VAE（权重留 CPU，按需上卡，之后常驻） ----
        from collections import Counter
        fmt_counter = Counter(
            getattr(m, "quant_format", None) or "none"
            for m in unet.model.diffusion_model.modules()
            if getattr(m, "quant_format", None) is not None or getattr(m, "weight", None) is not None
        )
        fmt_counter.pop("none", None)
        print("[model] quant_format linears:", dict(fmt_counter), flush=True)
        print("[model] DiT class:", type(unet.model.diffusion_model).__name__, flush=True)

        # ---- 3. mlp 行分块 ----
        if args.mlp_chunk > 0:
            self.set_stage("安装 MLP 行分块")
            import h3_mlp_chunk  # runner 目录已在 sys.path
            h3_mlp_chunk.MLP_CHUNK = args.mlp_chunk
            n = h3_mlp_chunk.apply(unet.model.diffusion_model)
            print("[mlp-chunk] patched %d MLP modules (chunk=%d rows)"
                  % (n, h3_mlp_chunk.MLP_CHUNK), flush=True)

        # ComfyAttention 统计（checkpoint 内嵌 config 会被 override 压过，仅作信息）
        attn_modules = [m for m in unet.model.diffusion_model.modules()
                        if isinstance(m, attention.ComfyAttention)]
        attn_bound = [m for m in attn_modules if getattr(m, "function", None) is not None]
        print("[model] ComfyAttention modules: %d, bound: %d" % (len(attn_modules), len(attn_bound)),
              flush=True)

        self.set_stage("加载视频 VAE（约 34s）")
        t0 = time.perf_counter()
        vsd, vmd = rt.stream_load_safetensors_to_dict(args.video_vae, device="cuda")
        vvae = _build_video_vae(comfy, vsd, vmd, device, torch)
        vvae.first_stage_model.requires_grad_(False)
        del vsd
        gc.collect()
        torch.cuda.empty_cache()
        print("[load] video VAE in %.1fs (%s)" % (time.perf_counter() - t0, args.video_vae), flush=True)
        self._vram_ledger("视频VAE", torch)

        self.set_stage("加载音频 VAE（约 8s）")
        t0 = time.perf_counter()
        asd, amd = rt.stream_load_safetensors_to_dict(args.audio_vae, device="cuda")
        avae = comfy.sd.VAE(sd=asd, metadata=amd, device=device)
        avae.first_stage_model.requires_grad_(False)
        del asd
        gc.collect()
        torch.cuda.empty_cache()
        print("[load] audio VAE in %.1fs" % (time.perf_counter() - t0), flush=True)
        self._vram_ledger("音频VAE", torch)

        # VAE 常驻（offload_device=cpu 只作兜底，不在任务间搬运）
        vvae.patcher.offload_device = torch.device("cpu")
        avae.patcher.offload_device = torch.device("cpu")

        # ---- 4. sigma shift（采样 guider 用） ----
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        model_shifted = rt.unpack(MiniMaxH3SigmaShift.execute(unet, 12.0, 3.0))[0]
        model_shifted.model.manual_cast_dtype = torch.bfloat16
        model_shifted.offload_device = device
        if args.attn == "veda":
            # veda：ModelPatcher.clone() 会深拷 model_options 并复制
            # callbacks/wrappers（comfy/model_patcher.py:431-505），所以
            # sigma-shift 的 clone 自带 override 与 ON_PREPARE_STATE 回调；
            # 这里再显式 install 一次只是双保险（install() 幂等：
            # current in self.installed 即返回）。
            if self.veda_patch is not None:
                self.veda_patch.install(
                    model_shifted.model_options.setdefault("transformer_options", {}))
                print("[attention] veda override installed on model_shifted.model_options",
                      flush=True)
        elif self.attn_override is not None:
            model_shifted.model_options.setdefault("transformer_options", {})[
                "optimized_attention_override"] = self.attn_override
            print("[attention] %s override installed on model_shifted.model_options" % args.attn,
                  flush=True)

        # ---- 5. ffmpeg ----
        self.ffmpeg = find_ffmpeg()
        if not self.ffmpeg:
            raise RuntimeError("找不到 ffmpeg（imageio_ffmpeg 与 PATH 均无）")
        print("[ffmpeg] %s" % self.ffmpeg, flush=True)

        # 保存常驻引用。注意：token_refiner / condition_proj **刻意不卸载**，
        # preprocess_text_embeds 保持原实现（逐任务 refine）。
        self.unet = unet
        self.model_shifted = model_shifted
        self.vvae = vvae
        self.avae = avae
        self.diff_model = unet.model.diffusion_model

        # ---- 6. 可选：DiT 分块卸载到 RAM，给文本编码器腾地方 ----
        n_off = int(getattr(args, "offload_dit_blocks", 0) or 0)
        if n_off > 0:
            self._offload_dit_blocks(n_off, torch)

        # ---- 7. 可选：文本编码器整份常驻 RAM（cpu），不占显卡 ----
        if getattr(args, "te_gguf", None) or getattr(args, "te_4b", None):
            self._load_text_encoder_ram(torch)

        gc.collect()
        torch.cuda.empty_cache()
        rt.log_memory("Runtime ready")
        self._vram_ledger("READY常驻", torch)
        self.ready = True
        self.stage = "就绪"
        print("[runtime] READY — free VRAM %.2f GB"
              % (torch.cuda.mem_get_info()[0] / 1e9), flush=True)


    # ---------- 可选：DiT 分块卸载 / 文本编码器常驻 RAM ----------
    def _offload_dit_blocks(self, n, torch):
        """把 blocks.0..n-1 整块搬到 CPU（RAM），腾出同等显存。

        实测结论：QuantizedTensor.to("cpu") 保留 _qdata 的 int8 存储，字节不变
        （不会解包成 bf16）；显存 1:1 换成主机内存。中间块约 0.2GB，首两块约 0.4GB。
        注意 ComfyUI 自带的 ModelPatcher.partially_unload 在"刚加载、无 patched
        weights"的状态下是空转（实测 freed=0），所以这里直接搬模块。
        """
        import gc
        blocks = self.diff_model.blocks
        n = min(n, len(blocks))
        self.set_stage("卸载 DiT blocks 到 RAM (%d)" % n)
        t0 = time.perf_counter()
        before = torch.cuda.memory_allocated()
        for i in range(n):
            blocks[i].to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        freed = (before - torch.cuda.memory_allocated()) / 1e9
        self.offloaded_blocks = list(range(n))
        print("[offload] %d blocks -> cpu in %.1fs, freed %.2f GB (%.2f GiB)"
              % (n, time.perf_counter() - t0, freed, freed / 1.073741824), flush=True)
        self._vram_ledger("DiT卸载%d块" % n, torch)

    def _load_text_encoder_ram(self, torch):
        """文本编码器（Qwen3-VL + mmproj）整份加载到 CPU 常驻，不占显卡。

        走 ComfyUI-GGUF：GGMLOps 把 GGUF 反量化成 torch 张量，结构由
        comfy/text_encoders/minimax.py（CLIPType.MINIMAX）定义。
        本进程强制 gpu_only=True，text_encoder_offload_device() 会返回 cuda，
        所以必须显式 initial_device=cpu，否则就上卡了。

        失败不阻塞服务：记录原因，self.text_encoder 保持 None，DiT 照常待命。
        """
        import gc
        import importlib
        import types
        args = self.args
        if getattr(args, "te_4b", None):
            return self._load_te_4b_proj(torch)
        path = args.te_gguf
        self.set_stage("文本编码器加载到 RAM")
        t0 = time.perf_counter()
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            import comfy
            import comfy.sd
            # 注意：本 checkout 的 comfy 是 namespace package，__file__ 为 None，
            # 只有 __path__ 有效（[<ComfyUI>/comfy]）。依次尝试两条来源 + sys.path 兜底。
            cands = []
            cpath = getattr(comfy, "__path__", None)
            if cpath:
                cands.append(os.path.dirname(os.path.abspath(list(cpath)[0])))
            cfile = getattr(comfy, "__file__", None)
            if cfile:
                cands.append(os.path.dirname(os.path.dirname(os.path.abspath(cfile))))
            cands.extend(os.path.abspath(p) for p in sys.path if p)
            pkg_dir = None
            for c in cands:
                probe = os.path.join(c, "custom_nodes", "ComfyUI-GGUF")
                if os.path.isdir(probe):
                    pkg_dir = probe
                    print("[te] ComfyUI 根目录: %s" % c, flush=True)
                    break
            if pkg_dir is None:
                raise FileNotFoundError("找不到 ComfyUI-GGUF（试过 %d 个候选根）" % len(cands))
            if "comfyui_gguf" not in sys.modules:
                pkg = types.ModuleType("comfyui_gguf")
                pkg.__path__ = [pkg_dir]
                sys.modules["comfyui_gguf"] = pkg
            gguf_loader = importlib.import_module("comfyui_gguf.loader")
            gguf_ops = importlib.import_module("comfyui_gguf.ops")

            print("[te] 读 GGUF: %s" % os.path.basename(path), flush=True)
            sd = gguf_loader.gguf_clip_loader(path)
            # 视觉塔可能以 visual.*（realrebelai 的 Q2_K 导出）或 model.visual.*
            # （本插件的 mmproj 分离加载）两种前缀出现，两种都算。
            has_vis = any(k.startswith(("visual.", "model.visual.")) for k in sd)
            if not has_vis:
                mm = getattr(args, "te_mmproj", None)
                if mm and os.path.isfile(mm):
                    print("[te] 主权重未带视觉塔，单独加载 mmproj: %s"
                          % os.path.basename(mm), flush=True)
                    sd.update(gguf_loader.gguf_qwen3vl_mmproj_loader(mm))
                    has_vis = any(k.startswith(("visual.", "model.visual.")) for k in sd)
            print("[te] GGUF 张量 %d 个，视觉塔=%s" % (len(sd), has_vis), flush=True)

            # detect_te_model 先看 "model.visual.deepstack_merger_list..." —— 命中就判 4B/8B；
            # 要判成 QWEN3VL_32B（H3 视频版 cond，hidden 5120 / 50 层）必须同时满足
            #   "visual.deepstack_merger_list.0.norm.weight"（无 model. 前缀）
            #   "model.layers.49.self_attn.q_proj.weight"
            # 而 GGUF 插件给视觉塔统一加了 model. 前缀，于是误落 8B 分支（按 4096 建模型，
            # 与 checkpoint 的 5120 尺寸不符）。这里只剥视觉键的前缀，tokenizer/层键不动。
            n_ren = 0
            for k in [k for k in list(sd.keys()) if k.startswith("model.visual.")]:
                sd[k[len("model."):]] = sd.pop(k)
                n_ren += 1
            if n_ren:
                print("[te] 视觉键 model.visual.* -> visual.*: %d 个" % n_ren, flush=True)
            try:
                print("[te] detect_te_model -> %s" % comfy.sd.detect_te_model(sd), flush=True)
            except Exception as _e:
                print("[te] detect 失败: %s" % _e, flush=True)

            clip = comfy.sd.load_text_encoder_state_dicts(
                clip_type=comfy.sd.CLIPType.MINIMAX,
                state_dicts=[sd],
                model_options={
                    "custom_operations": gguf_ops.GGMLOps,
                    # 待命在内存、要算上显卡：
                    #   initial_device —— 构造时权重落在 cpu（不占显存）
                    #   load_device    —— encode 时 CLIP.load_model() 把它搬到这里；
                    #                     显式给 cuda 才会上卡算。但要留神：load_models_gpu
                    #                     会为腾地方卸掉其它模型（DiT），实测时确认它动了什么
                    #   offload_device —— 用完退回 cpu
                    "initial_device": torch.device("cpu"),
                    "load_device": torch.device("cuda:0"),
                    "offload_device": torch.device("cpu"),
                },
                embedding_directory=[],
            )
            del sd
            gc.collect()
            self.text_encoder = clip
            self.te_load_s = time.perf_counter() - t0
            devs = {}
            try:
                for p in clip.patcher.model.parameters():
                    d = str(p.device)
                    devs[d] = devs.get(d, 0) + 1
            except Exception:
                pass
            print("[te] 文本编码器就绪 %.1fs | 参数分布 %s" % (self.te_load_s, devs), flush=True)
        except Exception as e:
            import traceback
            self.te_load_s = time.perf_counter() - t0
            print("[te] 加载失败（DiT 照常待命）: %s: %s" % (type(e).__name__, e), flush=True)
            traceback.print_exc()


    # ---------- 4B 编码器 + ClipProj（--te-4b / --te-proj） ----------
    def _load_te_4b_proj(self, torch):
        """Qwen3-VL-4B（int8_convrot）+ ClipProj 投影，整份常驻 CPU。

        4B 的 hidden 是 2560 维，H3 的 condition_proj 只收 5120，
        中间靠 ClipProj 的投影补上（ridge + 残差，作者自报 cos≈0.79）。
        投影按 tap=24 取第 24 层 hidden，且不做 final norm —— 与 H3 契约一致。

        加载完一切都在 CPU：encode 时才上卡，算完立刻退回（见 encode_condition）。
        """
        import gc
        args = self.args
        te_path = args.te_4b
        proj_name = args.te_proj
        self.set_stage("4B 编码器 + 投影加载到 RAM")
        t0 = time.perf_counter()
        try:
            if not os.path.isfile(te_path):
                raise FileNotFoundError(te_path)
            if not proj_name:
                raise ValueError("--te-4b 需要配套的 --te-proj")
            import comfy
            import comfy.sd

            # 定位 ComfyUI 根（comfy 是 namespace package，__file__ 可能是 None）
            cands = []
            cpath = getattr(comfy, "__path__", None)
            if cpath:
                cands.append(os.path.dirname(os.path.abspath(list(cpath)[0])))
            cfile = getattr(comfy, "__file__", None)
            if cfile:
                cands.append(os.path.dirname(os.path.dirname(os.path.abspath(cfile))))
            cands.extend(os.path.abspath(p) for p in sys.path if p)
            comfy_root = None
            for c in cands:
                if os.path.isdir(os.path.join(c, "custom_nodes")):
                    comfy_root = c
                    break
            if comfy_root is None:
                raise FileNotFoundError("找不到 ComfyUI 根目录（custom_nodes 所在处）")
            cn = os.path.join(comfy_root, "custom_nodes")
            if cn not in sys.path:
                sys.path.insert(0, cn)
            if not os.path.isdir(os.path.join(cn, "clipproj")):
                raise FileNotFoundError("custom_nodes/clipproj 缺失（ClipProj 节点未安装）")
            print("[te4b] ComfyUI 根: %s" % comfy_root, flush=True)

            print("[te4b] 读编码器: %s" % os.path.basename(te_path), flush=True)
            clip = comfy.sd.load_clip(
                ckpt_paths=[te_path],
                embedding_directory=[],
                clip_type=comfy.sd.CLIPType.MINIMAX,
                model_options={
                    # 构造就落 CPU，不占显存；encode 时按 load_device 上卡
                    "initial_device": torch.device("cpu"),
                    "load_device": torch.device("cuda:0"),
                    "offload_device": torch.device("cpu"),
                },
            )
            nparam = sum(p.numel() for p in clip.patcher.model.parameters())
            print("[te4b] 编码器就绪 %.1fs | %.2f B 参数 | 主设备 %s"
                  % (time.perf_counter() - t0, nparam / 1e9,
                     clip.patcher.model.device), flush=True)

            from clipproj.clipproj_projection import register_folder
            register_folder()          # 把 models/clip_projections 登记给 folder_paths
            from clipproj.clipproj_nodes import ProjectedCLIP
            pc = ProjectedCLIP(clip, proj_name)
            proj = pc._proj
            print("[te4b] 投影 %s | d_in=%d d_out=%d tap=%d | mlp=%s | cos_test=%s"
                  % (proj_name, proj["mean_in"].shape[0], proj["mean_out"].shape[0],
                     proj.get("tap"), "mlp" in str(proj.get("mlp", "")).lower() or ("mlp.0.weight" in proj),
                     proj.get("cos_test")), flush=True)

            self.te_proj = pc
            self.text_encoder = pc      # 兼容既有引用
            self.te_load_s = time.perf_counter() - t0
            devs = {}
            try:
                for p in pc.patcher.model.parameters():
                    d = str(p.device)
                    devs[d] = devs.get(d, 0) + 1
            except Exception:
                pass
            print("[te4b] 4B+投影 就绪 %.1fs | 参数分布 %s（全部待命在 RAM）"
                  % (self.te_load_s, devs), flush=True)
        except Exception as e:
            import traceback
            self.te_load_s = time.perf_counter() - t0
            print("[te4b] 加载失败（DiT 照常待命）: %s: %s" % (type(e).__name__, e), flush=True)
            traceback.print_exc()

    def _vae_to(self, device, torch, label=""):
        """把视频 VAE 整份搬到 device；返回释放的显存（GB，负数为占用）。"""
        if self.vvae is None:
            return 0.0
        before = torch.cuda.memory_allocated()
        try:
            self.vvae.patcher.model.to(device)
        except Exception as e:
            print("[swap] VAE -> %s 失败: %s: %s" % (device, type(e).__name__, e), flush=True)
            return 0.0
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        freed = (before - after) / 1e9
        print("[swap] VAE -> %-5s alloc %.2f -> %.2f GB (%+.2f) %s"
              % (str(device).replace("cuda:0", "gpu"), before / 1e9, after / 1e9, -freed, label),
              flush=True)
        return freed

    # ============================================================
    # 文本编码器（TE）原子化操作：VAE 让位 / TE 上卡 / TE 退回 / VAE 归位
    # 这些操作可逐个步骤通过 HTTP API 触发，便于远程高密度 GPU 下调试。
    # ============================================================

    def _te_model(self):
        """当前已加载的 TE 对象；由显式权重管理器持有。"""
        return self.te.pc

    def te_vae_swap_out(self):
        """VAE 让位到 CPU，给 TE 腾显存。返回 {"freed_gb", "elapsed_s"}。"""
        import torch
        self.set_stage("TE: VAE 让位")
        t0 = time.perf_counter()
        freed = self._vae_to(torch.device("cpu"), torch, "为 TE 让位")
        elapsed = time.perf_counter() - t0
        return {"freed_gb": round(freed, 3), "elapsed_s": round(elapsed, 3)}

    def te_vae_swap_back(self):
        """VAE 归位到 GPU。返回 {"elapsed_s"}。"""
        import torch
        self.set_stage("TE: VAE 归位")
        t0 = time.perf_counter()
        self._vae_to(torch.device("cuda:0"), torch, "TE 用完，归位")
        elapsed = time.perf_counter() - t0
        self.te_swaps += 1
        return {"elapsed_s": round(elapsed, 3)}

    def te_load_to_vram(self):
        """显式 ram2vram：只在本服务动作授权下搬权重，不推理。"""
        import torch
        if self._te_model() is None:
            raise RuntimeError("TE 未加载：先用 POST /api/weights/te/disk2ram")
        result = self.te.run_action("ram2vram", _all_request())
        return {
            "elapsed_s": result.get("elapsed_s", 0.0),
            "device": ((result.get("audit") or {}).get("units") or {}).get("te.model", {}).get("device"),
            "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "audit": result.get("audit"),
        }

    def te_offload_to_ram(self):
        """显式 vram2ram：搬回 CPU 并清掉 ClipProj 的 GPU 缓存副本。"""
        if self._te_model() is None:
            raise RuntimeError("TE 未加载")
        result = self.te.run_action("vram2ram", _all_request())
        return {
            "elapsed_s": result.get("elapsed_s", 0.0),
            "device": ((result.get("audit") or {}).get("units") or {}).get("te.model", {}).get("device"),
            "audit": result.get("audit"),
        }

    def _te_preprocess_refs(self, image_bytes_list, mode, width, height, ref_image_size):
        """Legacy input decoding delegates to the same CPU preprocessing."""
        import base64
        images = [base64.b64encode(value).decode("ascii")
                  for value in image_bytes_list or [] if value]
        return self.te._preprocess(images, mode, width, height, ref_image_size)

    def te_encode_tokens(self, prompt, ref_items, mode="ref2va"):
        """Legacy 调试入口也复用受保护的 TE 推理，不再维护第二份实现。"""
        return self.te._encode_tokens(prompt, ref_items, mode)

    def te_step(self, action, payload=None):
        """原子化 TE 步骤状态机。返回 {"ok", "action", "result", "state"}。

        action 取值：
          vae_swap_out : VAE -> CPU
          te_load_vram : TE RAM -> VRAM（不推理）
          te_offload_ram : TE VRAM -> RAM
          preprocess   : 参考图预处理（CPU）
          te_encode    : TE 推理（需先 te_load_vram + preprocess）
          vae_swap_back: VAE -> GPU
          reset        : 重置状态机并恢复 VAE/TE 位置
        """
        import torch
        payload = payload or {}
        state = self._te_step_state

        def _set(step, **kwargs):
            state["step"] = step
            state["last_action"] = action
            state["last_at"] = time.time()
            state.update(kwargs)

        # reset 总是可用
        if action == "reset":
            try:
                self._vae_to(torch.device("cuda:0"), torch, "状态机 reset：VAE 归位")
            except Exception as e:
                print("[te] reset VAE 归位失败: %s" % e, flush=True)
            try:
                pc = self._te_model()
                if pc is not None:
                    self.te_offload_to_ram()
            except Exception as e:
                print("[te] reset TE 退回失败: %s" % e, flush=True)
            self._te_step_state = {}
            return {"ok": True, "action": action, "state": {}}

        if action == "vae_swap_out":
            res = self.te_vae_swap_out()
            _set("vae_swapped_out", vae_freed_gb=res["freed_gb"], elapsed=res["elapsed_s"])
            return {"ok": True, "action": action, "result": res, "state": state}

        if action == "te_load_vram":
            res = self.te_load_to_vram()
            _set("te_on_vram", device=res["device"], alloc_gb=res["alloc_gb"], elapsed=res["elapsed_s"])
            return {"ok": True, "action": action, "result": res, "state": state}

        if action == "te_offload_ram":
            res = self.te_offload_to_ram()
            _set("te_on_ram", device=res["device"], elapsed=res["elapsed_s"])
            return {"ok": True, "action": action, "result": res, "state": state}

        if action == "preprocess":
            mode = payload.get("mode", "ref2va")
            width = int(payload.get("width", 640))
            height = int(payload.get("height", 960))
            ref_image_size = payload.get("ref_image_size", "match")
            image_bytes = payload.get("image_bytes_list", [])
            imgs, ref_items, qwen_sizes = self._te_preprocess_refs(
                image_bytes, mode, width, height, ref_image_size)
            state["preprocessed"] = {
                "imgs": imgs,
                "ref_items": ref_items,
                "qwen_sizes": qwen_sizes,
                "mode": mode,
                "width": width,
                "height": height,
                "ref_image_size": ref_image_size,
            }
            _set("preprocessed", n_images=len(imgs), qwen_hw=qwen_sizes)
            return {"ok": True, "action": action, "result": {
                "n_images": len(imgs), "qwen_hw": qwen_sizes}, "state": state}

        if action == "te_encode":
            if "preprocessed" not in state:
                # 允许在单步里传图并预处理
                mode = payload.get("mode", "ref2va")
                width = int(payload.get("width", 640))
                height = int(payload.get("height", 960))
                ref_image_size = payload.get("ref_image_size", "match")
                image_bytes = payload.get("image_bytes_list", [])
                imgs, ref_items, qwen_sizes = self._te_preprocess_refs(
                    image_bytes, mode, width, height, ref_image_size)
            else:
                p = state["preprocessed"]
                ref_items = p["ref_items"]
                qwen_sizes = p["qwen_sizes"]
                mode = p["mode"]

            prompt = payload.get("prompt", "")
            out, enc_stats = self.te_encode_tokens(prompt, ref_items, mode=mode)
            # 兼容 te_encode_tokens 的命名
            result = {
                "cond_shape": enc_stats["cond_shape"],
                "n_tokens": enc_stats["n_tokens"],
                "n_images": len(qwen_sizes),
                "mode": mode,
                "qwen_hw": qwen_sizes,
                "tokenize_s": enc_stats.get("tokenize_s"),
                "encode_s": enc_stats.get("encode_s"),
            }
            _set("encoded", **result)
            out_dict = {
                "cond": out["cond"],
                "minimax_token_tags": out.get("minimax_token_tags"),
            }
            state["last_cond"] = out_dict
            return {"ok": True, "action": action, "result": result, "state": state, "cond": out_dict}

        if action == "vae_swap_back":
            res = self.te_vae_swap_back()
            _set("vae_swapped_back", elapsed=res["elapsed_s"], swaps=self.te_swaps)
            return {"ok": True, "action": action, "result": res, "state": state}

        raise ValueError("未知 action: %s" % action)

    def run_te_legacy(self, payload, operation=None):
        """Legacy TE 调试也在唯一 worker 中运行；回执不包含 Tensor。"""
        import base64
        action = payload.get("action")
        if not isinstance(action, str):
            raise OperationError("INVALID_REQUEST", "action 须为字符串", 400, False)
        spec = dict(payload)
        spec["image_bytes_list"] = [base64.b64decode(value.split(",", 1)[-1])
                                    for value in payload.get("image_b64s", [])]
        result = self.te_step(action, spec)
        cond = result.pop("cond", None)
        result["state"] = self._te_public_state()
        if cond is not None and payload.get("save_cond"):
            request = {"mode": result["result"]["mode"],
                       "width": int(payload.get("width", 640)),
                       "height": int(payload.get("height", 960)),
                       "ref_image_size": payload.get("ref_image_size", "match"),
                       "image_b64s": payload.get("image_b64s", [])}
            artifact = self.te._save_cond(cond, request, result["result"])
            result["cond_file"] = artifact["filename"]
            result["cond_bytes"] = artifact["bytes"]
        return result

    def run_te_probe(self, payload, operation=None):
        cycles = payload.get("cycles", 1)
        if type(cycles) is not int or not 1 <= cycles <= 10:
            raise OperationError("INVALID_REQUEST", "cycles 须在 1–10 之间", 400, False)
        results = []
        for _ in range(cycles):
            results.append({"load": self.te_load_to_vram(),
                            "offload": self.te_offload_to_ram()})
        return {"ok": True, "cycles": cycles, "results": results,
                "te_state": self.te_state()}

    def _te_public_state(self):
        return {key: value for key, value in self._te_step_state.items()
                if key not in ("preprocessed", "last_cond")}

    def te_state(self):
        """返回当前 TE 状态机快照（含 TE 模型当前 device、VRAM）。"""
        import torch
        state = self._te_public_state()
        state["weights"] = self.te.state()
        pc = self._te_model()
        if pc is not None:
            try:
                state["te_device"] = str(next(pc.patcher.model.parameters()).device)
            except Exception:
                state["te_device"] = None
        else:
            state["te_device"] = None
        try:
            state["vram_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            state["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
            free, total = torch.cuda.mem_get_info()
            state["vram_free_gb"] = round(free / 1e9, 3)
            state["vram_total_gb"] = round(total / 1e9, 3)
        except Exception:
            pass
        return state

    def encode_condition(self, prompt, image_bytes_list=None, mode="ref2va",
                         width=640, height=960, ref_image_size="match"):
        """居民式条件编码：要求 TE 已在显存，绝不自动上卡/交换。

        返回 (dict 含 cond/minimax_token_tags, stats)。调用方（pipeline/HTTP）
        必须先通过 POST /api/weights/te/disk2ram + ram2vram 把 TE 放上卡。
        """
        import torch
        t_all = time.perf_counter()
        self.te.preflight_encode(None)
        _, ref_items, qwen_sizes = self._te_preprocess_refs(
            image_bytes_list, mode, width, height, ref_image_size)

        out, enc_stats = self.te._encode_tokens(prompt, ref_items, mode=mode)
        audit = self.te._refresh()
        if audit["units"]["te.model"]["device"] != "cuda:0":
            raise OperationError("WEIGHT_MUTATED", "encode 期间 TE 权重被移出显存",
                                 500, False, audit)

        total_s = time.perf_counter() - t_all
        result = {
            "cond": out["cond"],
            "minimax_token_tags": out["minimax_token_tags"],
        }
        stats = {
            "n_tokens": enc_stats["n_tokens"],
            "cond_shape": enc_stats["cond_shape"],
            "n_images": len(qwen_sizes),
            "mode": mode,
            "qwen_hw": qwen_sizes,
            "tokenize_s": enc_stats.get("tokenize_s"),
            "encode_s": enc_stats.get("encode_s"),
            "total_s": round(total_s, 3),
            "audit": audit,
        }
        print("[encode] %d tokens | %d 图 | mode=%s | encode=%.2fs total=%.2fs"
              % (stats["n_tokens"], len(qwen_sizes), mode,
                 stats["encode_s"], total_s), flush=True)
        self._vram_ledger("encode后", torch)
        return result, stats


def _build_video_vae(comfy, vsd, vmd, device, torch):
    """按 checkpoint 实际 decoder 深度构造视频 VAE（Light 26 层 / 官方 36 层自适应）。

    LynnReal light_vae.build_vae 的最小 server 版：只借"深度自适应"思想，
    不装 FA3/batched decode/compile（sm75 无 FA3）。decoder 构造前临时注入
    num_layers（上游 ViT3DDecoder 默认 36，本 checkout 的 MiniMaxH3VideoVAE
    未透传该参数），构造完即还原。
    tile 几何用 LynnReal 的 Light/官方两档（272/16 vs 256/64）。
    """
    import comfy.ldm.minimax.vae as minimax_vae

    depth = 0
    prefix = "decoder.transformer_blocks."
    idx = set()
    for k in vsd:
        if k.startswith(prefix):
            head = k[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                idx.add(int(head))
    depth = (max(idx) + 1) if idx else 0
    light = (depth == 26)
    tile, overlap = ((272, 16) if light else (256, 64))
    print("[vae] checkpoint decoder depth=%d (%s) tile=%d/%d"
          % (depth, "light" if light else "official", tile, overlap), flush=True)

    orig_decoder_init = minimax_vae.ViT3DDecoder.__init__
    orig_vae_init = minimax_vae.MiniMaxH3VideoVAE.__init__

    def _patched_decoder_init(self, *args, **kwargs):
        kwargs.setdefault("num_layers", depth or 36)
        return orig_decoder_init(self, *args, **kwargs)

    def _patched_vae_init(self, *args, **kwargs):
        kwargs.setdefault("tile_size", tile)
        kwargs.setdefault("tile_overlap_min", overlap)
        return orig_vae_init(self, *args, **kwargs)

    minimax_vae.ViT3DDecoder.__init__ = _patched_decoder_init
    minimax_vae.MiniMaxH3VideoVAE.__init__ = _patched_vae_init
    try:
        return comfy.sd.VAE(sd=vsd, metadata=vmd, device=device)
    finally:
        minimax_vae.ViT3DDecoder.__init__ = orig_decoder_init
        minimax_vae.MiniMaxH3VideoVAE.__init__ = orig_vae_init


class _VedaWarmupShim:
    """给模块级 _install_attention 用的 warmup 转接头。

    _install_attention 是模块级函数（被 Runtime._load 以 rt/attention 等
    注入参数调用），拿不到 self，所以 warmup 逻辑放这个小类里；真正的
    Runtime._veda_warmup 是同一套逻辑的方法形态（见类定义）。
    """

    def run(self, torch, unet, patch, level, fallback):
        if level == "off":
            print("[warmup] veda 跳过（--veda-warmup off）", flush=True)
            return
        tiers = {"probe": 256, "tier1": 2048, "tier2": 8192, "tier3": 24500}
        if level not in tiers:
            raise ValueError("未知 warmup 级别: %s" % level)
        ok, vram = veda_attn_star7.check_nvml_sane(torch)
        if not ok:
            raise RuntimeError("[warmup %s] NVML 异常，中止：%s" % (level, vram))
        tokens = tiers[level]
        heads, dim = 56, 128
        print("[warmup veda probe] n=%d | %s" % (tokens, vram), flush=True)
        parts = []
        for s in range(0, tokens, 4096):
            m = min(4096, tokens - s)
            parts.append(torch.randn(1, heads, m, dim, device="cuda:0",
                                     dtype=torch.float16))
        q = torch.cat(parts, dim=2)
        del parts
        k, v = q.clone(), q.clone()
        torch.cuda.synchronize()
        options = unet.model_options.get("transformer_options", {})
        override = options.get("optimized_attention_override")
        if override is None:
            raise RuntimeError("[warmup veda] unet.model_options 里没有 override")
        opts = dict(options)
        opts["block_index"] = 1
        try:
            out = override(fallback, q, k, v, heads, mask=None, skip_reshape=True,
                           transformer_options=opts)
        except Exception as e:  # noqa: BLE001 - warmup 失败即回退
            raise RuntimeError("[warmup veda probe] override 调用失败: %s: %s"
                               % (type(e).__name__, e))
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out.float()).all())
        del q, k, v, out
        torch.cuda.empty_cache()
        calls = dict(patch.run.calls) if patch is not None else {}
        print("[warmup veda probe] n=%d finite=%s calls=%s | vram=%s"
              % (tokens, finite, calls, vram), flush=True)
        if not finite:
            raise RuntimeError("[warmup veda probe] 输出非 finite")
        # 口径说明：随机噪声 + 合成 options 仍可能被判 dense（无真实 packed
        # layout / conditioning payload），所以 ms 不是稀疏加速比；稀疏是否
        # 真跑只看真任务累计（/api/status.veda_calls）。


def _install_attention(rt, attention, minimax_model, comfy_kitchen, attn_choice,
                       args=None, unet=None):
    """移植 infer_h3_turing 的 attention 选择（不含 warmup）。返回 sage/w8a8/override 或 veda handle。

    veda（路线 B：vendored star7 实现）失败一律回退 int8；
    sdpa/int8/sage/w8a8 行为与 master 一致。

    veda 与 sage/w8a8 的本质差别：sage/w8a8 本想直连 dense kernel，而 veda 的
    override 是唯一入口 —— H3 的真实调用点
    （comfy.ldm.minimax.model.optimized_attention，comfy/ldm/minimax/model.py:200）
    只有经过 comfy 分发器（comfy/ldm/modules/attention.py:200-238 的
    wrap_attn）才会咨询 transformer_options["optimized_attention_override"]。
    所以 veda 档**绝不**把调用点模块级改指到 dense kernel（那正是旧 dev 分支
    veda 恒 dense 的根因二）；这里设 attention.optimized_attention 只是为了让
    **dense 回退**（Veda 拒绝的调用）走 int8 而不是裸 SDPA。
    """
    import torch

    capability = torch.cuda.get_device_capability(0)
    int8_ok = (attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
               and comfy_kitchen.int8_attention_is_available(torch.device("cuda:0")))
    print("[attention] device capability %s | comfy_kitchen int8 available: %s"
          % (capability, int8_ok), flush=True)

    handle = None
    if attn_choice == "veda":
        # 路线 B：vendored star7 VEDA（自带 SM75 CUDA DLL + Triton 无关 core）。
        try:
            from dit_svc import veda_star7
            handle, veda_info = veda_star7.install(
                unet, args, torch=torch,
                log=lambda s: print(s, flush=True))
        except Exception as e:
            import traceback
            print("[veda] 初始化失败，回退 int8: %s: %s"
                  % (type(e).__name__, e), flush=True)
            traceback.print_exc()
            handle, veda_info = None, None
        if handle is None:
            print("[veda] fallback: 走 int8 路径", flush=True)
        else:
            print("[veda] active: %s" % veda_info, flush=True)
            # int8 作 dense 回退底座（Veda 拒绝的调用：非 H3 调用 / dense 层 /
            # layout 不符）。注意：override 仍在 model_options 里，H3 前向
            # 照样经 wrap_attn 咨询它；这里换的只是"没 override 时的默认"。
            if int8_ok:
                attention.optimized_attention = rt._wrap_attn_mem_probe(
                    attention.attention_comfy_kitchen_int8, "veda-dense")
            else:
                attention.optimized_attention = attention.attention_sub_quad
                rt.install_tuned_sub_quad_chunking(attention)
            minimax_model.optimized_attention = attention.optimized_attention
            # 分级 warmup（默认 probe；tier3 全尺寸需显式指定）
            warmup_level = getattr(args, "veda_warmup", "probe") or "probe"
            fallback = (attention.attention_comfy_kitchen_int8 if int8_ok
                        else attention.attention_sub_quad)
            try:
                warmup = _VedaWarmupShim()
                warmup.run(torch, unet, handle, warmup_level, fallback)
            except Exception as e:
                import traceback
                print("[veda] warmup 失败，回退 int8: %s: %s"
                      % (type(e).__name__, e), flush=True)
                traceback.print_exc()
                handle = None
            if handle is None:
                print("[veda] warmup fallback: 走 int8 路径", flush=True)
            return handle
        # 回退到 int8：继续走下面的 int8 分支（handle=None）

    if attn_choice == "sdpa":
        attention.optimized_attention = rt._wrap_attn_mem_probe(attention.attention_pytorch, "sdpa")
        minimax_model.optimized_attention = attention.optimized_attention
        print("[attention] using SDPA (对照组)", flush=True)
        return None

    if attn_choice in ("sage", "w8a8"):
        if rt.TURING_UTILS_DIR not in sys.path:
            sys.path.insert(0, rt.TURING_UTILS_DIR)
        from comfyui_turing_utils.attention.patches import make_attention_override
        override = make_attention_override(attn_choice, torch.device("cuda:0"))
        print("[attention] %s override built: %s"
              % (attn_choice, getattr(override, "turing_utils_attention_implementation", "?")),
              flush=True)
        if int8_ok:
            attention.optimized_attention = rt._wrap_attn_mem_probe(
                attention.attention_comfy_kitchen_int8, attn_choice)
        else:
            attention.optimized_attention = attention.attention_sub_quad
            rt.install_tuned_sub_quad_chunking(attention)
        minimax_model.optimized_attention = attention.optimized_attention
        # 小张量实测：确认内核本体在跑而不是静默回落
        fallback = (attention.attention_comfy_kitchen_int8 if int8_ok
                    else attention.attention_sub_quad)
        probe_q = torch.randn(1, 4, 8, 128, device="cuda:0", dtype=torch.float16)
        probe_ref = fallback(probe_q, probe_q, probe_q, 4, skip_reshape=True)
        probe_out = override(fallback, probe_q, probe_q, probe_q, 4, skip_reshape=True)
        finite = bool(torch.isfinite(probe_out.float()).all())
        diff = (probe_out.float() - probe_ref.float()).abs().max().item()
        print("[attention] %s probe ok=%s shape=%s max|diff|=%.4g"
              % (attn_choice, finite, tuple(probe_out.shape), diff), flush=True)
        del probe_q, probe_ref, probe_out
        torch.cuda.empty_cache()
        return override

    if int8_ok:
        attention.optimized_attention = rt._wrap_attn_mem_probe(
            attention.attention_comfy_kitchen_int8, "int8")
        minimax_model.optimized_attention = attention.optimized_attention
        probe_q = torch.randn(1, 4, 8, 128, device="cuda:0", dtype=torch.float16)
        probe_out = attention.attention_comfy_kitchen_int8(
            probe_q, probe_q, probe_q, 4, skip_reshape=True)
        print("[attention] comfy_kitchen INT8 active, probe ok: %s %s"
              % (tuple(probe_out.shape), probe_out.dtype), flush=True)
        del probe_q, probe_out
        torch.cuda.empty_cache()
        return None

    # 最后防线：收口过的 sub_quad（绝不落裸 SDPA——sm75+bf16 会 Math 模式 OOM）
    attention.optimized_attention = attention.attention_sub_quad
    minimax_model.optimized_attention = attention.optimized_attention
    rt.install_tuned_sub_quad_chunking(attention)
    print("[attention] int8 unavailable -> tuned sub_quadratic", flush=True)
    return None
