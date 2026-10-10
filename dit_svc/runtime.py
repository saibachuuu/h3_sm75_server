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
import time
import traceback


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
    """服务级常驻状态。load() 在后台线程执行，ready 前拒绝任务。"""

    def __init__(self, args, rt):
        self.args = args
        self.rt = rt                    # infer_h3_turing 模块（工具函数来源）
        self.ready = False
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
        self.ffmpeg = None
        self.attn = args.attn

    # ---------- 阶段上报 ----------
    def set_stage(self, name):
        self.stage = name
        self.stage_t0 = time.perf_counter()
        print("[runtime] stage -> %s" % name, flush=True)

    def stage_elapsed(self):
        return time.perf_counter() - self.stage_t0

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
        print("[load] DiT loaded in %.1fs, vram: %.2f GB"
              % (time.perf_counter() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

        self.set_stage("安装 attention 后端 (%s)" % args.attn)
        self.attn_override = _install_attention(
            rt, attention, minimax_model, comfy_kitchen, args.attn,
            args=args, unet=unet)
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
        vvae = comfy.sd.VAE(sd=vsd, metadata=vmd, device=device)
        vvae.first_stage_model.requires_grad_(False)
        del vsd
        gc.collect()
        torch.cuda.empty_cache()
        print("[load] video VAE in %.1fs" % (time.perf_counter() - t0), flush=True)

        self.set_stage("加载音频 VAE（约 8s）")
        t0 = time.perf_counter()
        asd, amd = rt.stream_load_safetensors_to_dict(args.audio_vae, device="cuda")
        avae = comfy.sd.VAE(sd=asd, metadata=amd, device=device)
        avae.first_stage_model.requires_grad_(False)
        del asd
        gc.collect()
        torch.cuda.empty_cache()
        print("[load] audio VAE in %.1fs" % (time.perf_counter() - t0), flush=True)

        # VAE 常驻（offload_device=cpu 只作兜底，不在任务间搬运）
        vvae.patcher.offload_device = torch.device("cpu")
        avae.patcher.offload_device = torch.device("cpu")

        # ---- 4. sigma shift（采样 guider 用） ----
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        model_shifted = rt.unpack(MiniMaxH3SigmaShift.execute(unet, 12.0, 3.0))[0]
        model_shifted.model.manual_cast_dtype = torch.bfloat16
        model_shifted.offload_device = device
        if self.attn_override is not None:
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

        gc.collect()
        torch.cuda.empty_cache()
        rt.log_memory("Runtime ready")
        self.ready = True
        self.stage = "就绪"
        print("[runtime] READY — free VRAM %.2f GB"
              % (torch.cuda.mem_get_info()[0] / 1e9), flush=True)


def _install_attention(rt, attention, minimax_model, comfy_kitchen, attn_choice,
                       args=None, unet=None):
    """移植 infer_h3_turing 的 attention 选择（不含 warmup）。返回 sage/w8a8/veda 的 override。

    veda（dev 实验）失败一律回退 int8；sdpa/int8/sage/w8a8 行为与 master 一致。
    """
    import torch

    capability = torch.cuda.get_device_capability(0)
    int8_ok = (attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
               and comfy_kitchen.int8_attention_is_available(torch.device("cuda:0")))
    print("[attention] device capability %s | comfy_kitchen int8 available: %s"
          % (capability, int8_ok), flush=True)

    override = None
    if attn_choice == "veda":
        # dev 实验：Veda 稀疏 attention。任何失败 → 回退 int8（安全默认），绝不带病跑。
        try:
            from dit_svc import veda_attn
            override, veda_info = veda_attn.build_veda_override(
                args, unet, torch=torch,
                log=lambda s: print(s, flush=True))
        except Exception as e:
            import traceback
            print("[veda] 初始化失败，回退 int8: %s: %s"
                  % (type(e).__name__, e), flush=True)
            traceback.print_exc()
            override, veda_info = None, None
        if override is None:
            print("[veda] fallback: 走 int8 路径", flush=True)
        else:
            print("[veda] active: %s" % veda_info, flush=True)
            # int8 作 fallback 落点（Veda 稠密回退/非 H3 调用时用）
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

            def _veda_probe_call(q, k, v, heads):
                return override(fallback, q, k, v, heads, skip_reshape=True)
            try:
                veda_attn.run_tiered_warmup(
                    torch, torch.device("cuda:0"), _veda_probe_call, "veda",
                    warmup_level, log=lambda s: print(s, flush=True))
            except Exception as e:
                import traceback
                print("[veda] warmup 失败，回退 int8: %s: %s"
                      % (type(e).__name__, e), flush=True)
                traceback.print_exc()
                override = None
            if override is None:
                print("[veda] warmup fallback: 走 int8 路径", flush=True)
            return override
        # 回退到 int8：继续走下面的 int8 分支（override=None）

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
