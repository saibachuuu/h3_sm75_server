"""TE（文本编码器）后端：整组 unit + encode 原语。

TE 默认是整组模型：``te.model`` 是 ComfyUI CLIP 实例（Qwen3-VL +
tokenizer/视觉塔），``te.proj`` 是 ClipProj 投影。Comfy 的 ``load_clip`` /
``load_text_encoder_state_dicts`` 一次构造整组，因此：

* disk2ram 只允许整组（te.model + te.proj 同进同出），拒绝幽灵组合；
* 物理账走 ``weights.storage_inventory``：量化包装展开、按 storage 去重；
* ClipProj 的 GPU 缓存（``_proj`` 的 ``_gpu`` 副本）计入 te.proj 的 CUDA
  驻留，purge 前必须先 vram2ram 清掉；
* encode 是 resident 推理：要求 te.model 已在 vram，绝不自动上卡；编码后
  重新测量设备，被内部流程改变时 fail-closed。

本模块不 import torch/Comfy：只有真正执行动作时才惰性导入。
"""
import base64
import json
import os
import sys
import time

from dit_svc import weights as W
from dit_svc.group_weights import GroupBackend, suggestions
from dit_svc.operations import OperationError

MODEL_ID = "te"
UNIT_MODEL = "te.model"
UNIT_PROJ = "te.proj"
ADAPTER_VERSION = "te-comfy-v0"
VALID_KINDS = ("4b", "gguf", "checkpoint")


class TeBackend(GroupBackend):
    MODEL_ID = MODEL_ID
    ADAPTER_VERSION = ADAPTER_VERSION
    VALID_KINDS = VALID_KINDS
    PRIMARY_UNIT = UNIT_MODEL

    def __init__(self, runtime, registry):
        super().__init__(runtime, registry)
        self._clip = None
        self._projected = False
        self.encoded = 0

    # ---------------------------------------------------------- 兼容属性
    @property
    def pc(self):
        return self.obj

    @pc.setter
    def pc(self, value):
        self.obj = value

    def extra_state(self):
        return {
            "kind": (self.config or {}).get("kind"),
            "path": (self.config or {}).get("path"),
            "proj": (self.config or {}).get("proj"),
            "mmproj": (self.config or {}).get("mmproj"),
            "encoded": self.encoded,
        }

    # ---------------------------------------------------------- catalog
    def default_kind(self, cfg):
        path = (cfg.get("path") or "").lower()
        if cfg.get("mmproj") and path.endswith(".gguf"):
            return "gguf"
        if cfg.get("proj"):
            return "4b"
        if path.endswith(".gguf"):
            return "gguf"
        return "checkpoint"

    def validate_config(self, cfg):
        if cfg["kind"] == "gguf" and not cfg["path"].lower().endswith(".gguf"):
            raise ValueError("te.kind=gguf 需要 .gguf 主权重")
        return cfg

    def build_units(self, cfg):
        units = [W.UnitInfo(
            id=UNIT_MODEL, checkpoint_keys=(),
            expected_bytes=os.path.getsize(cfg["path"]),
            groups=(MODEL_ID,), factory="te-comfy-v1",
            capabilities={"encode": True, "text_only": False})]
        if cfg.get("proj"):
            proj_path = cfg["proj"]
            expected = (os.path.getsize(proj_path)
                        if os.path.isabs(proj_path) else 0)
            units.append(W.UnitInfo(
                id=UNIT_PROJ, checkpoint_keys=(),
                expected_bytes=expected, groups=(MODEL_ID,),
                factory="clipproj-v1", capabilities={"encode": True}))
        return units

    def _require_group(self, units):
        configured = (self.catalog.unit_ids() if self.catalog else [])
        if UNIT_MODEL in units and UNIT_PROJ in configured \
                and UNIT_PROJ not in units:
            raise OperationError(
                "SHARED_UNIT_REQUIRED",
                "te.proj 是 te.model 的构造依赖，必须一起选择", 409, False,
                {"missing": [UNIT_PROJ],
                 "hint": "用 mode=all，或 units=[\"te.model\",\"te.proj\"]"})
        if UNIT_PROJ in units and UNIT_MODEL not in units:
            raise OperationError(
                "SHARED_UNIT_REQUIRED",
                "te.proj 不能脱离 te.model 单独装载", 409, False,
                {"missing": [UNIT_MODEL]})

    # ---------------------------------------------------------- 构造/搬运
    def construct_cpu(self, operation=None):
        import torch

        kind = self.config["kind"]
        if kind == "gguf":
            clip = self._load_gguf_clip(torch)
        else:
            clip = self._load_checkpoint_clip(torch)
        pc = clip
        self._projected = False
        if kind != "gguf" and self.config.get("proj"):
            pc = self._wrap_projected(clip)
            self._projected = True
        self._clip = clip
        self.obj = pc

    def _clip_model(self):
        return self.obj.patcher.model if self.obj is not None else None

    def move_to(self, device):
        if self.obj is None:
            raise RuntimeError("TE 未加载")
        self.runtime.te_residence.move(self, device)

    def drop_caches(self):
        if self._projected and self.obj is not None:
            cache = self.obj.__dict__.get("_gpu")
            if cache:
                cache.clear()

    def release(self):
        self.runtime.te_residence.unregister()
        refs = []
        model = self._clip_model()
        if model is not None:
            refs.append(self._release_ref(model))
        if self.obj is not None:
            refs.append(self._release_ref(self.obj))
        self.drop_caches()
        self.obj = None
        self._clip = None
        self._projected = False
        return refs

    def inventory(self):
        out = {"units": {}, "uninspectable": [], "storage_digest": []}
        model = self._clip_model()
        if model is None:
            return out
        inv = W.storage_inventory(model, owner=UNIT_MODEL)
        out["units"][UNIT_MODEL] = {
            "device": self.dominant_device(model),
            "cpu": inv.total_cpu, "cuda": inv.total_cuda}
        out["uninspectable"] += list(inv.uninspectable)
        digest = [entry["storage_id"] for entry in inv.entries]
        if self._projected:
            proj = self.obj.__dict__.get("_proj") or {}
            gpu = self.obj.__dict__.get("_gpu") or {}
            inv_proj = W.storage_inventory(proj, owner=UNIT_PROJ)
            inv_gpu = W.storage_inventory(gpu, owner=UNIT_PROJ + ".gpu")
            out["units"][UNIT_PROJ] = {
                "device": "cuda:0" if inv_gpu.total_cuda > 0 else "cpu",
                "cpu": inv_proj.total_cpu, "cuda": inv_gpu.total_cuda}
            out["uninspectable"] += (list(inv_proj.uninspectable)
                                     + list(inv_gpu.uninspectable))
            digest += [e["storage_id"] for e in inv_proj.entries]
            digest += [e["storage_id"] for e in inv_gpu.entries]
        elif self.catalog is not None and UNIT_PROJ in self.catalog.units:
            out["units"][UNIT_PROJ] = {"device": "cpu", "cpu": 0, "cuda": 0}
        out["storage_digest"] = digest
        return out

    def _load_checkpoint_clip(self, torch):
        import comfy.sd
        path = self.config["path"]
        print("[te] load_clip: %s" % os.path.basename(path), flush=True)
        return comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=[],
            clip_type=comfy.sd.CLIPType.MINIMAX, disable_dynamic=True,
            model_options={
                "initial_device": torch.device("cpu"),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
            })

    def _load_gguf_clip(self, torch):
        import importlib
        import types
        import comfy.sd

        path = self.config["path"]
        mmproj = self.config.get("mmproj")
        root = self._comfy_root()
        pkg_dir = os.path.join(root, "custom_nodes", "ComfyUI-GGUF")
        if not os.path.isdir(pkg_dir):
            raise FileNotFoundError("找不到 ComfyUI-GGUF: %s" % pkg_dir)
        if "comfyui_gguf" not in sys.modules:
            pkg = types.ModuleType("comfyui_gguf")
            pkg.__path__ = [pkg_dir]
            sys.modules["comfyui_gguf"] = pkg
        loader = importlib.import_module("comfyui_gguf.loader")
        ops = importlib.import_module("comfyui_gguf.ops")
        print("[te] 读 GGUF: %s" % os.path.basename(path), flush=True)
        state = loader.gguf_clip_loader(path)
        has_vis = any(k.startswith(("visual.", "model.visual.")) for k in state)
        if not has_vis and mmproj and os.path.isfile(mmproj):
            print("[te] 追加 mmproj: %s" % os.path.basename(mmproj), flush=True)
            state.update(loader.gguf_qwen3vl_mmproj_loader(mmproj))
            has_vis = any(k.startswith(("visual.", "model.visual.")) for k in state)
        print("[te] GGUF 张量 %d 个，视觉塔=%s" % (len(state), has_vis), flush=True)
        renamed = 0
        for key in [k for k in list(state) if k.startswith("model.visual.")]:
            state[key[len("model."):]] = state.pop(key)
            renamed += 1
        if renamed:
            print("[te] 视觉键 model.visual.* -> visual.*: %d" % renamed,
                  flush=True)
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.MINIMAX, state_dicts=[state],
            model_options={
                "custom_operations": ops.GGMLOps,
                "initial_device": torch.device("cpu"),
                "load_device": torch.device("cuda:0"),
                "offload_device": torch.device("cpu"),
            }, embedding_directory=[], disable_dynamic=True)
        state.clear()
        return clip

    def _wrap_projected(self, clip):
        root = self._comfy_root()
        custom = os.path.join(root, "custom_nodes")
        if custom not in sys.path:
            sys.path.insert(0, custom)
        if not os.path.isdir(os.path.join(custom, "clipproj")):
            raise FileNotFoundError("custom_nodes/clipproj 缺失（ClipProj 未安装）")
        from clipproj.clipproj_projection import register_folder
        import folder_paths
        from clipproj.clipproj_nodes import ProjectedCLIP

        register_folder()
        name = self._projection_name(folder_paths)
        print("[te] ClipProj 投影: %s" % name, flush=True)
        return ProjectedCLIP(clip, name)

    def _projection_name(self, folder_paths):
        value = self.config["proj"]
        if os.path.isabs(value):
            folder = os.path.dirname(value)
            entry = folder_paths.folder_names_and_paths.get("clip_projections")
            if entry is None:
                folder_paths.folder_names_and_paths["clip_projections"] = (
                    [folder], {".pt", ".safetensors"})
            elif folder not in entry[0]:
                entry[0].append(folder)
            return os.path.basename(value)
        return value

    def _comfy_root(self):
        import comfy
        candidates = []
        package_path = getattr(comfy, "__path__", None)
        if package_path:
            candidates.append(os.path.dirname(os.path.abspath(list(package_path)[0])))
        package_file = getattr(comfy, "__file__", None)
        if package_file:
            candidates.append(os.path.dirname(os.path.dirname(
                os.path.abspath(package_file))))
        candidates.extend(os.path.abspath(p) for p in sys.path if p)
        for candidate in candidates:
            if os.path.isdir(os.path.join(candidate, "custom_nodes")):
                return candidate
        raise FileNotFoundError("找不到 ComfyUI 根目录（custom_nodes 所在处）")

    # ---------------------------------------------------------- encode
    def preflight_encode(self, request=None):
        """要求 te.model 已在 vram；缺资源给有序建议，绝不自动搬。"""
        self._require_configured()
        record = self.registry.unit_record(MODEL_ID, UNIT_MODEL)
        if record["state"] != "vram":
            raise OperationError(
                "MISSING_UNITS", "encode 需要 te 已在显存（当前 %s）"
                % record["state"], 409, False,
                {"missing": [MODEL_ID], "state": record["state"],
                 "suggested": suggestions(MODEL_ID, record["state"])})
        if not self.runtime.device_healthy:
            raise OperationError("DEVICE_UNHEALTHY", "CUDA 设备不可用", 503, False)

    def run_encode(self, request, operation=None):
        with self._lock:
            self.preflight_encode(request)
            import torch
            t_all = time.perf_counter()
            self._stage("TE 预处理参考图（CPU）", operation)
            _, ref_items, qwen_sizes = self._preprocess(
                request["image_b64s"], request["mode"], request["width"],
                request["height"], request["ref_image_size"])
            self._stage("TE 推理", operation)
            before = self._cuda_bytes(torch)
            try:
                out, stats = self._encode_tokens(
                    request["prompt"], ref_items, request["mode"], operation)
            except OperationError:
                raise
            except Exception as exc:
                self._mark_error()
                raise OperationError(
                    "ENCODE_FAILED", "TE 编码失败: %s: %s"
                    % (type(exc).__name__, exc), 500, True,
                    {"actual": self._safe_state()}) from exc
            audit = self._refresh()
            if audit["units"][UNIT_MODEL]["device"] != "cuda:0":
                self._mark_error()
                raise OperationError(
                    "WEIGHT_MUTATED",
                    "encode 期间 TE 权重设备被改变（actual %s）"
                    % audit["units"][UNIT_MODEL]["device"], 500, False, audit)
            artifact = self._save_cond(out, request, stats)
            self.encoded += 1
            total_s = time.perf_counter() - t_all
            result = {
                "artifact": artifact,
                "stats": dict(stats),
                "audit": audit,
                "elapsed_s": round(total_s, 3),
                "cuda_bytes_before": before,
                "cuda_bytes_after": self._cuda_bytes(torch),
            }
            print("[te] encode %d tokens | %d 图 | %.2fs | %s"
                  % (stats["n_tokens"], len(qwen_sizes), stats["encode_s"],
                     artifact["id"]), flush=True)
            return result

    def _preprocess(self, image_b64s, mode, width, height, ref_image_size):
        import io
        import numpy as np
        import torch
        from PIL import Image
        import comfy.utils as cu
        from common.protocol import target_frame_size

        imgs = []
        for value in image_b64s:
            if value.startswith("data:"):
                value = value.split(",", 1)[1]
            imgs.append(base64.b64decode(value))
        if not (1 <= len(imgs) <= 9):
            raise ValueError("参考图须为 1–9 张")
        if mode == "fl2va" and len(imgs) != 1:
            raise ValueError("fl2va 仅支持单图")

        qwen_sizes = []
        ref_items = []
        for raw in imgs:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.asarray(img).astype(np.float32) / 255.0
            frame = torch.from_numpy(arr).unsqueeze(0)
            rh, rw = int(frame.shape[1]), int(frame.shape[2])
            tw, th = target_frame_size(mode, rw, rh, width, height, ref_image_size)
            samples = frame[..., :3].movedim(-1, 1)
            samples = cu.common_upscale(samples, tw, th, "lanczos", "disabled")
            rz = samples.movedim(1, -1)
            qwen_sizes.append([int(rz.shape[1]), int(rz.shape[2])])
            ref_items.append({"type": "image", "data": rz})
        return imgs, ref_items, qwen_sizes

    def _encode_tokens(self, prompt, ref_items, mode, operation=None):
        """All encode callers share the resident guard, including legacy endpoints."""
        import torch
        with self._lock:
            self.preflight_encode()
            try:
                with self.runtime.te_residence.inference(self, operation) as proof:
                    with torch.no_grad():
                        out, stats = self._encode_tokens_impl(prompt, ref_items, mode)
                stats["residence"] = proof
                return out, stats
            except torch.OutOfMemoryError as exc:
                audit = self._refresh()
                # No weights moved: insufficient workspace does not corrupt TE.
                raise OperationError("CUDA_OOM", "TE 推理工作区不足，未自动卸载其它模型",
                                     409, True, {"actual": audit}) from exc

    def _encode_tokens_impl(self, prompt, ref_items, mode):
        pc = self.obj
        if pc is None:
            raise RuntimeError("TE 未加载")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        t0 = time.perf_counter()
        if mode == "fl2va":
            tokens = pc.tokenize(prompt, images=[ref_items[0]["data"]])
        else:
            tokens = pc.tokenize(prompt, minimax_ref_items=ref_items)
        tokenize_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cond_list = pc.encode_from_tokens_scheduled(tokens)
        encode_s = time.perf_counter() - t0
        tensor, values = cond_list[0][0], cond_list[0][1]
        tags = values.get("minimax_token_tags")
        if tags is None:
            raise OperationError(
                "INCOMPATIBLE_COND",
                "TE 没有返回 minimax_token_tags，不能写缺少必需 key 的 cond",
                422, False)
        out = {"cond": tensor.detach().float().cpu().contiguous(),
               "minimax_token_tags": tags.detach().cpu().contiguous()}
        stats = {"n_tokens": int(tensor.shape[1]),
                 "cond_shape": list(tensor.shape),
                 "tokenize_s": round(tokenize_s, 3),
                 "encode_s": round(encode_s, 3),
                 "mode": mode}
        return out, stats

    def _save_cond(self, out, request, stats):
        from safetensors.torch import save

        metadata = {
            "format": "h3.cond.v1",
            "mode": request["mode"],
            "width": str(request["width"]),
            "height": str(request["height"]),
            "ref_image_size": request["ref_image_size"],
            "n_tokens": str(stats["n_tokens"]),
            "n_images": str(len(request["image_b64s"])),
            "te_kind": (self.config or {}).get("kind", ""),
            "te_recipe": os.path.basename((self.config or {}).get("path", "")),
            "encode_s": str(stats["encode_s"]),
        }
        data = save(out, metadata=metadata)
        artifacts = getattr(self.runtime, "artifacts", None)
        filename = "cond_te_%d_%s.safetensors" % (
            int(time.time()), os.urandom(3).hex())
        if artifacts is None:
            raise OperationError("ENV_NOT_READY", "artifact store 未初始化",
                                 503, True)
        return artifacts.commit("cond", filename, data, metadata=metadata,
                                fmt="h3.cond.v1")


# 兼容旧导入名
TeManager = TeBackend
