"""常驻文本编码器 + ref2va/fl2va 编码 + 内容哈希磁盘缓存。

- 编码逻辑移植自 h3-runner/embed_h3.py，tokenize 调用方式与 embed_server.Encoder 一致；
- Encoder（Qwen3-VL-32B int4 convrot）在服务启动时加载一次并常驻；
- 输出 safetensors 契约 = infer_h3_turing.read_cond 所需：
    cond / minimax_token_tags / ref_image|first_frame / (attention_mask|pooled_output)?

依赖 embed_server.py（位于 work-on-comfyui 根目录）：import embed_server 会触发其
COMFY_DIR/sys.path 副作用，因此本模块只在 EncoderService.load() 内 import。
"""
import hashlib
import io
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048


def _resize(image, width, height, crop, comfy_utils):
    """[1,H,W,C] float -> 指定尺寸（与 embed_h3._resize 相同）。"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy_utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


class EncoderService:
    def __init__(self, model_path, cache_dir, use_int4_mma=True):
        self.model_path = model_path
        self.cache_dir = os.path.abspath(cache_dir)
        self.use_int4_mma = use_int4_mma
        self.enc = None
        self.ready = False
        self.load_error = None
        self.load_s = 0.0
        self.jobs = 0
        self.cache_hits = 0
        self._cache_lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------- 启动时加载一次（阻塞，调用方在 serve 前执行） ----------
    def load(self):
        t0 = time.perf_counter()
        try:
            from embed_server import Encoder  # noqa: F401  (触发 COMFY_DIR/sys.path 副作用)
            enc = Encoder(self.model_path, use_int4_mma=self.use_int4_mma)
            enc.load()
            self.enc = enc
            self.load_s = time.perf_counter() - t0
            self.ready = True
            print("[cond] encoder ready in %.1fs (int4_mma=%s, patched=%d)"
                  % (self.load_s, enc.use_int4_mma, enc.patched), flush=True)
        except Exception as e:
            self.load_error = "%s: %s" % (type(e).__name__, e)
            print("[cond] encoder load FAILED: %s" % self.load_error, flush=True)
            raise

    # ---------- 内容哈希缓存 ----------
    def _cache_key(self, prompt, image_bytes, mode, width, height, ref_image_size):
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(image_bytes)
        h.update(("\x00%s\x00%d\x00%d\x00%s" % (mode, width, height, ref_image_size)).encode())
        h.update(os.path.basename(self.model_path).encode("utf-8"))
        return h.hexdigest()

    def _cached_path(self, key):
        return os.path.join(self.cache_dir, key + ".safetensors")

    # ---------- 编码（同步；内部用 enc.lock 串行化 GPU 段） ----------
    def encode(self, prompt, image_bytes, mode="ref2va", width=640, height=960,
               ref_image_size="match"):
        if not self.ready:
            raise RuntimeError("编码器未就绪：%s" % (self.load_error or "加载中"))
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        if not image_bytes:
            raise ValueError("缺少图像输入")

        key = self._cache_key(prompt, image_bytes, mode, width, height, ref_image_size)
        path = self._cached_path(key)
        with self._cache_lock:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                self.cache_hits += 1
                return {"hash": key, "file": os.path.basename(path), "cached": True,
                        "stats": {"cached": True, "n_tokens": None}}

        import comfy.utils as cu  # embed_server 已把 COMFY_DIR 加进 sys.path

        t_start = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        frame = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3] float 0..1
        raw_h, raw_w = int(frame.shape[1]), int(frame.shape[2])

        if mode == "fl2va":
            frame_rz = _resize(frame[:1], width, height, "disabled", cu)
            images_arg, ref_items = [frame_rz], None
        else:  # ref2va
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (raw_w * raw_h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(raw_w, raw_h))
            tw = max(CANVAS_MULTIPLE, round(raw_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(raw_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frame_rz = _resize(frame[:1], tw, th, "disabled", cu)
            images_arg, ref_items = None, [{"type": "image", "data": frame_rz}]

        with self.enc.lock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if mode == "fl2va":
                tokens = self.enc.clip.tokenize(prompt, images=images_arg)
            else:
                tokens = self.enc.clip.tokenize(prompt, minimax_ref_items=ref_items)
            cond = self.enc.clip.encode_from_tokens_scheduled(tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_encode = time.perf_counter() - t0
            self.jobs += 1

        tensor, values = cond[0][0], cond[0][1]
        rz = frame_rz[0].contiguous().cpu()  # VAE 参考帧（远程 VAE-encode 用）
        out = {
            "cond": tensor.float().cpu().contiguous(),
            "minimax_token_tags": values["minimax_token_tags"].cpu().contiguous(),
            "first_frame": rz,
            # clone：safetensors save_file 拒绝共享存储的张量，两字段必须各占一份
            "ref_image": rz.clone(),
        }
        for k in ("attention_mask", "pooled_output"):
            if values.get(k) is not None:
                out[k] = values[k].float().cpu().contiguous()

        metadata = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "mode": json.dumps(mode),
            "width": json.dumps(width),
            "height": json.dumps(height),
            "ref_image_size": json.dumps(ref_image_size),
            "n_tokens": json.dumps(int(tensor.shape[1])),
            "source_model": json.dumps(os.path.basename(self.model_path)),
            "encode_s": json.dumps(round(t_encode, 2)),
        }
        with self._cache_lock:
            tmp = path + ".part"
            save_file(out, tmp, metadata=metadata)
            os.replace(tmp, path)  # 原子落盘，并发安全

        stats = {
            "cached": False,
            "n_tokens": int(tensor.shape[1]),
            "cond_shape": list(tensor.shape),
            "encode_s": round(t_encode, 3),
            "total_s": round(time.perf_counter() - t_start, 3),
            "image_hw": [raw_h, raw_w],
            "resized_hw": [int(frame_rz.shape[1]), int(frame_rz.shape[2])],
            "file_bytes": os.path.getsize(path),
        }
        print("[cond] encoded %s tokens=%d mode=%s %.2fs -> %s"
              % (key[:12], stats["n_tokens"], mode, t_encode, os.path.basename(path)),
              flush=True)
        return {"hash": key, "file": os.path.basename(path), "cached": False, "stats": stats}

    def health(self):
        return {
            "ok": self.ready,
            "loading": (not self.ready) and self.load_error is None,
            "error": self.load_error,
            "model": os.path.basename(self.model_path),
            "load_s": round(self.load_s, 1),
            "jobs": self.jobs,
            "cache_hits": self.cache_hits,
        }
