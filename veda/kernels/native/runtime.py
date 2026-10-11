"""Star7 VEDA SM75 原生 kernel 的稳定 C ABI 加载器（ctypes）。

public Tensor pointers/strides，不碰 Python/ATen 绑定。

移植修改（相对 minimax-h3-chunk-star7/vendor/veda/kernels/native/runtime.py）：
  1. ROOT 不再用 parents[4]（本仓库里层级不同），改为：
     env STAR7_VEDA_SM75_BIN 显式指定 > <repo>/bin 目录 > <package>/bin。
  2. 其余（manifest 读取、sha256 校验、ABI 版本/args size 校验、错误串）
     与原版逐字一致，确保与原版 DLL 的契约不变。
"""
import ctypes
import hashlib
import json
import os
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent          # <repo>/veda
_LIBRARY = None


def _bin_dir() -> pathlib.Path:
    """bin 目录优先序：env > <repo>/bin > <veda_pkg>/bin。"""
    override = os.environ.get("STAR7_VEDA_SM75_BIN")
    if override:
        return pathlib.Path(override).resolve()
    for candidate in (_PKG_ROOT.parent / "bin", _PKG_ROOT / "bin"):
        if (candidate / "veda_sm75_manifest.json").is_file():
            return candidate
    return _PKG_ROOT.parent / "bin"


ROOT = _bin_dir()


class Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('q', 'k', 'v', 'out', 'qs', 'ks', 'routes', 'valid', 'policy', 'stream')]
    _fields_ += [(name, ctypes.c_int32) for name in ('device', 'heads', 'slots')]
    _fields_ += [(name, ctypes.c_int64 * 3) for name in
                 ('q_stride', 'k_stride', 'v_stride', 'out_stride')]


def _load():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    manifest = json.loads((ROOT / 'veda_sm75_manifest.json').read_text(encoding='utf-8-sig'))
    platform = 'windows_x64' if sys.platform == 'win32' else 'linux_x86_64'
    if platform not in manifest:
        raise RuntimeError(f'No bundled VEDA SM75 binary for {platform}; build.py can build the CUDA library')
    entry = manifest[platform]
    path = ROOT / entry['file']
    if not path.is_file():
        raise RuntimeError(f'Missing independent VEDA CUDA library: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('VEDA CUDA library checksum mismatch; update the binary and manifest together')
    library = ctypes.CDLL(str(path))
    library.star7_veda_abi_version.argtypes = []
    library.star7_veda_abi_version.restype = ctypes.c_int
    library.star7_veda_args_size.argtypes = []
    library.star7_veda_args_size.restype = ctypes.c_int
    library.star7_veda_cuda_runtime_version.argtypes = []
    library.star7_veda_cuda_runtime_version.restype = ctypes.c_int
    library.star7_veda_last_error.argtypes = []
    library.star7_veda_last_error.restype = ctypes.c_char_p
    library.star7_veda_attention.argtypes = [ctypes.POINTER(Args)]
    library.star7_veda_attention.restype = ctypes.c_int
    if library.star7_veda_abi_version() != 1 or library.star7_veda_args_size() != ctypes.sizeof(Args):
        raise RuntimeError('VEDA CUDA C ABI mismatch')
    _LIBRARY = library
    return library


class Kernel:
    def __init__(self):
        self.library = _load()

    def attention(self, q, k, v, out, qs, ks, routes, valid):
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or q.shape[2] % 128:
            raise ValueError('VEDA SM75 expects [1, heads, padded slots, 128]')
        if k.shape != q.shape or v.shape != q.shape or out.shape != q.shape:
            raise ValueError('VEDA Q/K/V/output shapes must agree')
        tensors = (q, k, v, out, qs, ks, routes, valid)
        dtypes = (torch.int8, torch.int8, torch.float16, torch.float16,
                  torch.float32, torch.float32, torch.int32, torch.int32)
        if any(not t.is_cuda or t.device != q.device or t.dtype != dtype for t, dtype in zip(tensors, dtypes)):
            raise ValueError('VEDA tensor device or dtype mismatch')
        heads, slots = int(q.shape[1]), int(q.shape[2])
        if qs.numel() != heads * (slots // 16) or ks.numel() != heads * (slots // 64):
            raise ValueError('VEDA quantization scale size mismatch')
        if valid.numel() != slots or routes.numel() != heads * (slots // 128) * ((slots // 64 + 31) // 32):
            raise ValueError('VEDA route/validity size mismatch')
        if any(not t.is_contiguous() for t in (q, k, qs, ks, routes, valid)):
            raise ValueError('VEDA Q/K/scales/routes/validity must be contiguous')
        with torch.cuda.device(q.device):
            policy = torch.ones((slots + 63) // 64, dtype=torch.uint8, device=q.device)
            args = Args(*(t.data_ptr() for t in tensors), policy.data_ptr(),
                        torch.cuda.current_stream(q.device).cuda_stream,
                        q.device.index, heads, slots,
                        *( (ctypes.c_int64 * 3)(*t.stride()[:3]) for t in (q, k, v, out)))
            if self.library.star7_veda_attention(ctypes.byref(args)):
                error = self.library.star7_veda_last_error().decode('utf-8', 'replace')
                raise RuntimeError(f'VEDA SM75 CUDA launch failed: {error}')
