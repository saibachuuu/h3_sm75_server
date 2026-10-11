"""VEDA block routes on the SM75 INT8 Tensor Core kernel."""
import torch
from . import base
from ..kernels.native.runtime import Kernel

class TorchGatherBackend(base.Backend):
    name = 'star7-cuda-int8'
    display = 'Star7 VEDA INT8 Tensor Core (SM75)'
    dtypes = (torch.float16,)
    tolerance = 0.04
    chunk_share = 16

    def __init__(self):
        try:
            self.kernel = Kernel()
        except (OSError, RuntimeError, ValueError) as error:
            raise base.BackendUnavailable(
                f'Independent SM75 CUDA library failed to load: {error}. '
                'Update the bundled DLL and manifest together; a compatible NVIDIA '
                'driver is required. No fixed Python/PyTorch build is required.') from error

    def attend(self, q, k, v, block_mask, layout):
        slots, heads, dim = q.shape
        # Quantize each Q warp and K block once, rather than copying K/V per query.
        qb = q.reshape(slots // 16, 16, heads, dim)
        qs = (qb.abs().amax((1, 3)).float() / 127).clamp_min(1e-6)
        quantized = qb / qs[:, None, :, None].to(q.dtype)
        quantized.round_().clamp_(-127, 127)
        qi = quantized.to(torch.int8).reshape(q.shape).transpose(0, 1).contiguous().unsqueeze(0)
        del quantized, qb
        kb = k.reshape(slots // 64, 64, heads, dim)
        ks = (kb.abs().amax((1, 3)).float() / 127).clamp_min(1e-6)
        quantized = kb / ks[:, None, :, None].to(k.dtype)
        quantized.round_().clamp_(-127, 127)
        ki = quantized.to(torch.int8).reshape(k.shape).transpose(0, 1).contiguous().unsqueeze(0)
        del quantized, kb
        qs = qs.transpose(0, 1).unsqueeze(0).contiguous()
        ks = ks.transpose(0, 1).unsqueeze(0).contiguous()
        vh = v.transpose(0, 1).unsqueeze(0)
        routes = (block_mask & layout.kv_ok).repeat_interleave(2, dim=-1)
        key_blocks = slots // 64
        pad = -key_blocks % 32
        if pad:
            routes = torch.nn.functional.pad(routes, (0, pad))
        bits = torch.bitwise_left_shift(torch.ones(32, device=q.device, dtype=torch.int64),
                                      torch.arange(32, device=q.device))
        words = (routes.reshape(heads, slots // 128, -1, 32) * bits).sum(-1).to(torch.int32).unsqueeze(0).contiguous()
        valid = layout.slot_valid.to(torch.int32).contiguous()
        out = torch.empty_like(vh)
        self.kernel.attention(qi, ki, vh, out, qs.contiguous(), ks.contiguous(), words, valid)
        return out.squeeze(0).transpose(0, 1).contiguous()

def create(info):
    if info.kind != 'cuda' or info.cc != (7, 5):
        raise base.BackendUnavailable('Star7 Turing backend requires CUDA SM75')
    return TorchGatherBackend()
