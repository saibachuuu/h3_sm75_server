"""Explicit diagnostic backend: the same VEDA mask, bounded FP32 attention."""
import torch
import torch.nn.functional as F
from . import base


class TorchReferenceBackend(base.Backend):
    name = 'torch-reference'
    display = 'Star7 FP32 sparse reference (diagnostic)'
    dtypes = (torch.bfloat16, torch.float16)
    chunk_share = 16

    def attend(self, q, k, v, block_mask, layout):
        out = torch.zeros_like(q)
        valid = layout.valid_count.cpu().tolist()
        offsets = torch.arange(128, device=q.device)
        for head in range(q.shape[1]):
            for tile, count in enumerate(valid):
                if not count:
                    continue
                kept = torch.nonzero(block_mask[head, tile] & layout.kv_ok.flatten()).flatten()
                indices = kept[:, None] * 128 + offsets
                real = offsets[None, :] < layout.valid_count.index_select(0, kept)[:, None]
                indices = indices[real].long()
                if not indices.numel():
                    continue
                start = tile * 128
                query = q[start:start + count, head].float()[None, None]
                keys = k[:, head].index_select(0, indices).float()[None, None]
                values = v[:, head].index_select(0, indices).float()[None, None]
                result = F.scaled_dot_product_attention(query, keys, values,
                    dropout_p=0.0, is_causal=False, scale=q.shape[-1] ** -0.5)
                out[start:start + count, head] = result[0, 0].to(q.dtype)
        return out


def create(info):
    if info.kind != 'cuda' or info.cc is None or info.cc < (8, 0):
        raise base.BackendUnavailable('The explicit reference diagnostic is for SM80+ CUDA devices')
    return TorchReferenceBackend()
