"""Opt-in BF16 comparison path; preserves VEDA's predictor and tile mask."""
import functools

import torch

from . import base
from ..core import selection


@functools.cache
def _kernel():
    from ..kernels.sage import sparse_bf16
    return sparse_bf16


class TritonBF16Backend(base.Backend):
    name = 'triton-bf16'
    display = 'Star7 Triton BF16 sparse (comparison)'
    dtypes = (torch.bfloat16,)

    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        return _kernel().attend(q, k, v, index, count, layout.valid_count)

    def warmup_note(self):
        return 'compiling BF16 sparse comparison kernel (first run only)'


def create(info):
    if info.kind != 'cuda' or info.cc is None or info.cc < (8, 0):
        raise base.BackendUnavailable('VEDA BF16 sparse comparison needs CUDA SM80 or newer')
    try:
        _kernel()
    except ImportError as error:
        raise base.BackendUnavailable(f'Cannot load BF16 sparse comparison: {error}') from error
    return TritonBF16Backend()
