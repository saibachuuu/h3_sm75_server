"""Triton INT8 block-sparse attention: the default CUDA path.

This is the same arithmetic ComfyUI runs behind `--use-sage-attention`
(`from sageattention import sageattn`), so its accuracy is ComfyUI's, with
Veda's block sparsity on top. On an RTX 5070 it reaches ~125 TFLOPS dense
against ~45 for our bf16 CuTe kernel, and ~1.3% relative error against an
fp32 reference where FP8 costs ~5.3%.

Covers every CUDA GPU from SM80 on, because Triton does: one backend
instead of one per SM family, and no fallback behind it (see
backends/__init__.py for why that is deliberate).

Self-contained on purpose (see backends/base.py): imports only this file,
torch, and `veda_comfy.kernels.sage`.
"""

from __future__ import annotations

import functools
import sys

import torch

from . import base
from ..core import selection
from .sampled_reference import sampled_outputs, numerical_error

_MIN_CC = (8, 0)


@functools.cache
def _kernel():
    from ..kernels.sage import sparse_int8  # pylint: disable=import-outside-toplevel
    return sparse_int8


class TritonInt8Backend(base.Backend):
    """SageAttention's INT8 arithmetic, walking only the kept tiles."""

    name = 'triton-int8'
    display = 'Triton INT8'
    dtypes = (torch.bfloat16, torch.float16)
    # INT8 Q and K with one scale per block: measured 1.3% relative error
    # on a dense problem and 2.7% pointwise on the padding-heavy self-test
    # problem, which is the format's floor rather than a fault. 5% still
    # catches the real ones (a mis-strided V read came out at 650%).
    tolerance = 0.05

    def __init__(self, label: str, fp32_pv: bool = False, audit: bool = False):
        self.fp32_pv = fp32_pv
        self.audit = audit
        self.stages = 3
        self._verified_shapes = set()
        self.checks = []
        self._label = label
        self._last_layer = None
        self._probe_pending = True
        self.display = (f'Triton INT8 QK / FP16 PV / FP32 accumulation ({label})' if fp32_pv
                        else f'Triton INT8 ({label})')

    def begin_attention(self, layer=None):
        if self.fp32_pv and not self.audit:
            self._verified_shapes.clear()
        if self.audit and layer is not None:
            new_call = self._last_layer is None or layer <= self._last_layer
            self._last_layer = layer
            self._probe_pending = new_call and len(self.checks) < 2
            if self._probe_pending:
                self._verified_shapes.clear()

    def begin_run(self):
        self._verified_shapes.clear()
        self.checks.clear()
        self._last_layer = None
        self._probe_pending = True

    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.no_grad():
            if not self.fp32_pv and not self.audit:
                return _kernel().attend(q, k, v, index, count, layout.valid_count)
            output = _kernel().attend(q, k, v, index, count, layout.valid_count,
                                      fp32_pv=self.fp32_pv, num_stages=self.stages)
            shape = (tuple(q.shape), q.dtype, str(q.device))
            if shape not in self._verified_shapes and (not self.audit or (self._probe_pending and len(self.checks) < 2)):
                checked_heads = min(2, q.shape[1]) if self.audit else q.shape[1]
                def error_for(result):
                    got, expected = sampled_outputs(q[:, :checked_heads], k[:, :checked_heads], v[:, :checked_heads],
                        block_mask[:checked_heads], layout.valid_count, result[:, :checked_heads])
                    return numerical_error(got, expected)
                error, maximum, cosine = error_for(output)
                attempts = [(self.stages, self.fp32_pv, round(error, 5), round(cosine, 5))]
                acceptable = (error <= 0.12 and cosine >= 0.98) or maximum <= 1e-5
                if not acceptable and self.stages != 1:
                    output = _kernel().attend(q, k, v, index, count, layout.valid_count,
                                              fp32_pv=self.fp32_pv, num_stages=1)
                    error, maximum, cosine = error_for(output)
                    attempts.append((1, self.fp32_pv, round(error, 5), round(cosine, 5)))
                    acceptable = (error <= 0.12 and cosine >= 0.98) or maximum <= 1e-5
                    if acceptable:
                        self.stages = 1
                if not acceptable and not self.fp32_pv:
                    output = _kernel().attend(q, k, v, index, count, layout.valid_count,
                                              fp32_pv=True, num_stages=1)
                    error, maximum, cosine = error_for(output)
                    attempts.append((1, True, round(error, 5), round(cosine, 5)))
                    acceptable = (error <= 0.12 and cosine >= 0.98) or maximum <= 1e-5
                    if acceptable:
                        self.fp32_pv, self.stages = True, 1
                self.checks.append(dict(shape=tuple(q.shape), dtype=str(q.dtype),
                    strides=tuple(tuple(value.stride()) for value in (q, k, v)),
                    sampled_heads=checked_heads, relative_l2=round(error, 5),
                    max_abs=round(maximum, 5), cosine=round(cosine, 5),
                    fp32_accumulation=self.fp32_pv, stages=self.stages, passed=acceptable,
                    attempts=attempts))
                self._probe_pending = False
                if not acceptable:
                    raise base.BackendUnavailable(
                        f'SM80+ VEDA numerical check failed: relative-L2={error:.4f}, '
                        f'max-abs={maximum:.4g}, cosine={cosine:.4f}, shape={tuple(q.shape)}. '
                        'Sparse output rejected; use STAR7_VEDA_REFERENCE=1 for a same-mask reference test.')
                self._verified_shapes.add(shape)
                arithmetic = ' / FP32 accumulation' if self.fp32_pv else ''
                self.display = f'Triton INT8 QK / FP16 PV{arithmetic} ({self._label}) [stages={self.stages}]'
            return output

    def warmup_note(self) -> str:
        return 'compiling Triton kernels for this GPU (first run only)'


def create(info) -> base.Backend:
    if info.kind != 'cuda' or info.cc is None or info.cc < _MIN_CC:
        raise base.BackendUnavailable(
            'triton-int8 needs a CUDA GPU of SM80 or newer')
    try:
        _kernel()
    except ImportError as error:
        package = ('triton-windows' if sys.platform == 'win32'
                   else 'triton')
        raise base.BackendUnavailable(
            f'Triton is not installed ({error}); pip install {package}'
        ) from error
    return TritonInt8Backend(info.family.upper(), audit=True)
