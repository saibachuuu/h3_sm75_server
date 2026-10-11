# Third-party notices

* **SageAttention, 1.0.6** — `veda_comfy/kernels/sage/sparse_int8.py` is
  derived from `quant_per_block.py` and `attn_qk_int8_per_block.py` of the
  PyPI package `sageattention==1.0.6` (BSD-3-Clause, Thu-ML). Its INT8
  quantisation, scale folding and softmax are kept deliberately: the same
  code is what ComfyUI runs behind `--use-sage-attention`, so keeping the
  arithmetic keeps the accuracy. The block-sparse key walk, the padding
  mask and the host-side plumbing are ours; the module's docstring lists
  the changes.
* **Miowtion** (<https://github.com/veda-sparse/Miowtion>, MIT) — the Veda
  tiling, plan, predictor and selection rules in `veda_comfy/core` are a
  rewrite of `miowtion/veda`.
* **Veda predictor weights** (downloaded at run time, not shipped here) —
  MiniMax H3 Community License, inherited from MiniMax-H3.
* **Icon** — the Miowtion logo (`assets/icon.svg`), MIT.

* **SageAttention / local Star7 Turing adaptation** — `kernels/native` uses the Apache-2.0 INT8 Tensor Core online-softmax implementation, adapted to accept VEDA route words and per-token padding validity. Quantization runs once per head group. License: `kernels/native/LICENSE.Apache-2.0`.

* **Star7 CUDA C ABI bridge** — the SM75 wrapper and ctypes loader use public tensor addresses and CUDA streams, without ATen or CPython bindings. The Windows library statically links the CUDA runtime; a compatible NVIDIA driver remains required. See `kernels/native/README.md` for build and validation scope.
