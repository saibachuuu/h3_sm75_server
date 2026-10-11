"""VEDA sparse attention for MiniMax-H3, vendored from minimax-h3-chunk-star7.

Layout (unchanged from the source repo):
  core/      device-agnostic torch: tiling, plans, predictor, selection,
             the per-call engine (no ComfyUI imports)
  backends/  one module per kernel family, isolated from each other
  kernels/native/  the SM75 CUDA library through a stable C ABI (ctypes)
  comfy_patch.py / settings.py / status.py / hardware.py  the ComfyUI side

Not vendored (ComfyUI-node only, not needed by the headless server):
  nodes.py (comfy_api.latest io.ComfyNode), predictors.py (node dropdown)
"""

from . import backends
from . import comfy_patch
from . import hardware
from . import settings
from . import status

__version__ = '0.2.0-sm75server'

__all__ = ['backends', 'comfy_patch', 'hardware', 'settings', 'status']
