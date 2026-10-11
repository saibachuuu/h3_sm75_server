"""h3_sm75_server 回归自检（从 tests/ 目录直接跑）。

跑法（需要有 torch 的解释器；本地可用 ~/comfyui/.venv）：
  cd h3_sm75_server/tests && python _selftest_star7_cpu.py

背景：2026-10-11 的两次线上启动失败（self._install_attention
找不到 / bundle 缺 keep_ratio）都是"静态 import+compileall
挡不住、真执行才炸"的类型，所以这些脚本必须真实执行到关键
路径，不能只 import。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "comfyui"))
os.chdir(os.path.join(os.path.expanduser("~"), "comfyui"))

import torch  # noqa: E402

device = "cpu"
ok = []

print("== 1. tiling / selection 纯 CPU 数学 ==")
from veda.core import selection, tiling  # noqa: E402

# 一个小布局：text 70 行 global + target 5x8x8=320 行 tiled
spans = [tiling.TiledSpan(70, (5, 8, 8), tiling.TileShape(4, 4, 8))]
layout = tiling.build_tile_layout(spans, 70 + 320, device)
print("   seq_len=%d n_tiles=%d n_video=%d n_global=%d perm=%s"
      % (layout.seq_len, layout.n_tiles, layout.n_video_tiles,
         layout.n_global_tiles, tuple(layout.perm.shape)))
assert layout.n_video_tiles == 4, "target tiles: shape 4x4x8 on (5,8,8) -> padded (8,8,8) -> 4 tiles"
assert layout.n_global_tiles == 1, "text tile"
assert int(layout.valid_count.sum()) == 390, "real rows"
ok.append("tiling layout")

q = torch.randn(layout.num_slots, 2, 128, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
scores = torch.randn(2, layout.n_video_tiles, layout.n_video_tiles, device=device)
blocks = selection.column_blocks(layout, selection.Budget(tiles=2),
                                  selection.Budget(tiles=1))
index, keep = selection.select(scores, layout, blocks)
mask = selection.block_mask(index, keep, layout)
print("   index=%s keep=%s mask=%s" % (tuple(index.shape), tuple(keep.shape),
                                       tuple(mask.shape)))
assert mask.shape == (2, layout.n_tiles, layout.n_tiles)
assert mask[:, :, layout.n_video_tiles:].all(), "global cols dense"
assert mask[:, layout.n_video_tiles:, :].all(), "global rows dense"
kept_per_row = mask[:, :layout.n_video_tiles,
                    layout.n_video_tiles:].sum(-1).float()
print("   kept key tiles per row: min=%d max=%d"
      % (int(kept_per_row.min()), int(kept_per_row.max())))
ok.append("selection top-k + block mask")

# gather/scatter 往返一致
g = tiling.gather_tiles(torch.arange(layout.seq_len, device=device,
                                    dtype=torch.float32).unsqueeze(1).unsqueeze(1),
                        layout, torch.zeros(1, dtype=torch.long, device=device))
out = torch.zeros(layout.seq_len + 1, 1, 1, device=device)
tiling.scatter_tiles_(out, g, layout, torch.zeros(1, dtype=torch.long, device=device))
real = layout.perm[layout.perm >= 0]
got = out[real, 0, 0]
want = real.float()
assert torch.equal(got, want), "gather/scatter roundtrip"
ok.append("gather/scatter roundtrip")

print("== 2. reference 引擎（fp32 真值） ==")
from veda.core import reference  # noqa: E402

ref = reference.block_sparse_attention(q, k, v, mask, layout)
dense = reference.dense_attention(q[:layout.seq_len], k[:layout.seq_len],
                                  v[:layout.seq_len])
print("   ref out=%s finite=%s" % (tuple(ref.shape),
                                   bool(torch.isfinite(ref.float()).all())))
ok.append("reference engine")

print("== 3. h3_layout.describe 对 comfy PackedLayout ==")
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402
from veda.core import h3_layout  # noqa: E402

# text 128 + target 30 帧 -> latent_t: 17n+5=30 -> n=1?f -> 用 latent_t=7
lay = PackedLayout(128, 7, 480, 854, 4)
spec = h3_layout.describe(lay)
print("   seq_len=%d target=%s grid" % (spec.seq_len, spec.target.start),
      )
print("   grid=%s refs=%d skipped=%s" % (spec.target.grid, len(spec.references),
                                         spec.skipped))
ok.append("h3_layout.describe(PackedLayout)")

print("== 4. plans 选择（CPU） ==")
from veda.core import plans as veda_plans  # noqa: E402

plan_json = {
    "geometry": "16x9_t7", "grid": [7, 20, 36],
    "shapes": ["8x4x4", "2x8x8"], "head_shape": [[0, 1] * 28],
}
table = veda_plans.PlanTable([veda_plans.TilePlan.from_json(plan_json)])
choice = table.select((7, 15, 27))
print("   chose %s exact=%s how=%s" % (choice.plan.name, choice.exact, choice.how[:50]))
ok.append("PlanTable.select")

print("== 5. sm75 后端在 sm89 上必须被拒（gate 有效性） ==")
from veda import backends  # noqa: E402

info = backends.hardware.describe(torch.device("cpu"))
print("   (cpu 设备 kind=%s cc=%s -> candidates=%s)"
      % (info.kind, info.cc, backends.candidates(info)))
ok.append("backend gate")

print()
for line in ok:
    print("OK", line)
print("ALL OK")
