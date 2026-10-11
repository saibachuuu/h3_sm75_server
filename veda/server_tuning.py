"""h3_sm75_server 对 vendored veda 的调参适配层（唯一的偏离点，集中在此）。

红线 1（见 PROGRESS 交接文档第八节）：`veda/core|backends` 与上游
（minimax-h3-chunk-star7@e03ee50）逐字一致，要改先回上游。本文件因此用
**monkeypatch + 参数注入**的方式实现两个可调项，core/backends 不动：

  1) **tile_logits 精度**（`--veda-tile-precision fp32|fp16`，默认 fp32）
     上游 `core/predictor.py` 的三个 bmm 是 fp32。sm75 没有 fp32 tensor
     core，PyTorch 退回 MAGMA 仿算（profile 实证 ~236ms/kernel、14.2s/60
     次）；fp16 走 HMMA 快一个量级。打分只决定 tile 排序（top-k），下游
     INT8 稀疏核自带 ~1.3% 数值容忍（backends/base.py 的 tolerance），
     半精度对排序影响有限。**默认 fp32 = 上游行为**，fp16 是显式实验。

  2) **backend.chunk_share**（`--veda-chunk-share N`，默认 16 = 上游值）
     `engine._chunk_bytes()` = free // backend.chunk_share。engine.py 自己
     注释过：每 chunk ~60 个小 kernel，WDDM 的 launch latency 让 8 chunk
     每层比 2 chunk 慢 ~15ms。显存富余时调小（工作集变大、chunk 数变少）
     = 拿显存换 launch 次数。chunk_share=1 表示一次吃完剩余显存，慎用。

恢复原状：`set_tile_precision("fp32")` 会还原上游函数对象本身；
`chunk_share` 只是实例属性，重置为 16 即回上游行为。
"""
from __future__ import annotations

import math

_TILE_PRECISIONS = ("fp32", "fp16")
_ORIG_TILE_LOGITS = None
_ACTIVE_PRECISION = "fp32"          # 当前生效的打分精度


def _predictor_module():
    from veda.core import predictor
    return predictor


def _make_fp16_tile_logits():
    """tile_logits 的 fp16 版：输入 cast fp16 走 HMMA，输出回落 fp32。

    与上游 `core/predictor.py:tile_logits` 的差别只在 dtype：
      * 三个 bmm 的输入/权重 cast 成 fp16（权重 proj 加载时是 bf16）；
      * 残差项 `+ feats[..., :dim]` 与 logits 同步 fp16 运算；
      * 输出 /sqrt(d) 后回 fp32 —— select() 本来就会 `.float()`，
        保持输出 dtype 与上游一致，下游零感知。
    """
    import torch

    def tile_logits(feats_q, feats_k, proj_q, proj_k):
        dim = proj_q.shape[-1]
        fq = feats_q.to(torch.float16)
        fk = feats_k.to(torch.float16)
        pq = proj_q.to(torch.float16)
        pk = proj_k.to(torch.float16)
        q_hat = torch.bmm(fq, pq) + fq[..., :dim]
        k_hat = torch.bmm(fk, pk) + fk[..., :dim]
        logits = torch.bmm(q_hat, k_hat.transpose(1, 2))
        return (logits.float() / math.sqrt(dim))

    return tile_logits


def set_tile_precision(name: str, log=print) -> str:
    """切换打分精度并返回生效值。无效名称抛 ValueError。"""
    global _ORIG_TILE_LOGITS, _ACTIVE_PRECISION
    name = (name or "").strip().lower()
    if name not in _TILE_PRECISIONS:
        raise ValueError("--veda-tile-precision 须为 %s 之一，当前 %r"
                         % ("/".join(_TILE_PRECISIONS), name))
    module = _predictor_module()
    if _ORIG_TILE_LOGITS is None:            # 首次调用时记住上游原版
        _ORIG_TILE_LOGITS = module.tile_logits
    if name == "fp32":
        module.tile_logits = _ORIG_TILE_LOGITS
    else:
        module.tile_logits = _make_fp16_tile_logits()
    _ACTIVE_PRECISION = name
    log("[veda-tuning] tile_logits precision=%s（fp32=上游行为）" % name)
    return name


def set_chunk_share(backend, share: int, log=print) -> int:
    """设置后端 chunk_share（整除剩余显存得到 chunk_bytes）。无效抛 ValueError。"""
    if not isinstance(share, int) or share < 1:
        raise ValueError("--veda-chunk-share 须为 >=1 的整数，当前 %r" % (share,))
    if backend is None:
        return share
    old = getattr(backend, "chunk_share", None)
    backend.chunk_share = share             # 实例属性覆盖类属性，backends 零改动
    log("[veda-tuning] chunk_share=%s（上游默认 16；=free/%s MB 每 chunk）"
        % (share, share))
    del old
    return share


def configure(tile_precision: str = "fp32", chunk_share: int = 16,
              backend=None, log=print) -> dict:
    """一站式应用两个调参项，返回生效配置 dict。"""
    precision = set_tile_precision(tile_precision, log=log)
    share = set_chunk_share(backend, chunk_share, log=log)
    return {"tile_precision": precision, "chunk_share": share}


def state() -> dict:
    """当前生效配置（/api/status 可读）。"""
    module = _predictor_module()
    patched = module.tile_logits is not _ORIG_TILE_LOGITS
    return {"tile_precision": _ACTIVE_PRECISION, "patched": patched}
