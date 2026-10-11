"""Bounded FP32 sparse references for numerical verification on real inputs."""
import torch

TILE = 128


def sampled_outputs(q, k, v, mask, valid_count, output, tiles=None):
    """Compare a few real query rows against the exact same sparse selection.

    No dense N-by-N score matrix is allocated. The largest temporary is one
    head's selected K/V rows, and scores have at most three query rows.
    """
    counts = valid_count.cpu().tolist()
    count = len(counts)
    if tiles is None:
        tiles = sorted({0, count // 3, 2 * count // 3, count - 1})
    got, expected = [], []
    for head in range(q.shape[1]):
        for tile in tiles:
            live = counts[tile]
            if not live:
                continue
            rows = torch.tensor([tile * TILE + row for row in sorted({0, live // 2, live - 1})],
                                device=q.device, dtype=torch.int64)
            kept = torch.nonzero(mask[head, tile] & (valid_count > 0)).flatten()
            offsets = torch.arange(TILE, device=q.device)
            indices = kept[:, None] * TILE + offsets
            real = offsets[None, :] < valid_count.index_select(0, kept)[:, None]
            indices = indices[real].long()
            if not indices.numel():
                reference = torch.zeros(len(rows), q.shape[-1], device=q.device)
            else:
                keys = k[:, head].index_select(0, indices).float()
                values = v[:, head].index_select(0, indices).float()
                scores = q[:, head].index_select(0, rows).float() @ keys.transpose(0, 1)
                probabilities = torch.softmax(scores * q.shape[-1] ** -0.5, dim=-1)
                reference = probabilities @ values
            expected.append(reference)
            got.append(output[:, head].index_select(0, rows).float())
    return torch.cat(got), torch.cat(expected)


def numerical_error(got, expected):
    difference = got - expected
    norm = expected.norm().clamp_min(1e-8)
    relative = difference.norm() / norm
    cosine = torch.nn.functional.cosine_similarity(got.flatten(), expected.flatten(), dim=0, eps=1e-8)
    return torch.stack((relative, difference.abs().max(), cosine)).cpu().tolist()
