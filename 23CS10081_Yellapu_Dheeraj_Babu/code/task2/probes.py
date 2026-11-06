"""Structural probes and losses (Hewitt & Manning 2019; mirrors structural-probes/probe.py + loss.py).

Distance probe (TwoWordPSDProbe):  d_B(h_i, h_j)^2 = || B (h_i - h_j) ||^2  ~  tree distance d(w_i, w_j)
Depth probe    (OneWordPSDProbe):  || B h_i ||^2                         ~  tree depth ||w_i||

B is stored as `proj` with shape [768, k] (repo convention: transformed = h @ proj), initialised
U(-0.05, 0.05). The SQUARED distance / norm is what is trained to match the gold value (paper
App. A.1).

Losses (paper App. A.2, repo loss.py): L1 between prediction and gold, padded positions masked.
  distance: per sentence  sum_{i,j} |pred - gold| / n^2
  depth:    per sentence  sum_i     |pred - gold| / n
  then averaged over the sentences in the batch.
"""
from __future__ import annotations

import torch
from torch import nn


class DistanceProbe(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(dim, rank).uniform_(-0.05, 0.05))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, dim] -> predicted squared distances [B, L, L]."""
        t = h @ self.proj  # [B, L, k]
        # ||t_i - t_j||^2 = ||t_i||^2 + ||t_j||^2 - 2 t_i.t_j : same value as the explicit
        # difference, but O(B L^2) memory instead of O(B L^2 k). Clamp removes tiny negative
        # round-off on the diagonal.
        sq = (t * t).sum(-1)  # [B, L]
        d = sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * (t @ t.transpose(1, 2))
        return d.clamp_min(0.0)


class DepthProbe(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(dim, rank).uniform_(-0.05, 0.05))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, dim] -> predicted squared norms (depths) [B, L]."""
        t = h @ self.proj
        return (t * t).sum(-1)


def distance_loss(pred: torch.Tensor, gold: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """pred/gold: [B, L, L]; gold = -1 on padding. Returns the batch-mean of per-sentence losses."""
    mask = (gold >= 0).float()
    per_sent = (torch.abs(pred - gold) * mask).sum((1, 2)) / lengths.float().pow(2)
    return per_sent.mean()


def depth_loss(pred: torch.Tensor, gold: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """pred/gold: [B, L]; gold = -1 on padding."""
    mask = (gold >= 0).float()
    per_sent = (torch.abs(pred - gold) * mask).sum(1) / lengths.float()
    return per_sent.mean()


PROBES = {"distance": (DistanceProbe, distance_loss), "depth": (DepthProbe, depth_loss)}
