"""Batching of cached BERT word features + gold tree targets for the structural probes.

GoldData (per split, layer-independent): gold distance matrices, gold depths, lengths, offsets.
LayerFeatures (per split and layer): the [n_words, 768] matrix moved to the GPU once
(~0.3 GB for train in fp16), plus one zero row used as padding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import gold_depths, gold_distance_matrix, load_ewt  # noqa: E402
from common.features import load_layer, load_offsets  # noqa: E402


class GoldData:
    def __init__(self, split: str, device: torch.device):
        self.split = split
        self.sents = load_ewt(split)
        self.offsets = load_offsets(split)
        self.lengths = np.array([len(s) for s in self.sents])
        assert self.offsets[-1] == self.lengths.sum(), "features and EWT split are misaligned"
        self.device = device
        self.dist = [torch.from_numpy(gold_distance_matrix(s.heads)).to(device) for s in self.sents]
        self.depth = [torch.from_numpy(gold_depths(s.heads)).to(device) for s in self.sents]
        self.offsets_t = torch.as_tensor(self.offsets[:-1], device=device)
        self.lengths_t = torch.as_tensor(self.lengths, device=device)

    def __len__(self) -> int:
        return len(self.sents)

    def targets(self, idx: torch.Tensor):
        """Padded gold distances [B, L, L] and depths [B, L] (padding = -1), lengths [B]."""
        lens = self.lengths_t[idx]
        L = int(lens.max())
        B = len(idx)
        dist = torch.full((B, L, L), -1.0, device=self.device)
        dep = torch.full((B, L), -1.0, device=self.device)
        for b, s in enumerate(idx.tolist()):
            n = self.lengths[s]
            dist[b, :n, :n] = self.dist[s]
            dep[b, :n] = self.depth[s]
        return dist, dep, lens


class LayerFeatures:
    def __init__(self, gold: GoldData, layer: int):
        X = np.asarray(load_layer(gold.split, layer, mmap=True))
        X = np.concatenate([X, np.zeros((1, X.shape[1]), dtype=X.dtype)])  # padding row
        self.X = torch.from_numpy(X).to(gold.device)  # fp16 on GPU
        self.pad_row = X.shape[0] - 1
        self.gold = gold

    def inputs(self, idx: torch.Tensor) -> torch.Tensor:
        """[B, L, 768] float32 word vectors for the sentences in idx (zero-padded)."""
        lens = self.gold.lengths_t[idx]
        L = int(lens.max())
        ar = torch.arange(L, device=self.X.device)
        rows = self.gold.offsets_t[idx][:, None] + ar[None, :]
        rows = torch.where(ar[None, :] < lens[:, None], rows, torch.full_like(rows, self.pad_row))
        return self.X[rows].float()
