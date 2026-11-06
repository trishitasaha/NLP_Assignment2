"""Helpers shared by the Task 3 scripts.

* load_t1 / load_t3_static : the two static vector sets (same vocabulary and row order)
* abtt                     : "all-but-the-top" post-processing (Mu & Viswanath, ICLR 2018)
* anisotropy               : mean cosine between random word-type pairs (Ethayarajh 2019)
* ContextualTest           : raw in-context BERT word vectors for test sentences (Task 2 cache)
* probe_matrix / probe_sq_dist : the trained Task 2 distance probe B, applied to any vectors
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.data import Sentence  # noqa: E402
from common.features import load_layer, load_meta, load_offsets  # noqa: E402

STATIC_DIR = C.TASK3_OUT / "static"


def load_w2i() -> dict[str, int]:
    vocab = json.loads((C.TASK1_OUT / "vocab.json").read_text())
    return {w: i for i, w in enumerate(vocab)}


def load_t1(tag: str | None = None) -> np.ndarray:
    """T1 PPMI-SVD vectors: the spec run (tag=None) or an ablation (e.g. 'p0')."""
    d = C.TASK1_OUT if tag is None else C.TASK1_OUT / "ablations" / tag
    return np.load(d / "vectors.npy").astype(np.float32)


def t1_dir(tag: str | None = None) -> Path:
    return C.TASK1_OUT if tag is None else C.TASK1_OUT / "ablations" / tag


def load_t3_static(layer: int) -> np.ndarray:
    return np.load(STATIC_DIR / f"layer{layer}.npy").astype(np.float32)


def best_layer() -> dict:
    return json.loads((C.TASK2_OUT / "best_layer.json").read_text())


# --------------------------------------------------------------------------- post-processing
def abtt(V: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """All-but-the-top: subtract the mean vector, then remove the projections on the top-D
    principal directions. D = d/100 by default (the paper's rule of thumb: 7 for BERT's 768
    dims, 1 for the 100-d SVD vectors). Zero rows (words with no context) stay zero."""
    V = np.asarray(V, dtype=np.float64)
    D = max(1, V.shape[1] // 100) if n_components is None else n_components
    nz = np.linalg.norm(V, axis=1) > 0
    mu = V[nz].mean(0)
    Xc = V - mu
    # top-D right singular vectors of the centred (non-zero) rows = top-D principal directions
    _, _, Vt = np.linalg.svd(Xc[nz], full_matrices=False)
    U = Vt[:D].T  # [d, D]
    out = Xc - (Xc @ U) @ U.T
    out[~nz] = 0.0
    return out.astype(np.float32)


def anisotropy(V: np.ndarray, n_pairs: int = 20000, seed: int = C.SEED) -> float:
    """Expected cosine similarity of two random (distinct, non-zero) word types.
    ~0 for isotropic vectors; large positive = all vectors in a narrow cone."""
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    idx = np.flatnonzero(np.linalg.norm(V, axis=1) > 0)
    a, b = rng.choice(idx, n_pairs), rng.choice(idx, n_pairs)
    keep = a != b
    U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return float((U[a[keep]] * U[b[keep]]).sum(1).mean())


def l2_normalise(V: np.ndarray) -> np.ndarray:
    return (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------- contextual (test split)
class ContextualTest:
    """In-context word vectors of test sentences, from the Task 2 feature cache."""

    def __init__(self):
        self.offsets = load_offsets("test")
        self.row = {sid: i for i, sid in enumerate(load_meta("test")["sent_ids"])}
        self._layers: dict[int, np.ndarray] = {}

    def vectors(self, s: Sentence, layer: int) -> np.ndarray:
        if layer not in self._layers:
            self._layers[layer] = load_layer("test", layer, mmap=True)
        i = self.row[s.sent_id]
        h = np.asarray(self._layers[layer][self.offsets[i]:self.offsets[i + 1]], dtype=np.float64)
        assert len(h) == len(s), f"feature/token mismatch in {s.sent_id}"
        return h


# --------------------------------------------------------------------------- Task 2 probe
def probe_matrix(layer: int, rank: int, probe_dir: Path | None = None) -> np.ndarray:
    """B [768, k] of the trained distance probe."""
    probe_dir = probe_dir or (C.TASK2_OUT / "probes")
    ck = torch.load(probe_dir / f"L{layer}_distance_k{rank}.pt", map_location="cpu")
    return ck["state_dict"]["proj"].double().numpy()


def probe_sq_dist(h: np.ndarray, B: np.ndarray) -> np.ndarray:
    """||B^T (h_i - h_j)||^2 for all pairs (the probe's predicted tree distance)."""
    t = h @ B
    sq = (t * t).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * t @ t.T, 0.0)


def euclidean_dist(h: np.ndarray) -> np.ndarray:
    sq = (h * h).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * h @ h.T, 0.0))
