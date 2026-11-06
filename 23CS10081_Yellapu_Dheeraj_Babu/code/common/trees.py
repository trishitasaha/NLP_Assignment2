"""Minimum spanning trees, UUAS and simple tree baselines (shared by T1, T2, T3).

UUAS follows Hewitt & Manning's reporter.py: an MST is built over the *predicted* distance
matrix and over the *gold* tree-distance matrix, both restricted to non-punctuation words,
and the undirected edge sets are compared. Gold and predicted trees both have m-1 edges,
so precision = recall = UUAS.
"""
from __future__ import annotations

import numpy as np

from .data import gold_distance_matrix

Edge = tuple[int, int]


def mst_edges(dist: np.ndarray) -> set[Edge]:
    """Prim's algorithm on a dense symmetric distance matrix. Returns undirected edges (i<j).

    Deterministic: starts at node 0, and ties are broken by the lowest index (np.argmin).
    """
    n = dist.shape[0]
    if n <= 1:
        return set()
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].astype(np.float64).copy()
    parent = np.zeros(n, dtype=int)
    edges = set()
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        v = int(np.argmin(cand))
        edges.add(tuple(sorted((int(parent[v]), v))))
        in_tree[v] = True
        closer = dist[v] < best
        best = np.where(closer, dist[v], best)
        parent = np.where(closer, v, parent)
    return edges


def _restrict(dist: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(keep)
    return dist[np.ix_(idx, idx)], idx


def mst_on_subset(dist: np.ndarray, keep: np.ndarray) -> set[Edge]:
    """MST over the words with keep=True, returned in ORIGINAL word indices."""
    sub, idx = _restrict(dist, keep)
    return {(int(idx[a]), int(idx[b])) for a, b in mst_edges(sub)}


def gold_mst_edges(heads: list[int], keep: np.ndarray) -> set[Edge]:
    """Gold tree among kept words: MST of the gold distance matrix (equals the gold edges
    whenever the removed punctuation tokens are leaves, which is almost always the case)."""
    return mst_on_subset(gold_distance_matrix(heads), keep)


def uuas_counts(pred_dist: np.ndarray, heads: list[int], keep: np.ndarray) -> tuple[int, int]:
    """(# correct undirected edges, # gold edges) for one sentence."""
    gold = gold_mst_edges(heads, keep)
    pred = mst_on_subset(pred_dist, keep)
    return len(gold & pred), len(gold)


def cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    v = np.asarray(vecs, dtype=np.float64)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return 1.0 - v @ v.T


# --------------------------------------------------------------------------- baselines
def linear_chain_dist(n: int) -> np.ndarray:
    """Distance |i-j|: its MST is the chain w1-w2-...-wn (Hewitt's LINEAR baseline)."""
    r = np.arange(n)
    return np.abs(r[:, None] - r[None, :]).astype(np.float64)


def random_tree_edges(n: int, rng: np.random.Generator) -> set[Edge]:
    """Uniformly random labelled tree on n nodes via a random Pruefer sequence."""
    if n <= 1:
        return set()
    if n == 2:
        return {(0, 1)}
    seq = rng.integers(0, n, size=n - 2).tolist()
    degree = [1] * n
    for x in seq:
        degree[x] += 1
    edges = set()
    for x in seq:
        leaf = next(i for i in range(n) if degree[i] == 1)
        edges.add(tuple(sorted((leaf, x))))
        degree[leaf] -= 1
        degree[x] -= 1
    u, v = [i for i in range(n) if degree[i] == 1]
    edges.add((u, v))
    return edges


def random_tree_uuas_counts(heads: list[int], keep: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    gold = gold_mst_edges(heads, keep)
    idx = np.flatnonzero(keep)
    pred = {(int(idx[a]), int(idx[b])) for a, b in random_tree_edges(len(idx), rng)}
    return len(gold & pred), len(gold)
