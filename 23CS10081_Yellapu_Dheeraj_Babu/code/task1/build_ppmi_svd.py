"""Task 1, steps 1-3: co-occurrence matrix -> PPMI -> truncated SVD (100-d static vectors).

Usage (from the project root):
    python code/task1/build_ppmi_svd.py                      # spec setting (writes outputs/task1/)
    python code/task1/build_ppmi_svd.py --cds-alpha 0.75 --eig-power 0.5 --tag cds075_p05
                                                             # (Tier C) ablation -> outputs/task1/ablations/<tag>/

Spec setting: EWT train split, raw words only (lowercased, decision D3; punctuation tokens
are kept as ordinary words), symmetric window +-5 inside sentence boundaries, unweighted
counts, plain PPMI, TruncatedSVD(100, random_state=42), vectors = U * Sigma.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.data import load_ewt  # noqa: E402


def build_vocab(sentences, min_count: int) -> tuple[list[str], Counter]:
    counts = Counter(w for s in sentences for w in s.lower)
    # deterministic order: frequency desc, then alphabetical
    vocab = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
    return vocab, counts


def cooccurrence_matrix(sentences, w2i: dict[str, int], window: int) -> sp.csr_matrix:
    """Symmetric word-word counts: C[w, c] = # times c occurs within +-window of w (same sentence).

    Vectorised: concatenate all sentences into one id stream with a parallel sentence-id stream,
    and for each offset d = 1..window pair position k with k+d when both are in the same sentence.
    Out-of-vocab words (only when min_count > 1) are dropped before windowing, as in word2vec.
    """
    ids, sid = [], []
    for n, s in enumerate(sentences):
        toks = [w2i[w] for w in s.lower if w in w2i]
        ids.extend(toks)
        sid.extend([n] * len(toks))
    ids, sid = np.asarray(ids, dtype=np.int64), np.asarray(sid, dtype=np.int64)
    V = len(w2i)
    rows, cols = [], []
    for d in range(1, window + 1):
        same = sid[:-d] == sid[d:]
        a, b = ids[:-d][same], ids[d:][same]
        rows += [a, b]  # both directions -> symmetric
        cols += [b, a]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = sp.coo_matrix((np.ones_like(rows, dtype=np.float64), (rows, cols)), shape=(V, V)).tocsr()
    M.sum_duplicates()
    return M


def ppmi_matrix(M: sp.csr_matrix, cds_alpha: float = 1.0) -> sp.csr_matrix:
    """PPMI(w, c) = max(0, log [P(w, c) / (P(w) P_alpha(c))]).

    cds_alpha = 1 is plain PPMI (spec). cds_alpha = 0.75 is the context-distribution smoothing
    of Levy, Goldberg & Dagan (2015), only used in the Tier C ablation.
    Computed on the non-zero entries only (zero counts give PMI = -inf, hence PPMI = 0).
    """
    M = M.tocoo()
    total = M.sum()
    row = np.asarray(M.sum(axis=1)).ravel()
    col = np.asarray(M.sum(axis=0)).ravel()
    p_w = row / total
    col_a = col ** cds_alpha
    p_c = col_a / col_a.sum()
    pmi = np.log((M.data / total) / (p_w[M.row] * p_c[M.col]))
    keep = pmi > 0
    P = sp.coo_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape).tocsr()
    return P


def svd_vectors(P: sp.csr_matrix, dim: int, eig_power: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD P ~ U S V^T; word vectors = U * S^eig_power (spec: eig_power = 1)."""
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=10, random_state=seed)
    US = svd.fit_transform(P)  # = U * S
    S = svd.singular_values_
    U = US / S
    return (U * S ** eig_power).astype(np.float32), S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=C.WINDOW)
    ap.add_argument("--dim", type=int, default=C.SVD_DIM)
    ap.add_argument("--min-count", type=int, default=C.MIN_COUNT)
    ap.add_argument("--cds-alpha", type=float, default=1.0)
    ap.add_argument("--eig-power", type=float, default=1.0)
    ap.add_argument("--tag", default=None, help="ablation name -> outputs/task1/ablations/<tag>/")
    args = ap.parse_args()
    C.set_seed()
    out = C.TASK1_OUT / "ablations" / args.tag if args.tag else C.TASK1_OUT
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train = load_ewt("train")
    vocab, counts = build_vocab(train, args.min_count)
    w2i = {w: i for i, w in enumerate(vocab)}
    print(f"[task1] {len(train)} train sentences, vocab = {len(vocab)} types (min_count={args.min_count})")

    M = cooccurrence_matrix(train, w2i, args.window)
    print(f"[task1] co-occurrence: shape={M.shape} nnz={M.nnz} total count={M.sum():.0f} "
          f"symmetric={abs(M - M.T).sum() == 0}")
    P = ppmi_matrix(M, args.cds_alpha)
    print(f"[task1] PPMI: nnz={P.nnz} ({P.nnz / M.nnz:.1%} of co-occurring pairs have PMI > 0), "
          f"max={P.max():.2f}")
    vecs, S = svd_vectors(P, args.dim, args.eig_power, C.SEED)
    zero_rows = int((np.linalg.norm(vecs, axis=1) == 0).sum())
    print(f"[task1] SVD: vectors {vecs.shape}; top-5 singular values {np.round(S[:5], 2).tolist()}; "
          f"zero vectors = {zero_rows}")

    np.save(out / "vectors.npy", vecs)
    (out / "vocab.json").write_text(json.dumps(vocab))
    meta = dict(window=args.window, dim=args.dim, min_count=args.min_count, cds_alpha=args.cds_alpha,
                eig_power=args.eig_power, seed=C.SEED, vocab_size=len(vocab),
                cooc_nnz=int(M.nnz), cooc_total=float(M.sum()), ppmi_nnz=int(P.nnz),
                singular_values=S.tolist(), seconds=round(time.time() - t0, 1),
                word_counts={w: counts[w] for w in vocab})
    (out / "build_meta.json").write_text(json.dumps(meta))
    print(f"[task1] saved vectors + vocab to {out}  ({meta['seconds']} s)")


if __name__ == "__main__":
    main()
