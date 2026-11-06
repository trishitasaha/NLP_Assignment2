"""Task 3a: BERT-derived *static* word vectors.

For every word type w of the Task 1 vocabulary (lowercased train types, same order, so row i of
both matrices is the same word):

    v_w(layer) = mean over all train occurrences of w of the layer-`layer` BERT word vector

The word vectors are the cached Task 2 features (each = mean of its wordpieces, computed IN
CONTEXT, i.e. the whole sentence was fed to BERT). Averaging over all contexts removes the
context and leaves one vector per type, which is what Bommasani et al. (2020) call a pooled
static embedding (mean pooling).

The required run is layer l* (from outputs/task2/best_layer.json). By default all 13 layers are
pooled, because the Tier B layer sweep (extensions_task3.py) needs them; it is cheap (one sparse
matrix product per layer).

Usage (from the project root):
    python code/task3/pool_static.py               # all layers 0..12
    python code/task3/pool_static.py --layers 8    # only l*

Outputs -> outputs/task3/static/
    layer{L}.npy   float32 [V, 768]
    counts.npy     int64 [V]   number of train occurrences (contexts averaged) per type
    meta.json      vocab source, layers, sanity checks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.data import load_ewt  # noqa: E402
from common.features import load_layer, load_meta, load_offsets  # noqa: E402

STATIC_DIR = C.TASK3_OUT / "static"


def load_t1_vocab() -> tuple[list[str], dict[str, int]]:
    vocab = json.loads((C.TASK1_OUT / "vocab.json").read_text())
    return vocab, {w: i for i, w in enumerate(vocab)}


def pooling_matrix(train, w2i) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sparse [V, N_words] matrix with 1/count(w) at (w, every occurrence of w)."""
    ids = np.array([w2i[w] for s in train for w in s.lower], dtype=np.int64)
    counts = np.bincount(ids, minlength=len(w2i))
    assert (counts > 0).all(), "every T1 vocab type occurs in train"
    S = sp.csr_matrix((1.0 / counts[ids], (ids, np.arange(len(ids)))), shape=(len(w2i), len(ids)),
                      dtype=np.float64)
    return S, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(C.BERT_NUM_LAYERS + 1)))
    args = ap.parse_args()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    vocab, w2i = load_t1_vocab()
    train = load_ewt("train")
    meta = load_meta("train")
    offsets = load_offsets("train")
    assert meta["sent_ids"] == [s.sent_id for s in train], "feature order != EWT order"
    assert int(offsets[-1]) == sum(len(s) for s in train), "feature rows != train tokens"
    S, counts = pooling_matrix(train, w2i)
    np.save(STATIC_DIR / "counts.npy", counts)
    print(f"[pool] vocab={len(vocab)} train words={S.shape[1]} singletons={(counts == 1).sum()}", flush=True)

    w = w2i["good"]  # sanity check word: its pooled vector must equal the plain mean of its rows
    rows = np.flatnonzero(np.array([x == "good" for s in train for x in s.lower]))
    checks = {}
    for layer in args.layers:
        t0 = time.time()
        X = np.asarray(load_layer("train", layer, mmap=False), dtype=np.float32)
        V = np.asarray(S @ X.astype(np.float64), dtype=np.float32)  # float64 accumulation
        err = float(np.abs(V[w] - X[rows].astype(np.float64).mean(0)).max())
        assert np.isfinite(V).all() and err < 1e-4, f"pooling check failed (err={err})"
        np.save(STATIC_DIR / f"layer{layer}.npy", V)
        checks[layer] = dict(max_abs_err_check=err, mean_norm=float(np.linalg.norm(V, axis=1).mean()))
        print(f"[pool] layer {layer:2d}: [{V.shape[0]}, {V.shape[1]}] mean |v|={checks[layer]['mean_norm']:.2f} "
              f"check err={err:.1e} ({time.time() - t0:.1f} s)", flush=True)
        del X, V

    (STATIC_DIR / "meta.json").write_text(json.dumps(dict(
        vocab="outputs/task1/vocab.json (same row order as the T1 vectors)",
        pooling="mean over all train occurrences of the in-context word vector (word = mean of wordpieces)",
        model=C.BERT_MODEL, revision=C.BERT_REVISION, layers=args.layers, checks=checks), indent=1))
    print("[pool] done ->", STATIC_DIR)


if __name__ == "__main__":
    main()
