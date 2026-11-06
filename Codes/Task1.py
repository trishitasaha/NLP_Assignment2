"""
Task 1: PPMI + truncated-SVD static word vectors on UD English EWT (train split),
evaluated on WordSim-353, Google analogies, and the 20-sentence MST structural check.

Run:   python task1_ppmi_svd.py
Options (all optional):
   --min_count 2      ignore words seen fewer than this many times in train
   --window 5         symmetric window size (assignment: 5)
   --dim 100          vector size (assignment: 100)
   --svd_power 1.0    vectors = U * S**power  (1.0 = plain truncated SVD; 0.5 is another common choice)
   --wordsim PATH     use your instructor's WordSim file instead of the downloaded one
   --analogy PATH     use your instructor's analogy file instead of the downloaded one
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

import common


# ---------------------------------------------------------------- step 1
def build_vocab(sents, min_count):
    """Lowercase words -> ids, keeping words seen at least min_count times."""
    cnt = Counter(t.lower() for s in sents for t in s["tokens"])
    words = [w for w, c in cnt.most_common() if c >= min_count]
    return {w: i for i, w in enumerate(words)}, cnt


def build_cooccurrence(sents, word2id, window):
    """Symmetric word-word co-occurrence counts (window words to each side).
    Each time words a and b are d<=window positions apart, we add 1 to C[a,b] and C[b,a]."""
    V = len(word2id)
    rows, cols = [], []
    for s in sents:
        ids = np.array([word2id.get(t.lower(), -1) for t in s["tokens"]])
        n = len(ids)
        for d in range(1, min(window, n - 1) + 1):
            a, b = ids[:-d], ids[d:]
            keep = (a >= 0) & (b >= 0)          # skip pairs involving rare/unknown words
            a, b = a[keep], b[keep]
            rows += [a, b]
            cols += [b, a]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float64)
    # duplicate (row, col) entries are summed automatically when converting to CSR
    return sparse.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()


# ---------------------------------------------------------------- step 2
def ppmi(C, alpha=1.0):
    """PPMI(w,c) = max(0, log [ P(w,c) / (P(w) P(c)) ]).
    alpha=1.0 -> plain PPMI. (alpha=0.75 would be 'context smoothing'; not required here.)"""
    C = C.tocoo()
    total = C.sum()
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_cnt = np.asarray(C.sum(axis=0)).ravel() ** alpha
    p_wc = C.data / total
    p_w = row_sum[C.row] / total
    p_c = col_cnt[C.col] / col_cnt.sum()
    pmi = np.log(p_wc / (p_w * p_c))
    pmi = np.maximum(pmi, 0.0)                  # the "positive" in PPMI
    M = sparse.coo_matrix((pmi, (C.row, C.col)), shape=C.shape).tocsr()
    M.eliminate_zeros()
    return M


# ---------------------------------------------------------------- step 3
def truncated_svd_vectors(M, dim, power=1.0, seed=0):
    """Keep the top `dim` singular directions. Word vectors = U * S**power."""
    v0 = np.random.RandomState(seed).rand(min(M.shape))     # fixed start -> reproducible
    U, S, _ = svds(M.astype(np.float64), k=dim, v0=v0)
    order = np.argsort(-S)                                  # svds returns ascending order
    U, S = U[:, order], S[order]
    return U * (S ** power)


def nearest(word, word2id, id2word, vecs, topn=5):
    Wn = common.normalize(vecs)
    sims = Wn @ Wn[word2id[word]]
    sims[word2id[word]] = -np.inf
    return [id2word[i] for i in np.argsort(-sims)[:topn]]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--svd_power", type=float, default=1.0)
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "conllu"])
    ap.add_argument("--wordsim", default=None)
    ap.add_argument("--analogy", default=None)
    ap.add_argument("--n_sample", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(common.RESULTS_DIR, exist_ok=True)

    # ---- data
    ewt = common.load_ewt(args.source)
    train, test = ewt["train"], ewt["test"]
    print(f"Sentences  train={len(train)}  dev={len(ewt['dev'])}  test={len(test)}")
    print("Example:", train[0]["tokens"][:12], train[0]["head"][:12])

    # ---- build vectors (train split only, raw words only)
    word2id, cnt = build_vocab(train, args.min_count)
    id2word = {i: w for w, i in word2id.items()}
    n_tokens = sum(cnt.values())
    print(f"Train tokens={n_tokens}  word types={len(cnt)}  vocab (count>={args.min_count})={len(word2id)}")

    C = build_cooccurrence(train, word2id, args.window)
    print(f"Co-occurrence matrix {C.shape}, non-zeros={C.nnz}, total count={int(C.sum())}")
    M = ppmi(C)
    print(f"PPMI matrix non-zeros={M.nnz}")
    vecs = truncated_svd_vectors(M, args.dim, args.svd_power)
    print(f"Word vectors: {vecs.shape}   ({time.time() - t0:.0f}s so far)")

    # quick sanity check: neighbours should look sensible
    for w in ["good", "people", "monday", "computer", "said"]:
        if w in word2id:
            print(f"  nearest to '{w}':", nearest(w, word2id, id2word, vecs))

    # ---- evaluation 1: WordSim-353
    pairs = common.load_wordsim(args.wordsim)
    ws = common.eval_wordsim(pairs, word2id, vecs)
    print(f"\nWordSim-353: {ws['n_covered']}/{ws['n_pairs']} pairs have both words in vocab")
    print(f"  Spearman (covered pairs only) = {ws['spearman_covered']:.4f}")
    print(f"  Spearman (all 353, OOV pairs -> sim 0) = {ws['spearman_all353']:.4f}")

    # ---- evaluation 2: Google analogies
    qs = common.load_analogy(args.analogy)
    an = common.eval_analogy(qs, word2id, vecs)
    print(f"\nGoogle analogies: {an['n_covered']}/{an['n_questions']} questions fully in vocab")
    print(f"  Accuracy (covered questions) = {an['acc_covered']:.4f}")
    print(f"  Accuracy (all questions, OOV = wrong) = {an['acc_all']:.4f}")
    print(f"  semantic = {an['acc_semantic_covered']:.4f}   syntactic = {an['acc_syntactic_covered']:.4f}")

    # ---- evaluation 3: bonus MST structural check on 20 test sentences
    sample, idx = common.sample_test_sentences(test, n=args.n_sample)
    mst = common.eval_mst_agreement(sample, word2id, vecs)
    base = common.baseline_agreement(sample)
    print(f"\nMST structural check on {mst['n_sentences']} test sentences (indices {idx})")
    print(f"  Undirected attachment agreement (micro) = {mst['agreement_micro']:.4f}")
    print(f"  Undirected attachment agreement (macro) = {mst['agreement_macro']:.4f}")
    print(f"  Words without a vector: {100 * mst['avg_fraction_words_without_vector']:.1f}% on average")
    print(f"  Reference: random tree = {base['random_tree']:.4f}, linear chain = {base['linear_chain']:.4f}")

    # ---- save everything for the report and for Task 3
    results = {"settings": vars(args), "vocab_size": len(word2id), "wordsim": ws,
               "analogy": an, "mst": mst, "baselines": base, "sample_indices": idx}
    with open(os.path.join(common.RESULTS_DIR, "task1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(common.RESULTS_DIR, "task1_vectors.npz"),
                        vectors=vecs, words=np.array([id2word[i] for i in range(len(id2word))]))
    print(f"\nSaved results/task1_results.json and results/task1_vectors.npz  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
