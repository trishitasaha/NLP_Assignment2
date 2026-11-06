"""Task 3 diagnostics: turn the three explanations of "T3 wins on meaning, loses on trees" into
measured numbers (plan.md Phase 4 caveat 1).

A. HUBNESS (Radovanovic et al., JMLR 2010). N_10(w) = how often w appears in the 10-nearest-
   neighbour lists of the other words. A long right tail (high skewness) means a few "hub"
   words are close to everything. Reported: skewness of N_10, the top hubs and how many of the
   top-50 hubs are function words. Vocabulary: word types with train frequency >= 5 (rare-word
   vectors are noise for both models).

B. HUBS IN THE MSTs (660-sentence pool). Mean degree of function-word tokens in the predicted
   trees vs the gold trees, and the largest node degree per sentence ("star" trees).

C. PARADIGMATIC vs SYNTAGMATIC (Sahlgren 2006).
   C1 nearest neighbours: share of each word's top-10 neighbours with the same majority UPOS
      (paradigmatic: substitutes share a category), and share that actually co-occur with it
      within +-5 words in train (syntagmatic: partners), each vs a random-pair baseline.
   C2 trees: P(same UPOS) for gold edges vs non-edge word pairs in the same sentence, and for the
      edges each representation predicts. If gold edges mostly join *different* categories,
      a similarity that prefers same-category pairs is the wrong signal for trees.
   C3 edge AUC: within each sentence, the probability that a gold edge pair is closer than a
      non-edge pair (0.5 = no information). Measures "does closeness predict attachment",
      independently of the MST algorithm.

Usage (from the project root, after evaluate_task3.py):
    python code/task3/diagnostics_task3.py
Outputs -> outputs/task3/diagnostics/: diagnostics.json, hubs.csv, edge_stats.csv
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task1"))
from common import config as C  # noqa: E402
from common.data import load_ewt, word_counts  # noqa: E402
from common.static_eval import _sentence_vectors, get_mst_sentences  # noqa: E402
from common.trees import cosine_distance_matrix, gold_mst_edges, linear_chain_dist, mst_on_subset  # noqa: E402

from build_ppmi_svd import cooccurrence_matrix  # noqa: E402
from t3_common import (ContextualTest, abtt, best_layer, load_t1, load_t3_static, load_w2i,  # noqa: E402
                       probe_matrix, probe_sq_dist)

OUT = C.TASK3_OUT / "diagnostics"
FUNCTION_UPOS = {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART"}
K = 10
MIN_FREQ_HUB = 5
MIN_FREQ_NN = 10


def majority_upos(train) -> dict[str, str]:
    c = defaultdict(Counter)
    for s in train:
        for w, u in zip(s.lower, s.upos):
            c[w][u] += 1
    return {w: cnt.most_common(1)[0][0] for w, cnt in c.items()}


def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def knn(U: np.ndarray, k: int = K, chunk: int = 2048) -> np.ndarray:
    """Indices of the k nearest neighbours (cosine) of every row, excluding itself."""
    out = np.empty((len(U), k), dtype=np.int64)
    for s in range(0, len(U), chunk):
        S = U[s:s + chunk] @ U.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        out[s:s + chunk] = np.take_along_axis(idx, order, 1)
    return out


# --------------------------------------------------------------------------- A + C1
def lexical_diagnostics(name, V, w2i, counts, upos, M, rng) -> tuple[dict, pd.DataFrame]:
    i2w = {i: w for w, i in w2i.items()}
    # A. hubness over frequent-enough types
    hub_ids = np.array(sorted(i for w, i in w2i.items() if counts[w] >= MIN_FREQ_HUB and np.any(V[i])))
    nn = knn(unit(V[hub_ids].astype(np.float64)))
    Nk = np.bincount(nn.ravel(), minlength=len(hub_ids))
    top = np.argsort(-Nk)[:50]
    hubs = pd.DataFrame(dict(system=name, rank=np.arange(1, 51), word=[i2w[int(hub_ids[t])] for t in top],
                             N10=Nk[top], upos=[upos.get(i2w[int(hub_ids[t])], "?") for t in top]))
    frac_fn_top50 = float(hubs.upos.isin(FUNCTION_UPOS).mean())
    # C1. neighbour category agreement and co-occurrence, among types with freq >= MIN_FREQ_NN
    ids = np.array(sorted(i for w, i in w2i.items() if counts[w] >= MIN_FREQ_NN and np.any(V[i])))
    nn = ids[knn(unit(V[ids].astype(np.float64)))]
    up = np.array([upos[i2w[int(i)]] for i in ids])
    up_of = dict(zip(ids.tolist(), up.tolist()))
    same = np.array([[up_of[int(j)] == up_of[int(q)] for j in row] for q, row in zip(ids, nn)])
    Mb = (M[ids][:, ids] > 0).tocsr()
    pos = {int(i): p for p, i in enumerate(ids)}
    cooc = np.array([[Mb[pos[int(q)], pos[int(j)]] for j in row] for q, row in zip(ids, nn)])
    ra, rb = rng.choice(len(ids), 200000), rng.choice(len(ids), 200000)
    keep = ra != rb
    ra, rb = ra[keep], rb[keep]
    base_same = float((up[ra] == up[rb]).mean())
    base_cooc = float(np.asarray(Mb[ra, rb]).ravel().mean())
    res = dict(system=name, n_hub_vocab=int(len(hub_ids)), N10_skewness=float(skew(Nk)), N10_max=int(Nk.max()),
               frac_function_words_top50_hubs=frac_fn_top50, top10_hubs=hubs.word.head(10).tolist(),
               n_nn_vocab=int(len(ids)), nn_same_upos=float(same.mean()), nn_cooccur=float(cooc.mean()),
               random_same_upos=base_same, random_cooccur=base_cooc)
    print(f"   {name:16s} N10 skew={res['N10_skewness']:.2f} max={res['N10_max']} fn-hubs(top50)={frac_fn_top50:.2f} "
          f"top hubs={res['top10_hubs'][:6]} | NN same-UPOS={res['nn_same_upos']:.3f} (random {base_same:.3f}) "
          f"NN co-occur={res['nn_cooccur']:.3f} (random {base_cooc:.3f})", flush=True)
    return res, hubs


# --------------------------------------------------------------------------- B + C2 + C3
def tree_diagnostics(pool, dist_fns: dict) -> pd.DataFrame:
    stats = {k: defaultdict(float) for k in ["gold", *dist_fns]}
    for s in pool:
        keep = s.nonpunct_mask
        idx = np.flatnonzero(keep)
        is_fn = np.array([u in FUNCTION_UPOS for u in s.upos])
        gold = gold_mst_edges(s.heads, keep)
        pairs = [(int(a), int(b)) for x, a in enumerate(idx) for b in idx[x + 1:]]
        non = [p for p in pairs if p not in gold]
        for key in stats:
            if key == "gold":
                edges, D = gold, None
            else:
                D = dist_fns[key](s)
                edges = mst_on_subset(D, keep)
            deg = Counter([a for a, _ in edges] + [b for _, b in edges])
            st = stats[key]
            st["n_sent"] += 1
            st["edges"] += len(edges)
            st["fn_tokens"] += int(is_fn[idx].sum())
            st["fn_degree_sum"] += sum(deg[int(i)] for i in idx if is_fn[i])
            st["edges_touching_fn"] += sum(is_fn[a] or is_fn[b] for a, b in edges)
            st["max_degree_sum"] += max(deg.values()) if deg else 0
            st["same_upos_edges"] += sum(s.upos[a] == s.upos[b] for a, b in edges)
            if key == "gold":
                st["nonedge_pairs"] += len(non)
                st["same_upos_nonedges"] += sum(s.upos[a] == s.upos[b] for a, b in non)
            elif non and gold:  # C3: AUC of -distance for gold edges vs non-edges (ties = 0.5)
                sc = np.array([-D[a, b] for a, b in list(gold) + non])
                r = rankdata(sc)
                n1, n0 = len(gold), len(non)
                st["auc_sum"] += (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                st["auc_n"] += 1
    rows = []
    for key, st in stats.items():
        rows.append(dict(
            system=key, mean_fn_token_degree=st["fn_degree_sum"] / max(st["fn_tokens"], 1),
            share_edges_touching_fn=st["edges_touching_fn"] / st["edges"],
            mean_max_degree=st["max_degree_sum"] / st["n_sent"],
            p_same_upos_edges=st["same_upos_edges"] / st["edges"],
            p_same_upos_nonedges=st["same_upos_nonedges"] / st["nonedge_pairs"] if key == "gold" else np.nan,
            edge_auc=st["auc_sum"] / st["auc_n"] if st["auc_n"] else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    C.set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    t0 = time.time()
    bl = best_layer()
    L, k = bl["l_star"], bl["distance_rank"]
    w2i = load_w2i()
    train = load_ewt("train")
    counts = word_counts(train)
    upos = majority_upos(train)
    M = cooccurrence_matrix(train, w2i, C.WINDOW)
    T1, T3 = load_t1(), load_t3_static(L)
    T3a = abtt(T3)

    print("[diag] A + C1: hubness and nearest-neighbour types", flush=True)
    lex, hub_tabs = [], []
    for name, V in (("T1 PPMI-SVD", T1), (f"T3 L{L}", T3), (f"T3 L{L} + ABTT", T3a), ("T3 L0", load_t3_static(0))):
        r, h = lexical_diagnostics(name, V, w2i, counts, upos, M, rng)
        lex.append(r)
        hub_tabs.append(h)
    pd.concat(hub_tabs).to_csv(OUT / "hubs.csv", index=False)

    print("[diag] B + C2 + C3: tree statistics on the 660-sentence pool", flush=True)
    _, pool = get_mst_sentences()
    ctx, Bm = ContextualTest(), probe_matrix(L, k)

    def static(V):
        return lambda s: cosine_distance_matrix(_sentence_vectors(s, V, w2i))

    fns = {"linear chain": lambda s: linear_chain_dist(len(s)), "T1 PPMI-SVD": static(T1),
           f"T3 L{L}": static(T3), f"T3 L{L} + ABTT": static(T3a),
           f"contextual L{L} cosine": lambda s: cosine_distance_matrix(ctx.vectors(s, L)),
           f"T2 probe L{L}": lambda s: probe_sq_dist(ctx.vectors(s, L), Bm)}
    trees = tree_diagnostics(pool, fns)
    trees.to_csv(OUT / "edge_stats.csv", index=False)

    (OUT / "diagnostics.json").write_text(json.dumps(dict(lexical=lex, trees=trees.to_dict("records"),
                                                          function_upos=sorted(FUNCTION_UPOS)), indent=1))
    print("\n================ TASK 3 DIAGNOSTICS ================")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(pd.DataFrame(lex).drop(columns=["top10_hubs"]).to_string(index=False))
        print()
        print(trees.to_string(index=False))
    print(f"[diag] done in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
