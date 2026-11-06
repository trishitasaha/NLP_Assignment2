"""Task 3c: analyses beyond the required comparison (plan.md Phase 4, Tier B + Tier C).

1. LAYER SWEEP (Tier B). For every BERT layer 0..12, pooled static vectors, raw and after
   all-but-the-top (ABTT): WS-353 Spearman, analogy accuracy (semantic / syntactic), MST UUAS
   (20 sentences + 660-sentence pool) and anisotropy. For context, the same pool is also parsed
   from the raw *contextual* layer vectors (cosine MST, no training) and with the Task 2
   distance probe of that layer. Question answered: is the best "syntax" layer l* also the best
   layer for static *semantics*?
   T1 is evaluated with ABTT too (D=1 for 100 dims), so any gain from ABTT is not credited to BERT.

2. DIAGNOSTIC LADDER (Tier B) on the same 20 sentences and the 660-sentence pool:
      random tree / linear chain            floor and the purely positional baseline
      T1 PPMI-SVD (spec, best variant)      classical static semantics
      T3 BERT-static l* (raw, ABTT)         BERT with the context averaged away
      raw contextual l*, cosine MST         context, but no training
      T2 distance probe l*                  context + a learned syntactic subspace
   with paired bootstrap tests between neighbouring rungs and per-relation recall.

3. TIER C variants (added to the ladder and, where meaningful, to the semantic benchmarks):
      concat   [L2(T1) ; L2(T3)]            are the two static spaces complementary?
      T3 static through the probe B         does the syntactic subspace help *static* vectors?

Usage (from the project root, after pool_static.py with all layers and evaluate_task3.py):
    python code/task3/extensions_task3.py

Outputs -> outputs/task3/extensions/
    layer_sweep.csv  ladder.csv  ladder_relations.csv  ladder_tests.json  ladder_20_edges.json
    tierC_semantic.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import benchmarks as B  # noqa: E402
from common import config as C  # noqa: E402
from common.data import load_ewt, word_counts  # noqa: E402
from common.static_eval import _sentence_vectors, get_mst_sentences, mst_check, static_mst_check  # noqa: E402
from common.stats import paired_bootstrap_uuas  # noqa: E402
from common.trees import cosine_distance_matrix  # noqa: E402

from t3_common import (ContextualTest, abtt, anisotropy, best_layer, l2_normalise, load_t1,  # noqa: E402
                       load_t3_static, load_w2i, probe_matrix, probe_sq_dist)

OUT = C.TASK3_OUT / "extensions"
T1_BEST_TAG = "p0"


def semantic_scores(V, w2i, counts) -> dict:
    ws = B.evaluate_wordsim(V, w2i)
    rec = pd.DataFrame(ws["records"])
    cov = rec[rec.covered]
    minf = cov.apply(lambda r: min(counts[r.word1], counts[r.word2]), axis=1)
    an = B.evaluate_analogy(V, w2i)
    return dict(ws_rho_cov=ws["spearman_covered"], ws_rho_full=ws["spearman_full"],
                ws_rho_freq20=float(spearmanr(cov.model[minf >= 20], cov.human[minf >= 20]).statistic),
                an_acc_cov=an["acc_covered"], an_sem_cov=an["by_kind"]["semantic"]["acc_covered"],
                an_syn_cov=an["by_kind"]["syntactic"]["acc_covered"])


def static_rows(name, V, w2i, counts, sample, pool, extra=None) -> dict:
    t = time.time()
    row = dict(system=name, **(extra or {}), **semantic_scores(V, w2i, counts),
               mst20_uuas=static_mst_check(sample, V, w2i, name, n_random=0)["uuas"],
               pool_uuas=static_mst_check(pool, V, w2i, name, n_random=0)["uuas"],
               anisotropy=anisotropy(V))
    print(f"   {name:34s} WS={row['ws_rho_cov']:.3f} (f>=20 {row['ws_rho_freq20']:.3f}) "
          f"an={row['an_acc_cov']:.4f} (sem {row['an_sem_cov']:.3f} syn {row['an_syn_cov']:.3f}) "
          f"MST20={row['mst20_uuas']:.3f} pool={row['pool_uuas']:.3f} aniso={row['anisotropy']:.3f} "
          f"({time.time() - t:.0f} s)", flush=True)
    return row


# --------------------------------------------------------------------------- 1. layer sweep
def layer_sweep(w2i, counts, sample, pool, ctx, k) -> pd.DataFrame:
    print("[ext] 1. layer sweep", flush=True)
    rows = []
    for tag, label in ((None, "T1 spec"), (T1_BEST_TAG, f"T1 {T1_BEST_TAG}")):
        V = load_t1(tag)
        rows.append(static_rows(f"{label} raw", V, w2i, counts, sample, pool, dict(source="T1", layer=-1, post="raw")))
        rows.append(static_rows(f"{label} ABTT(D=1)", abtt(V), w2i, counts, sample, pool,
                                dict(source="T1", layer=-1, post="abtt")))
    for L in range(C.BERT_NUM_LAYERS + 1):
        V = load_t3_static(L)
        for post, M in (("raw", V), ("abtt", abtt(V))):
            rows.append(static_rows(f"T3 L{L} {post}", M, w2i, counts, sample, pool,
                                    dict(source="T3", layer=L, post=post)))
        # contextual references on the same pool (no static averaging)
        cos = mst_check(pool, lambda s: cosine_distance_matrix(ctx.vectors(s, L)), "ctx", n_random=0)
        Bm = probe_matrix(L, k)
        prb = mst_check(pool, lambda s: probe_sq_dist(ctx.vectors(s, L), Bm), "probe", n_random=0)
        rows.append(dict(system=f"contextual L{L} cosine", source="contextual", layer=L, post="raw",
                         pool_uuas=cos["uuas"]))
        rows.append(dict(system=f"T2 probe L{L} k={k}", source="probe", layer=L, post="probe",
                         pool_uuas=prb["uuas"]))
        print(f"   contextual L{L}: raw cosine pool={cos['uuas']:.3f}  probe pool={prb['uuas']:.3f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2+3. ladder
def ladder(w2i, sample, pool, ctx, L, k):
    print("[ext] 2. diagnostic ladder", flush=True)
    T1, T1b = load_t1(), load_t1(T1_BEST_TAG)
    T3 = load_t3_static(L)
    T3a = abtt(T3)
    concat = np.concatenate([l2_normalise(T1), l2_normalise(T3)], axis=1)
    Bm = probe_matrix(L, k)

    def static_fn(V):
        return lambda s: cosine_distance_matrix(_sentence_vectors(s, V, w2i))

    rungs = [  # (key, label, tier, dist_fn)
        ("T1", "T1 PPMI-SVD (spec)", "required", static_fn(T1)),
        ("T1best", f"T1 PPMI-SVD ({T1_BEST_TAG})", "B", static_fn(T1b)),
        ("T3", f"T3 BERT-static L{L}", "required", static_fn(T3)),
        ("T3abtt", f"T3 BERT-static L{L} + ABTT", "B", static_fn(T3a)),
        ("concat", "concat [T1 ; T3] (L2-normalised)", "C", static_fn(concat)),
        ("T3probe", f"T3 BERT-static L{L} through probe B", "C",
         lambda s: probe_sq_dist(_sentence_vectors(s, T3, w2i), Bm)),
        ("ctx", f"contextual L{L}, cosine (no training)", "B",
         lambda s: cosine_distance_matrix(ctx.vectors(s, L))),
        ("probe", f"T2 distance probe L{L} k={k}", "reference",
         lambda s: probe_sq_dist(ctx.vectors(s, L), Bm)),
    ]
    res = {}
    for key, label, tier, fn in rungs:
        t = time.time()
        m20 = mst_check(sample, fn, label)  # 20 sentences: with the random-tree baseline
        mp = mst_check(pool, fn, label, n_random=0)
        res[key] = dict(label=label, tier=tier, m20=m20, pool=mp)
        print(f"   {label:42s} MST20={m20['uuas']:.3f} [{m20['uuas_ci95'][0]:.3f},{m20['uuas_ci95'][1]:.3f}] "
              f"pool={mp['uuas']:.3f} [{mp['uuas_ci95'][0]:.3f},{mp['uuas_ci95'][1]:.3f}] ({time.time() - t:.0f} s)",
              flush=True)

    any_ = next(iter(res.values()))
    rows = [dict(key="random", system="random tree (expected)", tier="baseline",
                 mst20_uuas=any_["m20"]["random_uuas"]),
            dict(key="chain", system="linear chain", tier="baseline", mst20_uuas=any_["m20"]["chain_uuas"],
                 mst20_ci_low=any_["m20"]["chain_ci95"][0], mst20_ci_high=any_["m20"]["chain_ci95"][1],
                 pool_uuas=any_["pool"]["chain_uuas"], pool_ci_low=any_["pool"]["chain_ci95"][0],
                 pool_ci_high=any_["pool"]["chain_ci95"][1])]
    for key, r in res.items():
        rows.append(dict(key=key, system=r["label"], tier=r["tier"],
                         mst20_uuas=r["m20"]["uuas"], mst20_ci_low=r["m20"]["uuas_ci95"][0],
                         mst20_ci_high=r["m20"]["uuas_ci95"][1], pool_uuas=r["pool"]["uuas"],
                         pool_ci_low=r["pool"]["uuas_ci95"][0], pool_ci_high=r["pool"]["uuas_ci95"][1],
                         mst20_vs_chain=r["m20"]["vs_chain"]["diff"], pool_vs_chain=r["pool"]["vs_chain"]["diff"]))

    # paired tests between rungs, on identical sentences
    def arr(m):
        return (np.array([p["correct"] for p in m["per_sentence"]]),
                np.array([p["total"] for p in m["per_sentence"]]))

    pairs = [("T3", "T1"), ("T3abtt", "T1"), ("T3abtt", "T3"), ("T1best", "T1"), ("concat", "T3"),
             ("T3probe", "T3"), ("ctx", "T3"), ("probe", "ctx"), ("probe", "T3")]
    tests = []
    for a, b in pairs:
        t = dict(a=res[a]["label"], b=res[b]["label"])
        for part in ("m20", "pool"):
            ca, tot = arr(res[a][part])
            cb, _ = arr(res[b][part])
            t[part] = paired_bootstrap_uuas(ca, cb, tot)
        tests.append(t)

    rel_rows = [dict(key=key, system=r["label"], relation=rel, **v)
                for key, r in res.items() for rel, v in r["pool"]["per_relation"].items()]
    edges20 = {key: [dict(sent_id=p["sent_id"], correct=p["correct"], total=p["total"],
                          pred_edges=p["pred_edges"]) for p in r["m20"]["per_sentence"]] for key, r in res.items()}
    edges20["gold"] = [dict(sent_id=p["sent_id"], tokens=p["tokens"], text=p["text"], gold_edges=p["gold_edges"])
                       for p in any_["m20"]["per_sentence"]]
    return pd.DataFrame(rows), pd.DataFrame(rel_rows), tests, edges20, concat


def main() -> None:
    C.set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    bl = best_layer()
    L, k = bl["l_star"], bl["distance_rank"]
    w2i = load_w2i()
    counts = word_counts(load_ewt("train"))
    sample, pool = get_mst_sentences()
    ctx = ContextualTest()
    t0 = time.time()

    lad, rel, tests, edges20, concat = ladder(w2i, sample, pool, ctx, L, k)
    lad.to_csv(OUT / "ladder.csv", index=False)
    rel.to_csv(OUT / "ladder_relations.csv", index=False)
    (OUT / "ladder_tests.json").write_text(json.dumps(tests, indent=1))
    (OUT / "ladder_20_edges.json").write_text(json.dumps(edges20, indent=1))

    print("[ext] 3. Tier C on the semantic benchmarks", flush=True)
    tc = pd.DataFrame([static_rows("concat [T1 ; T3]", concat, w2i, counts, sample, pool)])
    tc.to_csv(OUT / "tierC_semantic.csv", index=False)

    sweep = layer_sweep(w2i, counts, sample, pool, ctx, k)
    sweep.to_csv(OUT / "layer_sweep.csv", index=False)

    print("\n================ TASK 3 EXTENSIONS ================")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(lad[["system", "tier", "mst20_uuas", "mst20_ci_low", "mst20_ci_high", "pool_uuas", "pool_ci_low",
                   "pool_ci_high"]].to_string(index=False))
        print("\nPaired tests (A - B, same sentences):")
        for t in tests:
            print(f"  {t['a'][:34]:34s} - {t['b'][:28]:28s}: "
                  f"20: {t['m20']['diff']:+.3f} [{t['m20']['ci_low']:+.3f},{t['m20']['ci_high']:+.3f}]  "
                  f"pool: {t['pool']['diff']:+.3f} [{t['pool']['ci_low']:+.3f},{t['pool']['ci_high']:+.3f}]")
        st = sweep[sweep.source == "T3"]
        for post in ("raw", "abtt"):
            s = st[st.post == post]
            print(f"\nT3 {post}: best layer WS={int(s.loc[s.ws_rho_cov.idxmax(), 'layer'])} "
                  f"({s.ws_rho_cov.max():.3f}), analogy={int(s.loc[s.an_acc_cov.idxmax(), 'layer'])} "
                  f"({s.an_acc_cov.max():.3f}), pool MST={int(s.loc[s.pool_uuas.idxmax(), 'layer'])} "
                  f"({s.pool_uuas.max():.3f}); l*={L}")
    print(f"[ext] done in {(time.time() - t0) / 60:.1f} min -> {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
