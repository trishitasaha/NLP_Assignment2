"""Task 3b: evaluate the BERT-static vectors with the *identical* Task 1 protocol and compare.

What it runs (everything reuses common/static_eval.evaluate_static_vectors, the very function
Task 1 used, so the protocol is identical by construction):
  * T3   = BERT-static vectors from layer l* (dev-selected in Task 2), mean-pooled over train
  * T3+  = the same vectors after all-but-the-top (mean removal + top-7 PCs removed), the
           Tier B anisotropy fix
  * T2 reference row: the trained Task 2 distance probe (layer l*) on the same 20 sentences and
    the same 660-sentence pool (contextual, supervised; an upper reference, not a competitor)
  * paired significance tests against T1 on the *same items*:
        WordSim-353  paired bootstrap of the Spearman difference (same covered pairs)
        analogy      exact McNemar test (same covered questions)
        MST          paired bootstrap of the UUAS difference (same sentences)

Usage (from the project root, after pool_static.py):
    python code/task3/evaluate_task3.py

Outputs -> outputs/task3/
    main/, main_abtt/         full per-benchmark outputs (same files as outputs/task1/)
    t2_reference_mst_20.json  the probe's trees on the 20 sentences
    comparison_table.csv      T1 spec, T1 best variant (p0), T3, T3+ABTT, T2 reference
    paired_tests.json         the significance tests
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import benchmarks as B  # noqa: E402
from common import config as C  # noqa: E402
from common.static_eval import (evaluate_static_vectors, get_mst_sentences, mst_check,  # noqa: E402
                                per_sentence_frame, static_mst_check)
from common.stats import mcnemar_exact, paired_bootstrap_spearman, paired_bootstrap_uuas  # noqa: E402

from t3_common import (ContextualTest, abtt, best_layer, load_t1, load_t3_static, load_w2i,  # noqa: E402
                       probe_matrix, probe_sq_dist, t1_dir)

T1_BEST_TAG = "p0"  # best classical variant found in the Phase 1 ablations (eigenvalue weight p = 0)


def table_row(system: str, r: dict) -> dict:
    ws, an, m20, mp = r["wordsim"], r["analogy"], r["mst_20"], r.get("mst_pool") or {}
    return dict(
        system=system,
        ws_rho_cov=ws["spearman_covered"], ws_ci_low=ws["spearman_covered_ci95"][0],
        ws_ci_high=ws["spearman_covered_ci95"][1], ws_n=ws["n_covered"], ws_rho_full=ws["spearman_full"],
        ws_rho_freq20=ws.get("by_min_train_freq", {}).get("20", {}).get("spearman"),
        an_acc_cov=an["acc_covered"], an_sem_cov=an["by_kind"]["semantic"]["acc_covered"],
        an_syn_cov=an["by_kind"]["syntactic"]["acc_covered"], an_acc_full=an["acc_full"], an_n=an["n_covered"],
        mst20_uuas=m20["uuas"], mst20_ci_low=m20["uuas_ci95"][0], mst20_ci_high=m20["uuas_ci95"][1],
        pool_uuas=mp.get("uuas"))


def ws_pairs(d: Path) -> pd.DataFrame:
    """All 353 rows in benchmark order (WS-353 contains one duplicated pair, so rows are
    aligned by position, not merged on the words)."""
    return pd.read_csv(d / "wordsim_pairs.csv")[["word1", "word2", "human", "model", "covered"]]


def analogy_hits(examples: pd.DataFrame) -> pd.DataFrame:
    return examples[["section", "a", "b", "c", "gold", "correct"]]


def pool_frame(d: Path, vectors, w2i, pool) -> pd.DataFrame:
    """Per-sentence pool counts; recomputed if an older run did not save them."""
    f = d / "mst_pool_per_sentence.csv"
    if f.exists():
        return pd.read_csv(f)
    return per_sentence_frame(static_mst_check(pool, vectors, w2i, "pool", n_random=0))


def paired_block(name_a, name_b, A: dict, Bd: dict) -> dict:
    """A, Bd: dict(ws=DataFrame, an=DataFrame, m20=DataFrame, pool=DataFrame)."""
    out = dict(a=name_a, b=name_b)
    if A.get("ws") is not None and Bd.get("ws") is not None:
        a, b = A["ws"], Bd["ws"]
        assert (a.word1.values == b.word1.values).all() and (a.word2.values == b.word2.values).all()
        both = (a.covered & b.covered).values
        out["wordsim"] = dict(n=int(both.sum()), **paired_bootstrap_spearman(
            a.model.values[both], b.model.values[both], a.human.values[both]))
    if A.get("an") is not None and Bd.get("an") is not None:
        a = A["an"].merge(Bd["an"], on=["section", "a", "b", "c", "gold"], suffixes=("_a", "_b"))
        out["analogy"] = dict(n=len(a), acc_a=float(a.correct_a.mean()), acc_b=float(a.correct_b.mean()),
                              **mcnemar_exact(a.correct_a, a.correct_b))
    for key in ("m20", "pool"):
        m = A[key].merge(Bd[key], on=["sent_id", "total"], suffixes=("_a", "_b"))
        out["mst_20" if key == "m20" else "mst_pool"] = dict(
            n_sentences=len(m), **paired_bootstrap_uuas(m.correct_a, m.correct_b, m.total))
    return out


def main() -> None:
    C.set_seed()
    bl = best_layer()
    L, k = bl["l_star"], bl["distance_rank"]
    w2i = load_w2i()
    sample, pool = get_mst_sentences()
    print(f"[t3] l* = {L} (distance-probe rank {k}); vocab = {len(w2i)}", flush=True)

    # ---- T3 (required) and T3 + all-but-the-top (Tier B)
    V3 = load_t3_static(L)
    r3 = evaluate_static_vectors(V3, w2i, f"T3 BERT-static (layer {L}, mean-pooled)", C.TASK3_OUT / "main")
    r3a = evaluate_static_vectors(abtt(V3), w2i, f"T3 BERT-static (layer {L}) + ABTT(D=7)",
                                  C.TASK3_OUT / "main_abtt")

    # ---- T2 reference: the probe on the same sentences
    ctx, Bm = ContextualTest(), probe_matrix(L, k)
    fn = lambda s: probe_sq_dist(ctx.vectors(s, L), Bm)  # noqa: E731
    t2_20 = mst_check(sample, fn, f"T2 distance probe (layer {L}, k={k})")
    t2_pool = mst_check(pool, fn, "T2 probe pool", n_random=0)
    (C.TASK3_OUT / "t2_reference_mst_20.json").write_text(json.dumps(t2_20, indent=1))

    # ---- comparison table
    rows = [table_row("T1 PPMI-SVD (spec)", json.loads((t1_dir() / "results.json").read_text())),
            table_row(f"T1 PPMI-SVD best variant ({T1_BEST_TAG})",
                      json.loads((t1_dir(T1_BEST_TAG) / "results.json").read_text())),
            table_row(f"T3 BERT-static L{L}", r3), table_row(f"T3 BERT-static L{L} + ABTT", r3a)]
    rows.append(dict(system=f"T2 probe L{L} k={k} (contextual, supervised; reference)",
                     mst20_uuas=t2_20["uuas"], mst20_ci_low=t2_20["uuas_ci95"][0],
                     mst20_ci_high=t2_20["uuas_ci95"][1], pool_uuas=t2_pool["uuas"]))
    rows.append(dict(system="baseline: linear chain", mst20_uuas=t2_20["chain_uuas"], pool_uuas=t2_pool["chain_uuas"]))
    rows.append(dict(system="baseline: random tree", mst20_uuas=r3["mst_20"]["random_uuas"],
                     pool_uuas=r3["mst_pool"]["random_uuas"]))
    table = pd.DataFrame(rows)
    table.to_csv(C.TASK3_OUT / "comparison_table.csv", index=False)

    # ---- paired tests on identical items
    def per_sent_20(path: Path) -> pd.DataFrame:
        return per_sentence_frame(json.loads(path.read_text()))

    t1_examples = pd.DataFrame(B.evaluate_analogy(load_t1(), w2i)["examples"])  # all covered questions
    sys_ = {
        "T1": dict(ws=ws_pairs(t1_dir()), an=analogy_hits(t1_examples),
                   m20=per_sent_20(t1_dir() / "mst_20_sentences.json"),
                   pool=pool_frame(t1_dir(), load_t1(), w2i, pool)),
        "T3": dict(ws=ws_pairs(C.TASK3_OUT / "main"),
                   an=analogy_hits(pd.read_csv(C.TASK3_OUT / "main" / "analogy_examples.csv")),
                   m20=per_sent_20(C.TASK3_OUT / "main" / "mst_20_sentences.json"),
                   pool=pd.read_csv(C.TASK3_OUT / "main" / "mst_pool_per_sentence.csv")),
        "T3+ABTT": dict(ws=ws_pairs(C.TASK3_OUT / "main_abtt"),
                        an=analogy_hits(pd.read_csv(C.TASK3_OUT / "main_abtt" / "analogy_examples.csv")),
                        m20=per_sent_20(C.TASK3_OUT / "main_abtt" / "mst_20_sentences.json"),
                        pool=pd.read_csv(C.TASK3_OUT / "main_abtt" / "mst_pool_per_sentence.csv")),
        "T2probe": dict(m20=per_sentence_frame(t2_20), pool=per_sentence_frame(t2_pool)),
    }
    assert len(sys_["T1"]["an"]) == r3["analogy"]["n_covered"], "analogy items differ between T1 and T3"
    tests = [paired_block("T3", "T1", sys_["T3"], sys_["T1"]),
             paired_block("T3+ABTT", "T1", sys_["T3+ABTT"], sys_["T1"]),
             paired_block("T3+ABTT", "T3", sys_["T3+ABTT"], sys_["T3"]),
             paired_block("T2probe", "T3", sys_["T2probe"], sys_["T3"])]
    (C.TASK3_OUT / "paired_tests.json").write_text(json.dumps(tests, indent=1))

    # ---- summary
    print("\n================ TASK 3 vs TASK 1 (identical protocol) ================")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(table[["system", "ws_rho_cov", "ws_ci_low", "ws_ci_high", "ws_rho_full", "ws_rho_freq20",
                     "an_acc_cov", "an_sem_cov", "an_syn_cov", "an_acc_full", "mst20_uuas", "pool_uuas"]])
    print("\nPaired tests (A - B on identical items):")
    for t in tests:
        parts = []
        if "wordsim" in t:
            w = t["wordsim"]
            parts.append(f"WS d_rho={w['diff']:+.3f} [{w['ci_low']:+.3f},{w['ci_high']:+.3f}] (n={w['n']})")
        if "analogy" in t:
            a = t["analogy"]
            parts.append(f"analogy {a['acc_a']:.3f} vs {a['acc_b']:.3f}, McNemar {a['a_only']}/{a['b_only']} p={a['p_value']:.1e}")
        for key in ("mst_20", "mst_pool"):
            m = t[key]
            parts.append(f"{key} dUUAS={m['diff']:+.3f} [{m['ci_low']:+.3f},{m['ci_high']:+.3f}]")
        print(f"  {t['a']:8s} - {t['b']:8s}: " + " | ".join(parts))
    print("=" * 72)


if __name__ == "__main__":
    main()
