"""Task 2 add-on: is the dev-selected l* significantly better than its neighbouring layers?

The test UUAS curve is a plateau over layers 6-8 (79.6 / 79.5 / 79.5). This script runs a
paired bootstrap over sentences (same sentences for both layers) for every layer vs l*, on dev
and on test, so the report can state whether "l* = 8" is a real peak or one point of a plateau.

Usage (from the project root, after train_probes.py / evaluate_task2.py):
    python code/task2/layer_significance.py
Output: outputs/task2/layer_significance.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task3"))
from common import config as C  # noqa: E402
from common.data import load_ewt  # noqa: E402
from common.features import load_layer, load_meta, load_offsets  # noqa: E402
from common.stats import bootstrap_uuas, paired_bootstrap_uuas  # noqa: E402
from common.trees import gold_mst_edges, mst_on_subset  # noqa: E402

from t3_common import probe_matrix, probe_sq_dist  # noqa: E402


def per_sentence_counts(split: str, layer: int, rank: int, sents, gold) -> tuple[np.ndarray, np.ndarray]:
    X = load_layer(split, layer, mmap=True)
    off = load_offsets(split)
    Bm = probe_matrix(layer, rank)
    corr, tot = np.zeros(len(sents)), np.zeros(len(sents))
    for i, s in enumerate(sents):
        keep = s.nonpunct_mask
        if keep.sum() < 2:
            continue
        h = np.asarray(X[off[i]:off[i + 1]], dtype=np.float64)
        pred = mst_on_subset(probe_sq_dist(h, Bm), keep)
        corr[i], tot[i] = len(pred & gold[i]), len(gold[i])
    return corr, tot


def main() -> None:
    bl = json.loads((C.TASK2_OUT / "best_layer.json").read_text())
    L, k = bl["l_star"], bl["distance_rank"]
    res = pd.read_csv(C.TASK2_OUT / "results_task2.csv")
    rows = []
    for split in ("dev", "test"):
        sents = load_ewt(split)
        assert load_meta(split)["sent_ids"] == [s.sent_id for s in sents]
        gold = [gold_mst_edges(s.heads, s.nonpunct_mask) if s.nonpunct_mask.sum() >= 2 else set() for s in sents]
        ref_c, tot = per_sentence_counts(split, L, k, sents, gold)
        # consistency check against evaluate_task2.py
        expected = res[(res.model == "BERT") & (res.probe == "distance") & (res.layer == L) &
                       (res["rank"] == k) & (res.split == split)].uuas.iloc[0]
        # (float64 here vs float32 on GPU there: allow a rare MST tie flip)
        assert abs(ref_c.sum() / tot.sum() - expected) < 5e-4, (ref_c.sum() / tot.sum(), expected)
        print(f"[{split}] check vs evaluate_task2: {ref_c.sum() / tot.sum():.5f} vs {expected:.5f}", flush=True)
        for layer in range(C.BERT_NUM_LAYERS + 1):
            c = ref_c if layer == L else per_sentence_counts(split, layer, k, sents, gold)[0]
            t = paired_bootstrap_uuas(c, ref_c, tot) if layer != L else dict(diff=0.0, ci_low=0.0, ci_high=0.0,
                                                                             frac_le_zero=float("nan"))
            lo, hi = bootstrap_uuas(c, tot)
            rows.append(dict(split=split, layer=layer, rank=k, uuas=c.sum() / tot.sum(), ci_low=lo, ci_high=hi,
                             diff_vs_lstar=t["diff"], diff_ci_low=t["ci_low"], diff_ci_high=t["ci_high"],
                             significant=bool(t["ci_high"] < 0 or t["ci_low"] > 0)))
            r = rows[-1]
            print(f"[{split}] L{layer:2d} UUAS={r['uuas']:.4f} [{lo:.4f},{hi:.4f}]  vs L{L}: {r['diff_vs_lstar']:+.4f} "
                  f"[{r['diff_ci_low']:+.4f},{r['diff_ci_high']:+.4f}]{' *' if r['significant'] else ''}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.TASK2_OUT / "layer_significance.csv", index=False)
    for split in ("dev", "test"):
        d = df[df.split == split]
        tied = d[~d.significant].layer.tolist()
        print(f"[{split}] layers statistically indistinguishable from l*={L} (95% paired CI covers 0): {tied}")


if __name__ == "__main__":
    main()
