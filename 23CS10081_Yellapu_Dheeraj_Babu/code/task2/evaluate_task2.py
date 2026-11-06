"""Task 2: evaluate every trained probe on dev and test, pick the best layer l*, compare to the paper.

Usage (from the project root, after train_probes.py):
    CUDA_VISIBLE_DEVICES=0 python code/task2/evaluate_task2.py

Metrics (exact definitions of Hewitt's structural-probes/reporter.py, checked 6 Nov 2026):
  UUAS   (distance probe) Prim MST on predicted squared distances vs MST on gold distances, both over
         NON-punctuation words; micro-average of correct undirected edges over all sentences.
  DSpr.  (distance probe) Spearman of each word's predicted-distance row vs its gold row (all words incl.
         punctuation and the diagonal); pooled per sentence length; macro-average over lengths 5..50.
  NSpr.  (depth probe; the assignment's "depth-probe Spearman") Spearman of predicted vs gold depths per
         sentence; averaged per length; macro-average over lengths 5..50.
  Root%  (depth probe) argmin of predicted depth over non-punctuation words == gold root.
Baseline: LINEAR from the paper (distance = |i-j|, depth = position i).

l* selection (decision D6): for each layer 1..12 take the rank with the best DEV UUAS; l* = layer with
the best dev UUAS (ties -> higher dev NSpr). Test numbers are reported for every layer and rank.

Outputs (outputs/task2/):
  results_task2.csv          one row per (layer, rank, probe, split)
  per_relation_test.csv      distance-probe recall per dependency relation (test)
  best_layer.json            l*, chosen ranks, dev/test metrics, agreement with the test argmax
  predictions_best.pkl       test predictions of the l* probes and the layer-0 probes (for plots)
  paper_comparison.csv       our l* (and best-test) rows next to Hewitt & Manning Table 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.trees import gold_mst_edges, linear_chain_dist, mst_on_subset  # noqa: E402

from probe_data import GoldData, LayerFeatures  # noqa: E402
from probes import PROBES  # noqa: E402

PAPER = {  # Hewitt & Manning 2019, Table 1 (PTB WSJ test, Stanford Dependencies)
    "LINEAR": dict(uuas=48.9, dspr=0.58, root=2.9, nspr=0.27),
    "BERTBASE7": dict(uuas=79.8, dspr=0.85, root=88.0, nspr=0.87),
    "BERTLARGE15": dict(uuas=82.5, dspr=0.86, root=89.4, nspr=0.88),
    "BERTLARGE16": dict(uuas=81.7, dspr=0.87, root=90.1, nspr=0.89),
}


# --------------------------------------------------------------------------- metric helpers
def _row_spearman(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Spearman per row (average ranks for ties, as scipy.spearmanr). P, G: [m, n]."""
    rp, rg = rankdata(P, axis=1), rankdata(G, axis=1)
    rp -= rp.mean(1, keepdims=True)
    rg -= rg.mean(1, keepdims=True)
    den = np.sqrt((rp ** 2).sum(1) * (rg ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rp * rg).sum(1) / den


def _macro_by_length(per_len: dict[int, list[float]]) -> float:
    means = [np.nanmean(per_len[n]) for n in range(C.SPEARMAN_MIN_LEN, C.SPEARMAN_MAX_LEN + 1) if per_len.get(n)]
    return float(np.mean(means))


class SplitEval:
    """Gold structures of one split, cached once and reused for every probe."""

    def __init__(self, gold: GoldData):
        self.gold = gold
        self.sents = gold.sents
        self.keep = [s.nonpunct_mask for s in self.sents]
        self.gold_dist = [d.cpu().numpy() for d in gold.dist]
        self.gold_depth = [d.cpu().numpy() for d in gold.depth]
        self.gold_edges = [gold_mst_edges(s.heads, k) for s, k in zip(self.sents, self.keep)]
        self.root = [s.heads.index(0) for s in self.sents]

    def distance_metrics(self, preds: list[np.ndarray], with_relations: bool = False) -> dict:
        correct = total = 0
        per_len = defaultdict(list)
        rel = defaultdict(lambda: [0, 0])
        per_sent = []
        for s, P, G, keep, gold in zip(self.sents, preds, self.gold_dist, self.keep, self.gold_edges):
            pred_edges = mst_on_subset(P, keep)
            c = len(gold & pred_edges)
            correct += c
            total += len(gold)
            per_sent.append((c, len(gold)))
            n = len(s)
            if C.SPEARMAN_MIN_LEN <= n <= C.SPEARMAN_MAX_LEN:
                per_len[n].extend(_row_spearman(P, G).tolist())
            if with_relations:
                for i, h in enumerate(s.heads):
                    if h > 0 and keep[i] and keep[h - 1]:
                        r = s.deprels[i].split(":")[0]
                        rel[r][1] += 1
                        rel[r][0] += tuple(sorted((i, h - 1))) in pred_edges
        out = dict(uuas=correct / total, dspr=_macro_by_length(per_len), per_sentence=per_sent)
        if with_relations:
            out["relations"] = {k: dict(recovered=v[0], total=v[1], recall=v[0] / v[1]) for k, v in rel.items()}
        return out

    def depth_metrics(self, preds: list[np.ndarray]) -> dict:
        per_len = defaultdict(list)
        root_ok = 0
        for s, p, g, keep, root in zip(self.sents, preds, self.gold_depth, self.keep, self.root):
            cand = np.where(keep, p, np.inf) if keep.any() else p
            root_ok += int(np.argmin(cand) == root)
            n = len(s)
            if C.SPEARMAN_MIN_LEN <= n <= C.SPEARMAN_MAX_LEN:
                per_len[n].append(float(_row_spearman(p[None], g[None])[0]))
        return dict(nspr=_macro_by_length(per_len), root=root_ok / len(self.sents))


@torch.no_grad()
def predict(model, feats: LayerFeatures, gold: GoldData) -> list[np.ndarray]:
    out = []
    idx_all = torch.arange(len(gold), device=gold.device)
    for s in range(0, len(gold), 100):
        idx = idx_all[s:s + 100]
        pred = model(feats.inputs(idx)).cpu().numpy()
        for b, i in enumerate(idx.tolist()):
            n = gold.lengths[i]
            out.append(pred[b, :n, :n] if pred.ndim == 3 else pred[b, :n])
    return out


def load_probe(path: Path, device) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu")
    cls, _ = PROBES[ck["probe"]]
    m = cls(C.BERT_DIM, ck["rank"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=str(C.TASK2_OUT / "probes"))
    ap.add_argument("--out-dir", default=str(C.TASK2_OUT),
                    help="where the result files go (use a sub-folder for ablations, e.g. the spec-literal run)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = Path(args.probe_dir)
    golds = {sp: GoldData(sp, device) for sp in ("dev", "test")}
    evals = {sp: SplitEval(g) for sp, g in golds.items()}

    rows, rel_rows, preds_cache = [], [], {}
    # LINEAR baseline (paper): distance |i - j|, depth = position
    for sp, ev in evals.items():
        dm = ev.distance_metrics([linear_chain_dist(len(s)) for s in ev.sents])
        dp = ev.depth_metrics([np.arange(len(s), dtype=float) for s in ev.sents])
        rows.append(dict(layer=-1, rank=0, probe="distance", split=sp, uuas=dm["uuas"], dspr=dm["dspr"], model="LINEAR"))
        rows.append(dict(layer=-1, rank=0, probe="depth", split=sp, nspr=dp["nspr"], root=dp["root"], model="LINEAR"))

    ckpts = sorted(pdir.glob("L*_*_k*.pt"))
    layers = sorted({int(p.name.split("_")[0][1:]) for p in ckpts})
    print(f"[eval] {len(ckpts)} probes over layers {layers}", flush=True)
    if not any(l >= 1 for l in layers):
        sys.exit(f"[eval] no trained probes for layers 1-12 in {pdir}. Probes are written when a layer "
                 f"finishes all its epochs: wait for '[train] all done' in the training log(s), then rerun.")
    for layer in layers:
        feats = {sp: LayerFeatures(g, layer) for sp, g in golds.items()}
        for path in sorted(pdir.glob(f"L{layer}_*_k*.pt")):
            ck = torch.load(path, map_location="cpu")
            model = load_probe(path, device)
            for sp, ev in evals.items():
                P = predict(model, feats[sp], golds[sp])
                base = dict(layer=layer, rank=ck["rank"], probe=ck["probe"], split=sp, model="BERT",
                            best_epoch=ck["best_epoch"], best_dev_loss=ck["best_dev_loss"])
                if ck["probe"] == "distance":
                    m = ev.distance_metrics(P, with_relations=(sp == "test"))
                    rows.append(dict(base, uuas=m["uuas"], dspr=m["dspr"]))
                    if sp == "test":
                        rel_rows += [dict(layer=layer, rank=ck["rank"], relation=k, **v) for k, v in m["relations"].items()]
                        preds_cache[(layer, "distance", ck["rank"])] = (P, m["per_sentence"])
                else:
                    m = ev.depth_metrics(P)
                    rows.append(dict(base, nspr=m["nspr"], root=m["root"]))
                    if sp == "test":
                        preds_cache[(layer, "depth", ck["rank"])] = (P, None)
            print(f"[eval] {path.name}: " + " ".join(
                f"{r['split']}:" + (f"UUAS={r['uuas']:.3f} DSpr={r['dspr']:.3f}" if r["probe"] == "distance"
                                    else f"NSpr={r['nspr']:.3f} Root={r['root']:.3f}")
                for r in rows[-2:]), flush=True)
        del feats
        torch.cuda.empty_cache() if device.type == "cuda" else None

    df = pd.DataFrame(rows)
    df.to_csv(out / "results_task2.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "per_relation_test.csv", index=False)

    # ---------------------------------------------------------------- l* selection on DEV
    bert = df[df.model == "BERT"]
    dist_dev = bert[(bert.probe == "distance") & (bert.split == "dev")]
    dep_dev = bert[(bert.probe == "depth") & (bert.split == "dev")]
    best_rank_d = dist_dev.loc[dist_dev.groupby("layer").uuas.idxmax()].set_index("layer")
    best_rank_n = dep_dev.loc[dep_dev.groupby("layer").nspr.idxmax()].set_index("layer")
    cand = best_rank_d.loc[[l for l in best_rank_d.index if l >= 1]]
    cand = cand.assign(nspr=best_rank_n.loc[cand.index, "nspr"])
    l_star = int(cand.sort_values(["uuas", "nspr"], ascending=False).index[0])
    k_d, k_n = int(best_rank_d.loc[l_star, "rank"]), int(best_rank_n.loc[l_star, "rank"])

    def get(layer, probe, rank, split):
        r = bert[(bert.layer == layer) & (bert.probe == probe) & (bert["rank"] == rank) & (bert.split == split)]
        return r.iloc[0].to_dict()

    test_d = bert[(bert.probe == "distance") & (bert.split == "test") & (bert.layer >= 1)]
    test_n = bert[(bert.probe == "depth") & (bert.split == "test") & (bert.layer >= 1)]
    best = dict(
        l_star=l_star, distance_rank=k_d, depth_rank=k_n,
        selection="dev UUAS (best rank per layer, layers 1-12), tie-break dev NSpr",
        dev=dict(uuas=get(l_star, "distance", k_d, "dev")["uuas"], dspr=get(l_star, "distance", k_d, "dev")["dspr"],
                 nspr=get(l_star, "depth", k_n, "dev")["nspr"], root=get(l_star, "depth", k_n, "dev")["root"]),
        test=dict(uuas=get(l_star, "distance", k_d, "test")["uuas"], dspr=get(l_star, "distance", k_d, "test")["dspr"],
                  nspr=get(l_star, "depth", k_n, "test")["nspr"], root=get(l_star, "depth", k_n, "test")["root"]),
        test_argmax_uuas_layer=int(test_d.loc[test_d.uuas.idxmax(), "layer"]),
        test_argmax_nspr_layer=int(test_n.loc[test_n.nspr.idxmax(), "layer"]),
        dev_argmax_nspr_layer=int(best_rank_n.loc[[l for l in best_rank_n.index if l >= 1], "nspr"].idxmax()),
    )
    (out / "best_layer.json").write_text(json.dumps(best, indent=2))

    keep_preds = {k: v for k, v in preds_cache.items()
                  if (k[0] == l_star and k[2] == (k_d if k[1] == "distance" else k_n)) or k[0] == 0}
    with open(out / "predictions_best.pkl", "wb") as f:
        pickle.dump(dict(l_star=l_star, k_distance=k_d, k_depth=k_n, preds=keep_preds,
                         sent_ids=[s.sent_id for s in evals["test"].sents]), f)

    # ---------------------------------------------------------------- paper comparison
    lin = df[(df.model == "LINEAR") & (df.split == "test")]
    comp = [dict(system=f"Ours: BERT-base-uncased L{l_star} (dev-selected), EWT test", uuas=100 * best["test"]["uuas"],
                 dspr=best["test"]["dspr"], root=100 * best["test"]["root"], nspr=best["test"]["nspr"]),
            dict(system="Ours: LINEAR baseline, EWT test", uuas=100 * lin[lin.probe == "distance"].uuas.iloc[0],
                 dspr=lin[lin.probe == "distance"].dspr.iloc[0], root=100 * lin[lin.probe == "depth"].root.iloc[0],
                 nspr=lin[lin.probe == "depth"].nspr.iloc[0])]
    comp += [dict(system=f"Paper: {k} (PTB WSJ test)", **v) for k, v in PAPER.items()]
    pd.DataFrame(comp).to_csv(out / "paper_comparison.csv", index=False)

    # ---------------------------------------------------------------- printout
    print("\n================ TASK 2 SUMMARY ================")
    print("Per layer (best rank chosen on dev; test metrics shown):")
    print(f"{'layer':>5} | {'k_d':>4} {'UUAS dev':>8} {'UUAS test':>9} {'DSpr test':>9} | {'k_n':>4} {'NSpr dev':>8} {'NSpr test':>9} {'Root test':>9}")
    for layer in sorted(best_rank_d.index):
        kd, kn = int(best_rank_d.loc[layer, "rank"]), int(best_rank_n.loc[layer, "rank"])
        td, tn = get(layer, "distance", kd, "test"), get(layer, "depth", kn, "test")
        print(f"{layer:>5} | {kd:>4} {best_rank_d.loc[layer, 'uuas']:8.3f} {td['uuas']:9.3f} {td['dspr']:9.3f} | "
              f"{kn:>4} {best_rank_n.loc[layer, 'nspr']:8.3f} {tn['nspr']:9.3f} {tn['root']:9.3f}")
    print("\nRank sweep at l* (test):")
    for probe, metric in (("distance", "uuas"), ("depth", "nspr")):
        sub = bert[(bert.layer == l_star) & (bert.probe == probe) & (bert.split == "test")].sort_values("rank")
        print(f"   {probe:8s} " + "  ".join(f"k={int(r['rank'])}: {r[metric]:.3f}" for _, r in sub.iterrows()))
    print(f"\nl* = {l_star} (distance k={k_d}, depth k={k_n}); test argmax UUAS layer = {best['test_argmax_uuas_layer']}, "
          f"test argmax NSpr layer = {best['test_argmax_nspr_layer']}, dev argmax NSpr layer = {best['dev_argmax_nspr_layer']}")
    print("\nPaper comparison (UUAS/Root% in %):")
    print(pd.DataFrame(comp).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("================================================")


if __name__ == "__main__":
    main()
