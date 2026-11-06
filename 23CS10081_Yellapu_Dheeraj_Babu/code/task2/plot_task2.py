"""Task 2 figures (PNG for slides + PDF for the LaTeX report) and qualitative examples.

Usage (from the project root, after evaluate_task2.py):
    python code/task2/plot_task2.py

Figures -> outputs/task2/figures/:
  uuas_vs_layer          test UUAS vs layer, one line per probe rank, LINEAR baseline, l* marked
  nspr_vs_layer          depth-probe Spearman (NSpr) vs layer, same layout
  metrics_vs_layer       2x2 small multiples: UUAS, DSpr, NSpr, Root%
  relation_recall_heatmap  distance-probe recall per dependency relation x layer (best rank)
  example_<kind>_*       gold vs predicted tree arcs for a good / typical / poor test sentence
  distance_heatmap_*     gold tree distances vs predicted squared distances (paper Fig. 4 style)
  depth_example_*        gold vs predicted depth per word
qualitative_examples.json : the chosen sentences with missed / spurious edges (for the report)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.data import gold_depths, gold_distance_matrix, load_ewt  # noqa: E402
from common.trees import gold_mst_edges, mst_on_subset  # noqa: E402

# ---- palette (dataviz reference instance, light surface; fixed categorical order) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: blue, orange, aqua
MARKERS = ["o", "s", "^"]  # secondary encoding: identity never colour-only
SEQ_CMAP = "Blues"  # single-hue sequential for magnitudes
OUT = C.TASK2_OUT / "figures"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 1.6,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_panel(ax, df, probe, metric, l_star, lin_value, ylabel, scale=1.0, label_lines=True):
    sub = df[(df.model == "BERT") & (df.probe == probe) & (df.split == "test")]
    ranks = sorted(sub["rank"].unique())
    ends = []
    for i, k in enumerate(ranks):
        s = sub[sub["rank"] == k].sort_values("layer")
        ax.plot(s.layer, s[metric] * scale, color=SERIES[i], marker=MARKERS[i], markersize=5,
                label=f"rank k={k}")
        ends.append((s[metric].iloc[-1] * scale, f"k={k}", s.layer.iloc[-1]))
    if lin_value is not None:
        ax.axhline(lin_value * scale, color=INK2, linestyle="--", linewidth=1.0)
        ax.annotate("LINEAR baseline", (0, lin_value * scale), xytext=(2, 4), textcoords="offset points",
                    color=INK2, fontsize=8)
    ax.axvline(l_star, color=INK2, linestyle=":", linewidth=1.0)
    lo, hi = ax.get_ylim()
    hi = hi + 0.10 * (hi - lo)  # headroom above the data so the ℓ* label never sits on a marker
    ax.set_ylim(lo, hi)
    ax.annotate(f"ℓ* = {l_star} (dev)", (l_star, hi), xytext=(4, -10), textcoords="offset points",
                color=INK2, fontsize=8)
    if label_lines:  # direct labels at the right end, dodged so they never overlap
        gap = 0.045 * (hi - lo)
        placed = []
        for y, txt, x in sorted(ends):
            y_lab = max(y, placed[-1] + gap) if placed else y
            placed.append(y_lab)
            ax.annotate(txt, (x, y), xytext=(x + 0.25, y_lab), textcoords="data", color=INK2, fontsize=8,
                        va="center", annotation_clip=False)
    ax.set_xticks(range(0, 13))
    ax.set_xlabel("BERT layer ℓ (0 = embeddings)")
    ax.set_ylabel(ylabel)


def arc_diagram(ax, tokens, gold, pred, title, pred_label: str = "probe MST"):
    """Gold edges as arcs above the words, predicted MST edges below.
    Predicted edges: blue = correct, orange dashed = wrong (not in gold)."""
    n = len(tokens)
    for i, t in enumerate(tokens):
        ax.text(i, 0, t, ha="center", va="center", fontsize=8, rotation=0, color=INK)
    for a, b in gold:
        ax.add_patch(Arc(((a + b) / 2, 0.35), b - a, 0.6 * (b - a) ** 0.8 + 0.4, theta1=0, theta2=180,
                         color=INK2, linewidth=1.2))
    for a, b in pred:
        ok = (a, b) in gold
        ax.add_patch(Arc(((a + b) / 2, -0.35), b - a, 0.6 * (b - a) ** 0.8 + 0.4, theta1=180, theta2=360,
                         color=SERIES[0] if ok else SERIES[1], linewidth=1.6 if ok else 1.4,
                         linestyle="-" if ok else "--"))
    h = 0.6 * max(1, n) ** 0.8 / 2 + 1
    ax.set_xlim(-1, n)
    ax.set_ylim(-h, h)
    ax.axis("off")
    ax.text(-1, h * 0.9, "gold tree (above)", fontsize=8, color=INK2)
    ax.text(-1, -h * 0.95, f"{pred_label} (below): solid blue = correct, dashed orange = wrong", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, loc="left")


def main() -> None:
    df = pd.read_csv(C.TASK2_OUT / "results_task2.csv")
    best = json.loads((C.TASK2_OUT / "best_layer.json").read_text())
    l_star = best["l_star"]
    lin = df[(df.model == "LINEAR") & (df.split == "test")]
    lin_d, lin_n = lin[lin.probe == "distance"].iloc[0], lin[lin.probe == "depth"].iloc[0]

    # 1-2. headline single-metric figures (slides)
    for probe, metric, lv, ylabel, scale, name in (
            ("distance", "uuas", lin_d.uuas, "UUAS on EWT test (%)", 100, "uuas_vs_layer"),
            ("depth", "nspr", lin_n.nspr, "Depth Spearman (NSpr), EWT test", 1, "nspr_vs_layer")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        metric_panel(ax, df, probe, metric, l_star, lv, ylabel, scale)
        ax.set_title(f"{'Distance' if probe == 'distance' else 'Depth'} probe on bert-base-uncased: {ylabel.split(' (')[0].split(',')[0]} by layer",
                     loc="left")
        ax.legend(loc="best", fontsize=8)
        save(fig, name)

    # 3. 2x2 small multiples (report)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    for ax, (probe, metric, lv, ylabel, scale) in zip(axes.ravel(), (
            ("distance", "uuas", lin_d.uuas, "UUAS (%)", 100), ("distance", "dspr", lin_d.dspr, "DSpr.", 1),
            ("depth", "nspr", lin_n.nspr, "NSpr. (depth Spearman)", 1), ("depth", "root", lin_n.root, "Root (%)", 100))):
        metric_panel(ax, df, probe, metric, l_star, lv, ylabel, scale, label_lines=False)
        ax.set_title(ylabel, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Structural probes on EWT test, by BERT layer (dashed = LINEAR baseline, dotted = dev-selected ℓ*)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(1.0, 0.965))
    save(fig, "metrics_vs_layer")

    # 4. per-relation recall heatmap (best dev rank per layer)
    rel = pd.read_csv(C.TASK2_OUT / "per_relation_test.csv")
    dev_d = df[(df.model == "BERT") & (df.probe == "distance") & (df.split == "dev")]
    best_rank = dev_d.loc[dev_d.groupby("layer").uuas.idxmax()].set_index("layer")["rank"]
    rel = rel[rel.apply(lambda r: best_rank.get(r.layer) == r["rank"], axis=1)]
    top = rel[rel.layer == l_star].sort_values("total", ascending=False).relation.head(14).tolist()
    M = rel[rel.relation.isin(top)].pivot(index="relation", columns="layer", values="recall").loc[top]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(M.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns)
    ax.set_yticks(range(M.shape[0]), [f"{r} (n={int(rel[(rel.layer == l_star) & (rel.relation == r)].total.iloc[0])})" for r in top])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
    ax.grid(False)
    ax.set_xlabel("BERT layer ℓ")
    ax.set_title("Distance-probe recall of gold edges, by dependency relation (EWT test)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, label="recall")
    save(fig, "relation_recall_heatmap")

    # 5-7. qualitative examples at l*
    with open(C.TASK2_OUT / "predictions_best.pkl", "rb") as f:
        P = pickle.load(f)
    test = load_ewt("test")
    dist_preds, per_sent = P["preds"][(l_star, "distance", P["k_distance"])]
    depth_preds, _ = P["preds"][(l_star, "depth", P["k_depth"])]
    cand = [i for i, s in enumerate(test) if 10 <= s.nonpunct_mask.sum() <= 18]
    scores = {i: per_sent[i][0] / max(per_sent[i][1], 1) for i in cand}
    order = sorted(cand, key=lambda i: scores[i])
    picks = {"good": order[-1], "typical": order[len(order) // 2], "poor": order[0]}
    qual = {}
    for kind, i in picks.items():
        s = test[i]
        keep = s.nonpunct_mask
        idx = np.flatnonzero(keep)
        remap = {int(o): j for j, o in enumerate(idx)}
        gold = {tuple(sorted((remap[a], remap[b]))) for a, b in gold_mst_edges(s.heads, keep)}
        pred = {tuple(sorted((remap[a], remap[b]))) for a, b in mst_on_subset(dist_preds[i], keep)}
        toks = [s.tokens[o] for o in idx]
        fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(toks)), 3.6))
        arc_diagram(ax, toks, gold, pred, f"{kind.capitalize()} example, layer {l_star}: UUAS {scores[i]:.2f}  ({s.sent_id})")
        save(fig, f"example_{kind}")
        rel_of = {tuple(sorted((remap[d], remap[h - 1]))): s.deprels[d].split(":")[0]
                  for d, h in enumerate(s.heads) if h > 0 and keep[d] and keep[h - 1]}
        qual[kind] = dict(sent_id=s.sent_id, text=s.text, uuas=scores[i],
                          missed=[f"{toks[a]}-{toks[b]} ({rel_of.get((a, b), '?')})" for a, b in sorted(gold - pred)],
                          spurious=[f"{toks[a]}-{toks[b]}" for a, b in sorted(pred - gold)])

    i = picks["typical"]
    s = test[i]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, M, t in ((axes[0], gold_distance_matrix(s.heads), "gold tree distance"),
                     (axes[1], dist_preds[i], f"predicted squared distance (layer {l_star})")):
        im = ax.imshow(M, cmap=SEQ_CMAP + "_r")
        ax.set_xticks(range(len(s)), s.tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(s)), s.tokens, fontsize=7)
        ax.grid(False)
        ax.set_title(t, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Darker = closer in the tree. The probe recovers the block structure of the gold tree.", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "distance_heatmap_typical")

    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(s)), 3.4))
    x = np.arange(len(s))
    ax.plot(x, gold_depths(s.heads), color=INK2, marker="o", markersize=5, label="gold depth")
    ax.plot(x, depth_preds[i], color=SERIES[0], marker="s", markersize=5, label=f"predicted depth (layer {l_star})")
    ax.set_xticks(x, s.tokens, rotation=60, fontsize=8)
    ax.set_ylabel("depth (edges to root)")
    ax.set_title("Depth probe: gold vs predicted depth per word (typical test sentence)", loc="left")
    ax.legend(fontsize=8)
    save(fig, "depth_example_typical")

    (C.TASK2_OUT / "qualitative_examples.json").write_text(json.dumps(qual, indent=2))
    print(f"[plot] figures -> {OUT}")
    for k, v in qual.items():
        print(f"\n[{k}] UUAS={v['uuas']:.2f}  {v['text']}\n   missed:   {v['missed']}\n   spurious: {v['spurious']}")


if __name__ == "__main__":
    main()
