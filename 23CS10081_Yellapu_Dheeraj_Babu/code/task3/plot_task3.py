"""Task 3 figures + qualitative tables (same style/palette as the Task 2 figures).

Usage (from the project root, after evaluate_task3.py and extensions_task3.py):
    python code/task3/plot_task3.py

Figures -> outputs/task3/figures/ (PNG for slides, PDF for the report):
  layer_sweep        2x2: WS-353 rho, analogy acc, MST UUAS (pool), anisotropy vs BERT layer
  ladder             MST UUAS of every representation on the 660-sentence pool (95% CI) + the 20 sentences
  ladder_relations   recall per dependency relation x representation (pool)
  ws_by_frequency    WS-353 rho restricted to pairs whose words occur >= f times in train
  mst_example        one of the 20 sentences: gold tree vs T1 / T3 / contextual / probe MSTs
Tables -> outputs/task3/qualitative/:
  nearest_neighbours.csv   T1 vs T3 neighbours of the probe words
  analogy_examples.csv     questions solved by one model and not the other
  mst20_per_sentence.csv   UUAS of each representation on each of the 20 sentences
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task2"))
from common import config as C  # noqa: E402

import plot_task2 as P2  # noqa: E402  (palette, rcParams, save(), arc_diagram())
from plot_task2 import GRID, INK, INK2, MARKERS, SEQ_CMAP, SERIES, plt  # noqa: E402,F401

from t3_common import best_layer, t1_dir  # noqa: E402

P2.OUT = C.TASK3_OUT / "figures"
EXT = C.TASK3_OUT / "extensions"
QUAL = C.TASK3_OUT / "qualitative"


def ref_line(ax, y, text, style="--", right=False):
    """Horizontal reference line, labelled at the left (or right) end so two labels never collide."""
    ax.axhline(y, color=INK2, linestyle=style, linewidth=1.0)
    ax.annotate(text, (12 if right else 0, y), xytext=(-2 if right else 2, 3), textcoords="offset points",
                color=INK2, fontsize=8, ha="right" if right else "left")


def fig_layer_sweep(sw: pd.DataFrame, L: int) -> None:
    t3 = sw[sw.source == "T3"]
    t1 = sw[(sw.source == "T1") & sw.system.str.startswith("T1 spec")].set_index("post")
    ctx = sw[sw.source == "contextual"].sort_values("layer")
    prb = sw[sw.source == "probe"].sort_values("layer")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    panels = (("ws_rho_cov", "WordSim-353 Spearman ρ (covered pairs)", 1),
              ("an_acc_cov", "Analogy accuracy, covered (%)", 100),
              ("pool_uuas", "MST UUAS, 660 test sentences (%)", 100),
              ("anisotropy", "Anisotropy (mean cos of random word pairs)", 1))
    for ax, (col, title, sc) in zip(axes.ravel(), panels):
        for i, (post, lab) in enumerate((("raw", "T3 BERT-static (mean-pooled)"), ("abtt", "T3 + all-but-the-top"))):
            s = t3[t3.post == post].sort_values("layer")
            ax.plot(s.layer, s[col] * sc, color=SERIES[i], marker=MARKERS[i], markersize=5, label=lab)
        if col == "pool_uuas":
            ax.plot(ctx.layer, ctx[col] * sc, color=SERIES[2], marker=MARKERS[2], markersize=5,
                    label="contextual vectors, cosine (no training)")
            ax.plot(prb.layer, prb[col] * sc, color=INK, linewidth=1.0, marker=".", label="T2 probe (supervised)")
        ref_line(ax, t1.loc["raw", col] * sc, "T1 PPMI-SVD", right=(col == "anisotropy"))
        if col != "anisotropy":
            ref_line(ax, t1.loc["abtt", col] * sc, "T1 + ABTT", ":", right=True)
        lo, hi = ax.get_ylim()
        hi = hi + 0.10 * (hi - lo)
        ax.set_ylim(lo, hi)
        ax.axvline(L, color=INK2, linestyle=":", linewidth=1.0)
        ax.annotate(f"ℓ* = {L}", (L, hi), xytext=(4, -10), textcoords="offset points", color=INK2, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(range(0, 13))
        ax.set_xlabel("BERT layer ℓ")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2[2:], l1 + l2[2:], loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Static vectors pooled from each BERT layer vs. T1 (dashed/dotted = T1 references)",
                 x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    P2.save(fig, "layer_sweep")


def fig_ladder(lad: pd.DataFrame) -> None:
    lad = lad.iloc[::-1].reset_index(drop=True)  # first rung at the top
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = np.arange(len(lad))
    has = lad.pool_uuas.notna()
    err = np.vstack([(lad.pool_uuas - lad.pool_ci_low)[has], (lad.pool_ci_high - lad.pool_uuas)[has]]) * 100
    ax.errorbar(lad.pool_uuas[has] * 100, y[has], xerr=err, fmt="o", color=SERIES[0], markersize=7,
                elinewidth=1.4, capsize=3, label="660-sentence pool (95% CI)")
    ax.plot(lad.mst20_uuas * 100, y, linestyle="none", marker="s", markersize=6, markerfacecolor="none",
            markeredgecolor=SERIES[1], markeredgewidth=1.4, label="the 20 sampled sentences")
    for yi, v in zip(y, lad.pool_uuas.fillna(lad.mst20_uuas)):
        ax.annotate(f"{v * 100:.1f}", (v * 100, yi), xytext=(8, 5), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_yticks(y, lad.system)
    ax.set_xlabel("UUAS of the MST (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Structural check: how much of the dependency tree each representation recovers", loc="left", fontsize=10)
    P2.save(fig, "ladder")


def fig_ladder_relations(rel: pd.DataFrame) -> None:
    keys = ["T1", "T3", "T3abtt", "ctx", "probe"]
    rel = rel[rel.key.isin(keys)]
    top = rel[rel.key == "T1"].sort_values("total", ascending=False).relation.head(14).tolist()
    M = np.array([[rel[(rel.key == k) & (rel.relation == r)].recall.iloc[0] for k in keys] for r in top])
    names = ["T1\nPPMI-SVD", "T3\nBERT-static", "T3\n+ ABTT", "contextual\ncosine", "T2\nprobe"]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    im = ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(keys)), names, fontsize=8)
    n_tot = {r: int(rel[(rel.key == "T1") & (rel.relation == r)].total.iloc[0]) for r in top}
    ax.set_yticks(range(len(top)), [f"{r} (n={n_tot[r]})" for r in top])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="recall of gold edges")
    ax.set_title("Which dependency relations each representation recovers (660 test sentences)", loc="left", fontsize=10)
    P2.save(fig, "ladder_relations")


def fig_ws_by_frequency(results: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (name, r) in enumerate(results.items()):
        bf = r["wordsim"]["by_min_train_freq"]
        th = [t for t in bf if bf[t]["n"] >= 10]
        ax.plot([int(t) for t in th], [bf[t]["spearman"] for t in th], color=SERIES[i], marker=MARKERS[i],
                markersize=5, label=name)
        if i == 0:
            for t in th:
                ax.annotate(f"n={bf[t]['n']}", (int(t), bf[t]["spearman"]), xytext=(0, 8),
                            textcoords="offset points", fontsize=7, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50], ["≥1", "≥5", "≥10", "≥20", "≥50"])
    ax.set_xlabel("both words occur at least f times in EWT train")
    ax.set_ylabel("WordSim-353 Spearman ρ")
    ax.legend(loc="best", fontsize=8)
    ax.minorticks_off()
    ax.set_title("WordSim-353 ρ on pairs whose words are at least this frequent", loc="left", fontsize=10)
    P2.save(fig, "ws_by_frequency")


def fig_mst_example(edges: dict) -> dict:
    gold = edges["gold"]
    keys = [("T1", "T1 PPMI-SVD"), ("T3", "T3 BERT-static"), ("ctx", "contextual, cosine"), ("probe", "T2 probe")]
    rows = []
    for i, g in enumerate(gold):
        sc = {k: edges[k][i]["correct"] / edges[k][i]["total"] for k, _ in keys}
        rows.append(dict(i=i, sent_id=g["sent_id"], n=edges["T1"][i]["total"] + 1, text=g["text"], **sc))
    df = pd.DataFrame(rows)
    df.drop(columns="i").to_csv(QUAL / "mst20_per_sentence.csv", index=False)
    cand = df[(df.n >= 8) & (df.n <= 14)]
    cand = cand if len(cand) else df
    # a *representative* sentence (not a cherry-picked one): each method's UUAS on it is as close
    # as possible to that method's average over the 20 sentences
    dev = sum((cand[k] - df[k].mean()).abs() for k, _ in keys)
    pick = cand.assign(dev=dev).sort_values(["dev", "n"]).iloc[0]
    i = int(pick.i)
    toks = gold[i]["tokens"]
    g_edges = {tuple(e) for e in gold[i]["gold_edges"]}
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.3 * len(keys)))
    for ax, (k, lab) in zip(axes, keys):
        pred = {tuple(e) for e in edges[k][i]["pred_edges"]}
        P2.arc_diagram(ax, toks, g_edges, pred, f"{lab}: UUAS {len(pred & g_edges) / len(g_edges):.2f}",
                       pred_label=f"{lab} MST")
    fig.suptitle(f"A representative one of the 20 sentences ({gold[i]['sent_id']}): gold tree above, MST of each representation below",
                 x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    P2.save(fig, "mst_example")
    return dict(sent_id=gold[i]["sent_id"], text=gold[i]["text"], **{k: float(pick[k]) for k, _ in keys})


def tables(L: int) -> None:
    nn1 = json.loads((t1_dir() / "nearest_neighbours.json").read_text())
    nn3 = json.loads((C.TASK3_OUT / "main" / "nearest_neighbours.json").read_text())
    nn3a = json.loads((C.TASK3_OUT / "main_abtt" / "nearest_neighbours.json").read_text())
    pd.DataFrame([dict(word=w, T1=", ".join(x for x, _ in nn1.get(w, [])[:6]),
                       T3=", ".join(x for x, _ in nn3.get(w, [])[:6]),
                       T3_abtt=", ".join(x for x, _ in nn3a.get(w, [])[:6])) for w in nn3]).to_csv(
        QUAL / "nearest_neighbours.csv", index=False)
    a1 = pd.read_csv(t1_dir() / "analogy_examples.csv")
    a3 = pd.read_csv(C.TASK3_OUT / "main" / "analogy_examples.csv")
    m = a3.merge(a1, on=["section", "a", "b", "c", "gold"], suffixes=("_T3", "_T1"))
    pick = pd.concat([m[m.correct_T3 & ~m.correct_T1].groupby("section").head(2),
                      m[~m.correct_T3 & m.correct_T1].head(6),
                      m[~m.correct_T3 & ~m.correct_T1].groupby("section").head(1).head(12)])
    pick.to_csv(QUAL / "analogy_examples.csv", index=False)


def main() -> None:
    QUAL.mkdir(parents=True, exist_ok=True)
    L = best_layer()["l_star"]
    sw = pd.read_csv(EXT / "layer_sweep.csv")
    fig_layer_sweep(sw, L)
    fig_ladder(pd.read_csv(EXT / "ladder.csv"))
    fig_ladder_relations(pd.read_csv(EXT / "ladder_relations.csv"))
    fig_ws_by_frequency({"T1 PPMI-SVD": json.loads((t1_dir() / "results.json").read_text()),
                         f"T3 BERT-static L{L}": json.loads((C.TASK3_OUT / "main" / "results.json").read_text()),
                         f"T3 + ABTT": json.loads((C.TASK3_OUT / "main_abtt" / "results.json").read_text())})
    ex = fig_mst_example(json.loads((EXT / "ladder_20_edges.json").read_text()))
    tables(L)
    (QUAL / "mst_example.json").write_text(json.dumps(ex, indent=1))
    print("[plot3] figures ->", P2.OUT, "| tables ->", QUAL)
    print("[plot3] MST example:", ex)


if __name__ == "__main__":
    main()
