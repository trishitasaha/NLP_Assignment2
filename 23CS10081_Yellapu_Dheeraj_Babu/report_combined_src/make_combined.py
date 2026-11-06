"""Numbers, tables and figures of the COMBINED report (implementation A vs implementation B).

A = Yellapu Dheeraj Babu: read from outputs/ (same source as report/make_tables.py).
B = teammate: TRANSCRIBED by hand from the teammate's report (Project5_Report.pdf, Tables 1-3 and text);
    no raw outputs of B are available, so every B number lives in the dict `B` below.
X = cross-checks: A's own outputs re-scored under B's settings (count >= 2 vocabulary, missing pair = 0).

Usage (from the project root):  python report_combined/make_combined.py
Writes report_combined/generated/{numbers_b.tex, tab_*.tex, fig_layers_cmp.pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
GEN = ROOT / "report_combined" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "task2"))

# ----------------------------------------------------------------------------- B (transcribed)
B = dict(
    t1=dict(vocab=8865, ws_cov=0.189, ws_n=203, ws_all0=0.024, an_cov=0.0085, an_n=3406, an_sem=0.0198, an_syn=0.0070,
            an_all=0.0015, mst20_micro=0.167, mst20_macro=0.208, full_micro=0.178, full_macro=0.316,
            rand20=0.131, chain20=0.355, rand_full=0.143, chain_full=0.405, edges20=282, full_sents=1926, full_edges=23017,
            novec20=0.074, novec_full=0.115, cooc_nnz=582259, ppmi_nnz=519780, seconds=13),
    # Table 2 (test): layer -> (UUAS k32, k64, k128 [punctuation INCLUDED], UUAS no-punct k32, depth k32, k64, k128)
    t2=pd.DataFrame([
        (1, .5307, .5163, .4928, .5608, .6952, .6861, .6660), (2, .5496, .5395, .5144, .5779, .7134, .6952, .6469),
        (3, .5874, .5806, .5532, .6172, .7359, .7178, .6847), (4, .6273, .6245, .5817, .6547, .7636, .7557, .7118),
        (5, .6311, .6337, .6272, .6555, .7900, .7634, .7093), (6, .6354, .6406, .6304, .6673, .7851, .7590, .7180),
        (7, .6513, .6400, .6400, .6783, .7945, .7880, .7535), (8, .6637, .6516, .6445, .6888, .8019, .7820, .7570),
        (9, .6375, .6102, .6191, .6631, .7946, .7891, .7486), (10, .5994, .5820, .5702, .6288, .7805, .7476, .7146),
        (11, .5794, .5531, .5492, .6070, .7658, .7274, .7116), (12, .5834, .5737, .5423, .6185, .7760, .7635, .7523)],
        columns=["layer", "u32", "u64", "u128", "u32_np", "d32", "d64", "d128"]).set_index("layer"),
    t2best=dict(layer=8, rank=32, dev_uuas=0.6723, uuas=0.6637, uuas_np=0.6888, dspr=0.7836, nspr=0.8019, root=0.7236,
                root_k128_lo=0.48, root_k128_hi=0.57, n_probes=72),
    # Table 3: variant -> metrics
    t3={"same_vocab": dict(ws_cov=0.648, ws_n=203, ws_all0=0.173, an_cov=0.771, an_n=3406, an_all=0.134, sem=0.877, syn=0.757,
                           mst20=0.096, mst20_macro=0.134, full=0.145),
        "all_types": dict(ws_cov=0.577, ws_n=248, ws_all0=0.288, an_cov=0.677, an_n=5727, an_all=0.198, sem=0.745, syn=0.666,
                          mst20=0.099, mst20_macro=0.135, full=0.147),
        "centred": dict(ws_cov=0.706, ws_n=203, ws_all0=0.318, an_cov=0.783, an_n=3406, an_all=0.137, sem=0.889, syn=0.769,
                        mst20=0.135, mst20_macro=0.187, full=0.160),
        "pca100": dict(ws_cov=0.678, ws_n=203, ws_all0=0.296, an_cov=0.814, an_n=3406, an_all=0.142, sem=0.822, syn=0.813,
                       mst20=0.135, mst20_macro=0.180, full=0.155)},
    sweep=dict(ws_layer=1, ws=0.723, an_layer=2, an=0.862, mst_layer=10, mst=0.160),
)

MAC: dict[str, str] = {}


def M(k, v):
    assert k.isalpha(), k
    MAC[k] = str(v)


def f3(x):
    return f"{x:.3f}"


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


# ----------------------------------------------------------------------------- A (from outputs)
from common.data import load_ewt, word_counts  # noqa: E402

res = pd.read_csv(OUT / "task2/results_task2.csv")
lit = pd.read_csv(OUT / "task2/spec_literal/results_task2.csv")
bl = json.loads((OUT / "task2/best_layer.json").read_text())
blit = json.loads((OUT / "task2/spec_literal/best_layer.json").read_text())


def a2(df, layer, probe, rank, col):
    r = df[(df.model == "BERT") & (df.layer == layer) & (df.probe == probe) & (df["rank"] == rank) & (df.split == "test")]
    return float(r[col].iloc[0])


t1a = json.loads((OUT / "task1/results.json").read_text())
t3a = json.loads((OUT / "task3/main/results.json").read_text())
t3aa = json.loads((OUT / "task3/main_abtt/results.json").read_text())
sw = pd.read_csv(OUT / "task3/extensions/layer_sweep.csv")

# ----------------------------------------------------------------------------- X: cross-checks
cnt = word_counts(load_ewt("train"))
X = {}
for key, d in (("T1", OUT / "task1"), ("T3", OUT / "task3/main"), ("T3a", OUT / "task3/main_abtt")):
    w = pd.read_csv(d / "wordsim_pairs.csv")
    cov = w[w.covered]
    sub = cov[cov.apply(lambda r: cnt[r.word1] >= 2 and cnt[r.word2] >= 2, axis=1)]
    X[key] = dict(sub_n=len(sub), sub_rho=spearmanr(sub.model, sub.human).statistic,
                  all0=spearmanr(w.model.where(w.covered, 0.0), w.human).statistic)
X["vocab2"] = sum(1 for c in cnt.values() if c >= 2)
from common.benchmarks import load_analogies  # noqa: E402

X["an_n2"] = sum(all(cnt.get(x, 0) >= 2 for x in q) for s in load_analogies().values() for q in s)
assert X["vocab2"] == B["t1"]["vocab"] and X["an_n2"] == B["t1"]["an_n"]

# ----------------------------------------------------------------------------- macros
M("BVocab", f"{B['t1']['vocab']:,}")
M("BWSCov", f3(B["t1"]["ws_cov"]))
M("BWSAllZero", f3(B["t1"]["ws_all0"]))
M("BANCov", pc(B["t1"]["an_cov"], 2))
M("BANAll", pc(B["t1"]["an_all"], 2))
M("BMSTTwenty", pc(B["t1"]["mst20_micro"]))
M("BMSTFull", pc(B["t1"]["full_micro"]))
M("BRandTwenty", pc(B["t1"]["rand20"]))
M("BChainTwenty", pc(B["t1"]["chain20"]))
M("BRandFull", pc(B["t1"]["rand_full"]))
M("BChainFull", pc(B["t1"]["chain_full"]))
M("BEdgesTwenty", B["t1"]["edges20"])
M("BFullSents", f"{B['t1']['full_sents']:,}")
M("BFullEdges", f"{B['t1']['full_edges']:,}")
M("BNoVecTwenty", pc(B["t1"]["novec20"]))
M("BNoVecFull", pc(B["t1"]["novec_full"]))
bb = B["t2best"]
M("BUUASnp", pc(bb["uuas_np"]))
M("BUUAS", pc(bb["uuas"]))
M("BNSpr", f3(bb["nspr"]))
M("BDSpr", f3(bb["dspr"]))
M("BRoot", pc(bb["root"]))
M("BDevUUAS", pc(bb["dev_uuas"]))
t3b = B["t3"]
for k, nm in (("same_vocab", "Same"), ("all_types", "All"), ("centred", "Cen"), ("pca100", "Pca")):
    M(f"BT{nm}WS", f3(t3b[k]["ws_cov"]))
    M(f"BT{nm}AN", pc(t3b[k]["an_cov"]))
    M(f"BT{nm}MST", pc(t3b[k]["mst20"]))
    M(f"BT{nm}Full", pc(t3b[k]["full"]))
M("XTOneSub", f3(X["T1"]["sub_rho"]))
M("XTThreeSub", f3(X["T3"]["sub_rho"]))
M("XTThreeASub", f3(X["T3a"]["sub_rho"]))
M("XTOneAllZero", f3(X["T1"]["all0"]))
M("XTThreeAllZero", f3(X["T3"]["all0"]))
M("XTThreeAAllZero", f3(X["T3a"]["all0"]))
M("XSubN", X["T1"]["sub_n"])
M("ALitUUASonetwoeight", pc(a2(lit, bl["l_star"], "distance", 128, "uuas")))
M("AUUASthirtytwo", pc(a2(res, bl["l_star"], "distance", 32, "uuas")))
M("ANSprthirtytwo", f3(a2(res, bl["l_star"], "depth", 32, "nspr")))
M("BSweepWS", f3(B["sweep"]["ws"]))
M("BSweepAN", pc(B["sweep"]["an"]))
M("BSweepMST", pc(B["sweep"]["mst"]))
a_mst = sw[(sw.source == "T3") & (sw.post == "raw")].set_index("layer").pool_uuas
M("ASweepMSTLayer", int(a_mst.idxmax()))
M("ASweepMST", pc(a_mst.max()))
lit_hist = None
import torch  # noqa: E402

ck = torch.load(OUT / "task2/probes_spec_literal/L8_distance_k128.pt", map_location="cpu")
dv = [h["dev_loss"] for h in ck["history"]]
M("LitMinEpoch", int(np.argmin(dv)) + 1)
(GEN / "numbers_b.tex").write_text("% generated by report_combined/make_combined.py\n" +
                                   "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())) + "\n")


def write(name, s):
    (GEN / name).write_text(s)


# ----------------------------------------------------------------------------- table: cross-check
rows = [
    ("train / dev / test sentences", "12,544 / 2,001 / 2,077", "12,544 / 2,001 / 2,077", "identical"),
    ("train tokens; lowercased types", "204,578; 16,654", "204,578; 16,654", "identical"),
    ("types with count $\\ge$2 (B's T1 vocabulary)", f"{X['vocab2']:,}", f"{B['t1']['vocab']:,}", "identical"),
    ("analogy questions, all words count $\\ge$2", f"{X['an_n2']:,}", f"{B['t1']['an_n']:,}", "identical"),
    (f"T3 layer 8, WS-353 $\\rho$ on those {X['T3']['sub_n']} pairs", f3(X["T3"]["sub_rho"]), f3(t3b["same_vocab"]["ws_cov"]), "identical"),
    ("T3 layer 8 (all types), WS $\\rho$, 248 covered pairs", f3(t3a["wordsim"]["spearman_covered"]), f3(t3b["all_types"]["ws_cov"]), "identical"),
    ("T3 layer 8 (all types), WS $\\rho$, all 353, missing $=0$", f3(X["T3"]["all0"]), f3(t3b["all_types"]["ws_all0"]), "identical"),
    ("T3 layer 8 (all types), analogy, 5,727 covered", pc(t3a["analogy"]["acc_covered"]), pc(t3b["all_types"]["an_cov"]), "identical"),
    ("\\quad semantic / syntactic", f"{pc(t3a['analogy']['by_kind']['semantic']['acc_covered'])} / {pc(t3a['analogy']['by_kind']['syntactic']['acc_covered'])}",
     f"{pc(t3b['all_types']['sem'])} / {pc(t3b['all_types']['syn'])}", "identical"),
    ("T3 layer 8 (all types), analogy, all 19,544", pc(t3a["analogy"]["acc_full"]), pc(t3b["all_types"]["an_all"]), "identical"),
    (f"T1, WS $\\rho$ on the {X['T1']['sub_n']} pairs", f3(X["T1"]["sub_rho"]) + "$^a$", f3(B["t1"]["ws_cov"]) + "$^b$", "vocabulary cut-off"),
    ("T1, WS $\\rho$, all 353, missing $=0$", f3(X["T1"]["all0"]) + "$^a$", f3(B["t1"]["ws_all0"]) + "$^b$", "vocabulary cut-off"),
]
write("tab_crosscheck.tex", "\\begin{tabular}{lccl}\n\\toprule\nquantity & A (recomputed in B's setting) & B (reported) & \\\\\n\\midrule\n"
      + "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- table: T1
a1 = t1a
rows = [
    ("vocabulary", "all 16,654 train types", f"{B['t1']['vocab']:,} types (count $\\ge$2)"),
    ("WS-353 $\\rho$, covered pairs", f"{f3(a1['wordsim']['spearman_covered'])} (248)", f"{f3(B['t1']['ws_cov'])} (203)"),
    ("WS-353 $\\rho$, all 353", f"{f3(a1['wordsim']['spearman_full'])} (median) / {f3(X['T1']['all0'])} (zero)", f"{f3(B['t1']['ws_all0'])} (zero)"),
    ("analogy acc., covered", f"{pc(a1['analogy']['acc_covered'], 2)}\\% (5,727)", f"{pc(B['t1']['an_cov'], 2)}\\% (3,406)"),
    ("\\quad semantic / syntactic", f"{pc(a1['analogy']['by_kind']['semantic']['acc_covered'], 1)} / {pc(a1['analogy']['by_kind']['syntactic']['acc_covered'], 2)}\\%",
     f"{pc(B['t1']['an_sem'], 1)} / {pc(B['t1']['an_syn'], 2)}\\%"),
    ("analogy acc., all 19,544", f"{pc(a1['analogy']['acc_full'], 2)}\\%", f"{pc(B['t1']['an_all'], 2)}\\%"),
    ("MST UUAS, 20 sentences", f"{pc(a1['mst_20']['uuas'])} (234 edges)", f"{pc(B['t1']['mst20_micro'])} ({B['t1']['edges20']} edges)"),
    ("\\quad random tree / linear chain", f"{pc(a1['mst_20']['random_uuas'])} / {pc(a1['mst_20']['chain_uuas'])}", f"{pc(B['t1']['rand20'])} / {pc(B['t1']['chain20'])}"),
    ("MST UUAS, robustness set", f"{pc(a1['mst_pool']['uuas'])} (660 sent.)", f"{pc(B['t1']['full_micro'])} ({B['t1']['full_sents']:,} sent.)"),
    ("\\quad random tree / linear chain", f"{pc(a1['mst_pool']['random_uuas'])} / {pc(a1['mst_pool']['chain_uuas'])}", f"{pc(B['t1']['rand_full'])} / {pc(B['t1']['chain_full'])}"),
]
write("tab_t1_cmp.tex", "\\begin{tabular}{lll}\n\\toprule\nTask 1 & A & B \\\\\n\\midrule\n"
      + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- table: T2 per layer
rows = []
for layer in range(1, 13):
    b = B["t2"].loc[layer]
    cells = [pc(a2(res, layer, "distance", 32, "uuas")), pc(a2(res, layer, "distance", 128, "uuas")),
             pc(a2(lit, layer, "distance", 128, "uuas")), pc(b.u32_np), pc(b.u128) + "$^\\ast$",
             f3(a2(res, layer, "depth", 32, "nspr")), f3(a2(lit, layer, "depth", 128, "nspr")), f3(b.d32), f3(b.d128)]
    line = f"{layer} & " + " & ".join(cells) + " \\\\"
    rows.append(("\\rowcolor{black!8} " if layer == 8 else "") + line)
write("tab_t2_cmp.tex", "\\begin{tabular}{rccc|cc|cc|cc}\n\\toprule\n & \\multicolumn{5}{c|}{UUAS (\\%), punctuation excluded} & "
      "\\multicolumn{4}{c}{depth Spearman (NSpr)} \\\\\n & \\multicolumn{2}{c}{A, paper schedule} & A, const.\\ lr & "
      "\\multicolumn{2}{c|}{B, const.\\ lr} & A, paper & A, const. & \\multicolumn{2}{c}{B, const.\\ lr} \\\\\n"
      "$\\ell$ & $k{=}32$ & $k{=}128$ & $k{=}128$, last ep. & $k{=}32$ & $k{=}128^\\ast$ & $k{=}32$ & $k{=}128$ & $k{=}32$ & $k{=}128$ \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- table: T2 best + paper
ab = bl["test"]
al = blit["test"]
rows = [
    ("Paper: BERT-base (PTB WSJ, cased)", "7 (0-idx.)", "full", "79.8", "0.85", "0.87", "88.0"),
    ("A: paper schedule ($\\times$0.1 lr decay, best dev)", str(bl["l_star"]), str(bl["distance_rank"]), pc(ab["uuas"]), f3(ab["dspr"]), f3(ab["nspr"]), pc(ab["root"])),
    ("A: constant lr, last epoch (control)", str(blit["l_star"]), "128", pc(al["uuas"]), f3(al["dspr"]), f3(al["nspr"]), pc(al["root"])),
    ("B: constant lr, best dev epoch", str(bb["layer"]), str(bb["rank"]), pc(bb["uuas_np"]), f3(bb["dspr"]) + "$^\\dagger$", f3(bb["nspr"]) + "$^\\ddagger$", pc(bb["root"])),
]
write("tab_t2_best.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nsystem & $\\ell^*$ & $k$ & UUAS & DSpr & NSpr & Root\\% \\\\\n\\midrule\n"
      + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- table: T3 semantic
def a3(r):
    w, a = r["wordsim"], r["analogy"]
    return w, a


rows = []
w, a = t1a["wordsim"], t1a["analogy"]
rows.append(("A", "T1 PPMI-SVD", f"{f3(w['spearman_covered'])} (248)", f"{f3(X['T1']['sub_rho'])}", f"{pc(a['acc_covered'], 2)} (5,727)", f"{pc(a['acc_full'], 2)}"))
rows.append(("B", "T1 PPMI-SVD (count $\\ge$2)", "--", f"{f3(B['t1']['ws_cov'])}", f"{pc(B['t1']['an_cov'], 2)} (3,406)", f"{pc(B['t1']['an_all'], 2)}"))
rows.append(("\\midrule A", "T3 layer 8, all types", f"{f3(t3a['wordsim']['spearman_covered'])} (248)", f"{f3(X['T3']['sub_rho'])}", f"{pc(t3a['analogy']['acc_covered'])} (5,727)", f"{pc(t3a['analogy']['acc_full'])}"))
rows.append(("B", "T3 layer 8, all types", f"{f3(t3b['all_types']['ws_cov'])} (248)", "--", f"{pc(t3b['all_types']['an_cov'])} (5,727)", f"{pc(t3b['all_types']['an_all'])}"))
rows.append(("B", "T3 layer 8, T1 vocabulary (main)", "--", f"{f3(t3b['same_vocab']['ws_cov'])}", f"{pc(t3b['same_vocab']['an_cov'])} (3,406)", f"{pc(t3b['same_vocab']['an_all'])}"))
rows.append(("\\midrule A", "T3 + all-but-the-top ($D{=}7$)", f"{f3(t3aa['wordsim']['spearman_covered'])} (248)", f"{f3(X['T3a']['sub_rho'])}", f"{pc(t3aa['analogy']['acc_covered'])} (5,727)", f"{pc(t3aa['analogy']['acc_full'])}"))
rows.append(("B", "T3 mean-centred (T1 vocab.)", "--", f"{f3(t3b['centred']['ws_cov'])}", f"{pc(t3b['centred']['an_cov'])} (3,406)", f"{pc(t3b['centred']['an_all'])}"))
rows.append(("B", "T3 centred + PCA-100 (T1 vocab.)", "--", f"{f3(t3b['pca100']['ws_cov'])}", f"{pc(t3b['pca100']['an_cov'])} (3,406)", f"{pc(t3b['pca100']['an_all'])}"))
write("tab_t3_sem.tex", "\\begin{tabular}{llcccc}\n\\toprule\n & & \\multicolumn{2}{c}{WordSim-353 $\\rho$} & \\multicolumn{2}{c}{analogy accuracy (\\%)} \\\\\n"
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nimpl. & system & 248 covered & 203 (count $\\ge$2) & covered & all 19,544 \\\\\n\\midrule\n"
      + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- table: T3 structure
rows = [
    ("random tree", pc(t1a["mst_20"]["random_uuas"]), pc(B["t1"]["rand20"]), pc(t1a["mst_pool"]["random_uuas"]), pc(B["t1"]["rand_full"])),
    ("linear chain", pc(t1a["mst_20"]["chain_uuas"]), pc(B["t1"]["chain20"]), pc(t1a["mst_pool"]["chain_uuas"]), pc(B["t1"]["chain_full"])),
    ("\\midrule T1 PPMI-SVD", pc(t1a["mst_20"]["uuas"]), pc(B["t1"]["mst20_micro"]), pc(t1a["mst_pool"]["uuas"]), pc(B["t1"]["full_micro"])),
    ("T3 layer 8 (A: all types; B: T1 vocab.)", pc(t3a["mst_20"]["uuas"]), pc(t3b["same_vocab"]["mst20"]), pc(t3a["mst_pool"]["uuas"]), pc(t3b["same_vocab"]["full"])),
    ("T3 layer 8, all types", pc(t3a["mst_20"]["uuas"]), pc(t3b["all_types"]["mst20"]), pc(t3a["mst_pool"]["uuas"]), pc(t3b["all_types"]["full"])),
    ("T3 + anisotropy fix (A: ABTT; B: centred)", pc(t3aa["mst_20"]["uuas"]), pc(t3b["centred"]["mst20"]), pc(t3aa["mst_pool"]["uuas"]), pc(t3b["centred"]["full"])),
    ("\\midrule T2 probe, layer 8 (A only)", pc(0.811965811965812), "--", pc(0.8287541793865387), "--"),
]
cmp_ = pd.read_csv(OUT / "task3/comparison_table.csv")
t2r = cmp_[cmp_.system.str.startswith("T2 probe")].iloc[0]
rows[-1] = ("\\midrule T2 probe, layer 8 (A only)", pc(t2r.mst20_uuas), "--", pc(t2r.pool_uuas), "--")
write("tab_t3_struct.tex", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{20 sampled sentences} & \\multicolumn{2}{c}{robustness set} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nUUAS (\\%) & A (234 edges) & B (282 edges) & A: 660 sent. & B: 1,926 sent. \\\\\n\\midrule\n"
      + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- figure: layer curves, A vs B
import plot_task2 as P2  # noqa: E402  (shared palette + rcParams)
from plot_task2 import INK, INK2, MARKERS, SERIES, plt  # noqa: E402

layers = list(range(1, 13))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
ax.plot(layers, [100 * a2(res, l, "distance", 128, "uuas") for l in layers], color=SERIES[0], marker=MARKERS[0], ms=5, label="A, paper schedule, k=128")
ax.plot(layers, [100 * a2(res, l, "distance", 32, "uuas") for l in layers], color=INK, marker=".", lw=1.0, label="A, paper schedule, k=32")
ax.plot(layers, [100 * a2(lit, l, "distance", 128, "uuas") for l in layers], color=SERIES[1], marker=MARKERS[1], ms=5, label="A, constant lr, k=128 (last epoch)")
ax.plot(layers, [100 * B["t2"].loc[l].u32_np for l in layers], color=SERIES[2], marker=MARKERS[2], ms=5, label="B, constant lr, k=32 (best dev)")
ax.axhline(79.8, color=INK2, ls="--", lw=1.0)
ax.annotate("paper, BERT-base (PTB)", (1, 79.8), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK2)
ax.set_title("UUAS on EWT test (%), punctuation excluded", loc="left", fontsize=10)
ax = axes[1]
ax.plot(layers, [a2(res, l, "depth", 128, "nspr") for l in layers], color=SERIES[0], marker=MARKERS[0], ms=5)
ax.plot(layers, [a2(res, l, "depth", 32, "nspr") for l in layers], color=INK, marker=".", lw=1.0)
ax.plot(layers, [a2(lit, l, "depth", 128, "nspr") for l in layers], color=SERIES[1], marker=MARKERS[1], ms=5)
ax.plot(layers, [B["t2"].loc[l].d32 for l in layers], color=SERIES[2], marker=MARKERS[2], ms=5)
ax.axhline(0.87, color=INK2, ls="--", lw=1.0)
ax.annotate("paper", (1, 0.87), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK2)
ax.set_title("Depth-probe Spearman (NSpr) on EWT test", loc="left", fontsize=10)
for ax in axes:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    ax.axvline(8, color=INK2, ls=":", lw=1.0)
    ax.set_xticks(layers)
    ax.set_xlabel("BERT layer $\\ell$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(GEN / "fig_layers_cmp.pdf", bbox_inches="tight")
fig.savefig(GEN / "fig_layers_cmp.png", dpi=200, bbox_inches="tight")
print(f"[combined] {len(MAC)} macros, tables + figure -> {GEN}")
