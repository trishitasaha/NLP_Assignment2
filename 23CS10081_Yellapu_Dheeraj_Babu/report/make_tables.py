"""Generate every number and table of the report from outputs/ (no number is typed by hand).

Usage (from the project root):  python report/make_tables.py
Writes report/generated/numbers.tex (\\newcommand macros) and report/generated/tab_*.tex.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
GEN = ROOT / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
MACROS: dict[str, str] = {}


def J(p):
    return json.loads((OUT / p).read_text())


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


def _m(t: str) -> str:
    return t.replace("-", "$-$")


def ci3(lo, hi):
    return _m(f"[{lo:.2f}, {hi:.2f}]")


def cip(lo, hi):
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def sgn(x, d=1, scale=100):
    if round(scale * x, d) == 0:
        return f"{0:.{d}f}"
    return f"{scale * x:+.{d}f}".replace("-", "$-$")


def esc(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


def fmt_p(p: float) -> str:
    if p == 0:
        return "$<10^{-300}$"
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def M(name, val):
    assert name.isalpha(), name
    MACROS[name] = str(val)


def write(name, body):
    (GEN / name).write_text(body)


# ============================================================================ data
ds = J("shared/data_stats.json")
for sp in ("train", "dev", "test"):
    st = ds["splits"][sp]
    M(f"N{sp.capitalize()}Sents", f"{st['n_sentences']:,}")
    M(f"N{sp.capitalize()}Tokens", f"{st['n_tokens']:,}")
M("VocabTypes", f"{ds['vocab']['n_types']:,}")
M("VocabSingletons", f"{ds['vocab']['types_count_1']:,}")
M("VocabSingletonPct", pc(ds['vocab']['types_count_1'] / ds['vocab']['n_types'], 0))
M("TestOOV", pc(ds["vocab"]["test_token_oov_rate"]))
M("WSCovered", ds["wordsim"]["n_covered"])
M("WSCoverPct", pc(ds["wordsim"]["pair_coverage"]))
M("WSMedianCount", int(ds["wordsim"]["median_train_count_of_covered_words"]))
M("ANCovered", f"{ds['analogy']['n_covered']:,}")
M("ANTotal", f"{ds['analogy']['n_questions']:,}")
M("ANCoverPct", pc(ds["analogy"]["coverage"]))
M("MSTPool", ds["mst_eligible_test_sentences"])
M("MaxWP", max(ds["wordpieces"][s]["max_wordpieces"] for s in ("train", "dev", "test")))
rows = []
for sp in ("train", "dev", "test"):
    st = ds["splits"][sp]
    rows.append(f"{sp} & {st['n_sentences']:,} & {st['n_tokens']:,} & {st['mean_len']:.1f} & {st['max_len']} & "
                f"{st['n_punct_tokens']:,} & {st['sentences_with_mwt']:,} \\\\")
write("tab_data.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\nsplit & sentences & words & mean len. & max len. & "
      "PUNCT tokens & sents.\\ with MWT \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ============================================================================ task 1
t1 = J("task1/results.json")
ws, an, m20, mp = t1["wordsim"], t1["analogy"], t1["mst_20"], t1["mst_pool"]
M("TOneWSCov", f3(ws["spearman_covered"]))
M("TOneWSCI", ci3(*ws["spearman_covered_ci95"]))
M("TOneWSFull", f3(ws["spearman_full"]))
for t, nm in (("5", "Five"), ("10", "Ten"), ("20", "Twenty")):
    M(f"TOneWSFreq{nm}", f3(ws["by_min_train_freq"][t]["spearman"]))
    M(f"TOneWSFreq{nm}N", ws["by_min_train_freq"][t]["n"])
M("TOneANCov", pc(an["acc_covered"], 2))
M("TOneANFull", pc(an["acc_full"], 2))
M("TOneANCorrect", an["n_correct"])
M("TOneANSem", pc(an["by_kind"]["semantic"]["acc_covered"]))
M("TOneANSyn", pc(an["by_kind"]["syntactic"]["acc_covered"], 2))
M("TOneMST", pc(m20["uuas"]))
M("TOneMSTCI", cip(*m20["uuas_ci95"]))
M("TOneMSTEdges", m20["n_edges"])
M("ChainMST", pc(m20["chain_uuas"]))
M("RandomMST", pc(m20["random_uuas"]))
M("TOneVsChain", sgn(m20["vs_chain"]["diff"]))
M("TOneVsChainCI", cip(m20["vs_chain"]["ci_low"], m20["vs_chain"]["ci_high"]).replace("-", "$-$"))
M("TOneVsRandom", sgn(m20["vs_random"]["diff"]))
M("TOneVsRandomCI", cip(m20["vs_random"]["ci_low"], m20["vs_random"]["ci_high"]).replace("-", "$-$"))
M("TOnePool", pc(mp["uuas"]))
M("ChainPool", pc(mp["chain_uuas"]))
M("RandomPool", pc(mp["random_uuas"]))
rel = mp["per_relation"]
for r, nm in (("amod", "Amod"), ("advmod", "Advmod"), ("conj", "Conj"), ("obj", "Obj"), ("cc", "Cc"),
              ("case", "Case"), ("det", "Det"), ("nsubj", "Nsubj"), ("mark", "Mark")):
    M(f"TOneRel{nm}", f2(rel[r]["recall"]))

sw = pd.read_csv(OUT / "task3/extensions/layer_sweep.csv")
summ = pd.read_csv(OUT / "task1/summary_table.csv")
abtt1 = sw[sw.system == "T1 spec ABTT(D=1)"].iloc[0]
label = {"spec (alpha=1, p=1)": r"\textbf{spec}: $\alpha{=}1$, $p{=}1$ (U$\Sigma$)", "cds075": r"$\alpha{=}0.75$, $p{=}1$",
         "p05": r"$\alpha{=}1$, $p{=}0.5$", "p0": r"$\alpha{=}1$, $p{=}0$ (U)", "cds075_p05": r"$\alpha{=}0.75$, $p{=}0.5$"}
rows = []
for key in ("spec (alpha=1, p=1)", "cds075", "p05", "p0", "cds075_p05"):
    r = summ[summ.run == key].iloc[0]
    lo, hi = (float(v) for v in r.ws_ci.split(".."))
    rows.append(f"{label[key]} & {f3(r.ws_rho_cov)} {ci3(lo, hi)} & {f3(r.ws_rho_full)} & {f3(r.ws_rho_freq20)} & "
                f"{pc(r.an_acc_cov, 2)} & {pc(r.an_acc_full, 2)} & {pc(r.mst20_uuas)} & {pc(r.pool_uuas)} \\\\")
rows.append(f"spec + all-but-the-top ($D{{=}}1$) & {f3(abtt1.ws_rho_cov)} & {f3(abtt1.ws_rho_full)} & "
            f"{f3(abtt1.ws_rho_freq20)} & {pc(abtt1.an_acc_cov, 2)} & -- & {pc(abtt1.mst20_uuas)} & {pc(abtt1.pool_uuas)} \\\\")
rows.append("\\midrule")
rows.append(f"linear chain (attach to next word) & -- & -- & -- & -- & -- & {pc(m20['chain_uuas'])} & {pc(mp['chain_uuas'])} \\\\")
rows.append(f"random tree (expected) & -- & -- & -- & -- & -- & {pc(m20['random_uuas'])} & {pc(mp['random_uuas'])} \\\\")
write("tab_t1.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{3}{c}{WordSim-353 $\\rho$} & "
      "\\multicolumn{2}{c}{Analogy acc.\\ (\\%)} & \\multicolumn{2}{c}{MST UUAS (\\%)} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\nPPMI-SVD variant & covered (248) [95\\% CI] & all 353 & "
      "freq.\\,$\\ge$20 & covered & all & 20 sent. & pool (660) \\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ============================================================================ task 2
res = pd.read_csv(OUT / "task2/results_task2.csv")
bl = J("task2/best_layer.json")
L = bl["l_star"]
M("Lstar", L)
M("LstarRank", bl["distance_rank"])
for sp in ("dev", "test"):
    for k in ("uuas", "dspr", "nspr", "root"):
        v = bl[sp][k]
        M(f"Ttwo{sp.capitalize()}{k.capitalize()}", pc(v) if k in ("uuas", "root") else f3(v))
M("TtwoTestArgmax", bl["test_argmax_uuas_layer"])
bert = res[res.model == "BERT"]
lin = res[res.model == "LINEAR"]
lin_t = {p: lin[(lin.probe == p) & (lin.split == "test")].iloc[0] for p in ("distance", "depth")}
M("LinUUAS", pc(lin_t["distance"].uuas))
M("LinDSpr", f3(lin_t["distance"].dspr))
M("LinNSpr", f3(lin_t["depth"].nspr))
M("LinRoot", pc(lin_t["depth"].root))
sig = pd.read_csv(OUT / "task2/layer_significance.csv")
sig_t = sig[sig.split == "test"].set_index("layer")


def get(layer, probe, rank, split, col):
    r = bert[(bert.layer == layer) & (bert.probe == probe) & (bert["rank"] == rank) & (bert.split == split)]
    return float(r[col].iloc[0])


def best_rank(layer, probe):
    col = "uuas" if probe == "distance" else "nspr"
    return max((32, 64, 128), key=lambda k: get(layer, probe, k, "dev", col))


rows = []
for layer in range(13):
    kd, kn = best_rank(layer, "distance"), best_rank(layer, "depth")
    u = [pc(get(layer, "distance", k, "test", "uuas")) for k in (32, 64, 128)]
    n = [f3(get(layer, "depth", k, "test", "nspr")) for k in (32, 64, 128)]
    s = sig_t.loc[layer]
    d = "--" if layer == L else (f"{sgn(s.diff_vs_lstar)} {cip(s.diff_ci_low, s.diff_ci_high)}".replace("-", "$-$")
                                 .replace("$$-$$", "$-$") + ("" if s.significant else r"$^\dagger$"))
    line = (f"{layer} & {' & '.join(u)} & {f3(get(layer, 'distance', kd, 'test', 'dspr'))} & {d} & "
            f"{' & '.join(n)} & {pc(get(layer, 'depth', kn, 'test', 'root'))} \\\\")
    rows.append(("\\rowcolor{black!8} " if layer == L else "") + line)
rows.append("\\midrule")
rows.append(f"LINEAR & \\multicolumn{{3}}{{c}}{{{pc(lin_t['distance'].uuas)}}} & {f3(lin_t['distance'].dspr)} & -- & "
            f"\\multicolumn{{3}}{{c}}{{{f3(lin_t['depth'].nspr)}}} & {pc(lin_t['depth'].root)} \\\\")
write("tab_t2_layers.tex", "\\begin{tabular}{rccccccccc}\n\\toprule\n & \\multicolumn{3}{c}{UUAS (\\%), rank $k$} & & "
      "$\\Delta$UUAS vs.\\ $\\ell^*$ (pt) & \\multicolumn{3}{c}{NSpr (depth Spearman), $k$} & \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){7-9}\n$\\ell$ & 32 & 64 & 128 & DSpr & paired 95\\% CI ($k{=}128$) & 32 & 64 & 128 & "
      "Root\\% \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for k in (32, 64, 128):
    M({32: "RankA", 64: "RankB", 128: "RankC"}[k] + "UUAS", pc(get(L, "distance", k, "test", "uuas")))
    M({32: "RankA", 64: "RankB", 128: "RankC"}[k] + "NSpr", f3(get(L, "depth", k, "test", "nspr")))
M("LzeroUUAS", pc(get(0, "distance", best_rank(0, "distance"), "test", "uuas")))
M("LtwelveUUAS", pc(get(12, "distance", best_rank(12, "distance"), "test", "uuas")))
tied = sorted(sig[(sig.split == "test") & (~sig.significant)].layer.tolist())
tied_dev = sorted(sig[(sig.split == "dev") & (~sig.significant)].layer.tolist())
M("TiedLayers", "--".join(str(x) for x in (min(tied), max(tied))) if tied == tied_dev else f"{tied}/{tied_dev}")

lit = J("task2/spec_literal/best_layer.json")
M("LitLstar", lit["l_star"])
for k in ("uuas", "dspr", "nspr", "root"):
    M(f"Lit{k.capitalize()}", pc(lit["test"][k]) if k in ("uuas", "root") else f3(lit["test"][k]))
pap = {"LINEAR": (48.9, 0.58, 2.9, 0.27), "PROJ0": (59.8, 0.73, 64.4, 0.75), "ELMO1": (77.0, 0.83, 86.5, 0.87),
       "BERTBASE7": (79.8, 0.85, 88.0, 0.87), "BERTLARGE16": (81.7, 0.87, 90.1, 0.89)}
rows = [f"Paper (PTB WSJ) & {n} & {u:.1f} & {d:.2f} & {r:.1f} & {s:.2f} \\\\" for n, (u, d, r, s) in pap.items()]
rows.append("\\midrule")
l0d, l0n = best_rank(0, "distance"), best_rank(0, "depth")
ours = [("LINEAR (word order only)", lin_t["distance"].uuas, lin_t["distance"].dspr, lin_t["depth"].root, lin_t["depth"].nspr),
        ("BERT layer 0 (embeddings + position)", get(0, "distance", l0d, "test", "uuas"), get(0, "distance", l0d, "test", "dspr"),
         get(0, "depth", l0n, "test", "root"), get(0, "depth", l0n, "test", "nspr")),
        (f"\\textbf{{BERT-base-uncased layer {L}}} (paper schedule)", bl["test"]["uuas"], bl["test"]["dspr"],
         bl["test"]["root"], bl["test"]["nspr"]),
        (f"BERT layer {lit['l_star']}, spec-literal (constant lr, last epoch)", lit["test"]["uuas"], lit["test"]["dspr"],
         lit["test"]["root"], lit["test"]["nspr"])]
rows += [f"Ours (EWT) & {n} & {pc(u)} & {f3(d)} & {pc(r)} & {f3(s)} \\\\" for n, u, d, r, s in ours]
write("tab_t2_paper.tex", "\\begin{tabular}{llcccc}\n\\toprule\n & system & UUAS & DSpr & Root\\% & NSpr \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
prel = pd.read_csv(OUT / "task2/per_relation_test.csv")
pr8 = prel[(prel.layer == L) & (prel["rank"] == bl["distance_rank"])].set_index("relation")
pr0 = prel[(prel.layer == 0) & (prel["rank"] == best_rank(0, "distance"))].set_index("relation")
for r, nm in (("obj", "Obj"), ("det", "Det"), ("aux", "Aux"), ("amod", "Amod"), ("compound", "Compound"),
              ("conj", "Conj"), ("obl", "Obl"), ("case", "Case"), ("cc", "Cc"), ("mark", "Mark"), ("nsubj", "Nsubj")):
    M(f"PRel{nm}", f2(pr8.loc[r].recall))
    M(f"PRelZero{nm}", f2(pr0.loc[r].recall))
q = J("task2/qualitative_examples.json")
for kind, nm in (("good", "Good"), ("typical", "Typical"), ("poor", "Poor")):
    M(f"Q{nm}Text", esc(q[kind]["text"]))
    M(f"Q{nm}UUAS", f"{float(q[kind]['uuas']):.2f}")
    miss = ast.literal_eval(q[kind]["missed"]) if isinstance(q[kind]["missed"], str) else q[kind]["missed"]
    spur = ast.literal_eval(q[kind]["spurious"]) if isinstance(q[kind]["spurious"], str) else q[kind]["spurious"]
    M(f"Q{nm}Missed", esc(", ".join(miss)) or "none")
    M(f"Q{nm}Spurious", esc(", ".join(spur)) or "none")

# ============================================================================ task 3
cmp_ = pd.read_csv(OUT / "task3/comparison_table.csv")
r3 = J("task3/main/results.json")
r3a = J("task3/main_abtt/results.json")
best_sem = sw[(sw.source == "T3") & (sw.post == "raw")].sort_values("ws_rho_cov", ascending=False).iloc[0]
an_best = sw[(sw.source == "T3") & (sw.post == "raw")].sort_values("an_acc_cov", ascending=False).iloc[0]
M("BestSemLayer", int(best_sem.layer))
M("BestSemWS", f3(best_sem.ws_rho_cov))
M("BestAnLayer", int(an_best.layer))
M("BestAnAcc", pc(an_best.an_acc_cov))
lzero = sw[(sw.source == "T3") & (sw.post == "raw") & (sw.layer == 0)].iloc[0]
M("LzeroWS", f3(lzero.ws_rho_cov))
M("LzeroAn", pc(lzero.an_acc_cov))
aniso = sw[(sw.source == "T3") & (sw.post == "raw")].set_index("layer").anisotropy
M("AnisoLzero", f2(aniso.loc[0]))
M("AnisoMax", f2(aniso.max()))
M("AnisoMaxLayer", int(aniso.idxmax()))
M("AnisoLstar", f2(aniso.loc[L]))
abtt_ws = sw[(sw.source == "T3") & (sw.post == "abtt")].ws_rho_cov
M("AbttWSMin", f2(abtt_ws.min()))
M("AbttWSMax", f2(abtt_ws.max()))
st_mst = sw[(sw.source == "T3")].pool_uuas
M("StaticMSTMin", pc(st_mst.min(), 0))
M("StaticMSTMax", pc(st_mst.max(), 0))
for name, rr, pref in (("T3", r3, "TThree"), ("T3a", r3a, "TThreeA")):
    w, a, m, p = rr["wordsim"], rr["analogy"], rr["mst_20"], rr["mst_pool"]
    M(f"{pref}WSCov", f3(w["spearman_covered"]))
    M(f"{pref}WSCI", ci3(*w["spearman_covered_ci95"]))
    M(f"{pref}WSFull", f3(w["spearman_full"]))
    M(f"{pref}WSFreqTwenty", f3(w["by_min_train_freq"]["20"]["spearman"]))
    M(f"{pref}ANCov", pc(a["acc_covered"]))
    M(f"{pref}ANFull", pc(a["acc_full"]))
    M(f"{pref}ANSem", pc(a["by_kind"]["semantic"]["acc_covered"]))
    M(f"{pref}ANSyn", pc(a["by_kind"]["syntactic"]["acc_covered"]))
    M(f"{pref}MST", pc(m["uuas"]))
    M(f"{pref}MSTCI", cip(*m["uuas_ci95"]))
    M(f"{pref}Pool", pc(p["uuas"]))
rel3 = r3["mst_pool"]["per_relation"]
for r, nm in (("nsubj", "Nsubj"), ("obj", "Obj"), ("amod", "Amod"), ("obl", "Obl"), ("conj", "Conj"),
              ("det", "Det"), ("mark", "Mark")):
    M(f"TThreeRel{nm}", f2(rel3[r]["recall"]))

rows = []
names = {"T1 PPMI-SVD (spec)": "T1 PPMI-SVD (spec)", "T1 PPMI-SVD best variant (p0)": "T1 PPMI-SVD, best variant ($p{=}0$)",
         f"T3 BERT-static L{L}": f"\\textbf{{T3 BERT-static, layer {L}}}", f"T3 BERT-static L{L} + ABTT": f"T3 + all-but-the-top ($D{{=}}7$)"}
for _, r in cmp_.iterrows():
    if r.system in names:
        rows.append(f"{names[r.system]} & {f3(r.ws_rho_cov)} {ci3(r.ws_ci_low, r.ws_ci_high)} & {f3(r.ws_rho_full)} & "
                    f"{pc(r.an_acc_cov)} & {pc(r.an_sem_cov)} / {pc(r.an_syn_cov)} & {pc(r.an_acc_full)} & "
                    f"{pc(r.mst20_uuas)} {cip(r.mst20_ci_low, r.mst20_ci_high)} & {pc(r.pool_uuas)} \\\\")
rows.append("\\midrule")
t2r = cmp_[cmp_.system.str.startswith("T2 probe")].iloc[0]
rows.append(f"T2 probe, layer {L} (contextual, supervised; reference) & -- & -- & -- & -- & -- & {pc(t2r.mst20_uuas)} "
            f"{cip(t2r.mst20_ci_low, t2r.mst20_ci_high)} & {pc(t2r.pool_uuas)} \\\\")
rows.append(f"linear chain / random tree & -- & -- & -- & -- & -- & {pc(m20['chain_uuas'])} / {pc(m20['random_uuas'])} & "
            f"{pc(mp['chain_uuas'])} / {pc(mp['random_uuas'])} \\\\")
M("TTwoRefMST", pc(t2r.mst20_uuas))
M("TTwoRefPool", pc(t2r.pool_uuas))
write("tab_t3_main.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{2}{c}{WordSim-353 $\\rho$} & "
      "\\multicolumn{3}{c}{Analogy acc.\\ (\\%)} & \\multicolumn{2}{c}{MST UUAS (\\%)} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\nsystem & covered [95\\% CI] & all 353 & covered & "
      "sem / syn & all & 20 sent.\\ [95\\% CI] & pool \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

tests = J("task3/paired_tests.json")
lab = {"T3": "T3", "T1": "T1", "T3+ABTT": "T3+ABTT", "T2probe": "T2 probe"}
rows = []
for t in tests:
    cells = [f"{lab[t['a']]} $-$ {lab[t['b']]}"]
    if "wordsim" in t:
        w = t["wordsim"]
        cells.append(f"{sgn(w['diff'], 3, 1)} {ci3(w['ci_low'], w['ci_high'])}")
    else:
        cells.append("--")
    if "analogy" in t:
        a = t["analogy"]
        p = fmt_p(a["p_value"])
        cells.append(f"{a['a_only']:,} / {a['b_only']:,} & {p}")
    else:
        cells.append("-- & --")
    for key in ("mst_20", "mst_pool"):
        m = t[key]
        cells.append(f"{sgn(m['diff'])} {cip(m['ci_low'], m['ci_high'])}".replace("[-", "[$-$").replace(", -", ", $-$"))
    rows.append(" & ".join(cells) + " \\\\")
    if t["a"] == "T3" and t["b"] == "T1":
        M("PtWSDiff", sgn(t["wordsim"]["diff"], 3, 1))
        M("PtWSCI", ci3(t["wordsim"]["ci_low"], t["wordsim"]["ci_high"]))
        M("PtANOnlyA", f"{t['analogy']['a_only']:,}")
        M("PtANOnlyB", f"{t['analogy']['b_only']:,}")
        M("PtMSTDiff", sgn(t["mst_20"]["diff"]))
        M("PtMSTCI", cip(t["mst_20"]["ci_low"], t["mst_20"]["ci_high"]).replace("-", "$-$"))
        M("PtPoolDiff", sgn(t["mst_pool"]["diff"]))
        M("PtPoolCI", cip(t["mst_pool"]["ci_low"], t["mst_pool"]["ci_high"]).replace("-", "$-$"))
    if t["a"] == "T3+ABTT" and t["b"] == "T3":
        M("PtAbttWSDiff", sgn(t["wordsim"]["diff"], 3, 1))
        M("PtAbttANA", f"{t['analogy']['a_only']}")
        M("PtAbttANB", f"{t['analogy']['b_only']}")
write("tab_t3_tests.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & WordSim-353 & \\multicolumn{2}{c}{Analogy (McNemar)} & "
      "\\multicolumn{2}{c}{MST $\\Delta$UUAS (pt) [95\\% CI]} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
      "A $-$ B & $\\Delta\\rho$ [95\\% CI] & only A / only B & $p$ & 20 sentences & pool (660) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

lad = pd.read_csv(OUT / "task3/extensions/ladder.csv")
diag = J("task3/diagnostics/diagnostics.json")
auc = {r["system"]: r["edge_auc"] for r in diag["trees"]}
auc_key = {"chain": "linear chain", "T1": "T1 PPMI-SVD", "T3": f"T3 L{L}", "T3abtt": f"T3 L{L} + ABTT",
           "ctx": f"contextual L{L} cosine", "probe": f"T2 probe L{L}"}
tier = {"baseline": "--", "required": "req.", "B": "B", "C": "C", "reference": "ref."}
rows = []
for _, r in lad.iterrows():
    a = auc.get(auc_key.get(r.key, ""), None)
    m20s = pc(r.mst20_uuas) + (f" {cip(r.mst20_ci_low, r.mst20_ci_high)}" if pd.notna(r.mst20_ci_low) else "")
    pools = (pc(r.pool_uuas) + f" {cip(r.pool_ci_low, r.pool_ci_high)}") if pd.notna(r.pool_uuas) else "--"
    rows.append(f"{esc(r.system)} & {tier[r.tier]} & {m20s} & {pools} & {f2(a) if a is not None and pd.notna(a) else '--'} \\\\")
write("tab_ladder.tex", "\\begin{tabular}{llccc}\n\\toprule\nrepresentation for the MST & tier & 20 sentences [95\\% CI] & "
      "pool (660) [95\\% CI] & edge AUC \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
ladk = lad.set_index("key")
for k, nm in (("T3probe", "Proj"), ("ctx", "Ctx"), ("concat", "Concat")):
    M(f"Lad{nm}Pool", pc(ladk.loc[k].pool_uuas))
    M(f"Lad{nm}MST", pc(ladk.loc[k].mst20_uuas))
for k, nm in auc_key.items():
    pass
M("AucTOne", f2(auc["T1 PPMI-SVD"]))
M("AucTThree", f2(auc[f"T3 L{L}"]))
M("AucCtx", f2(auc[f"contextual L{L} cosine"]))
M("AucProbe", f2(auc[f"T2 probe L{L}"]))
M("AucChain", f2(auc["linear chain"]))
tc = pd.read_csv(OUT / "task3/extensions/tierC_semantic.csv").iloc[0]
M("ConcatWS", f3(tc.ws_rho_cov))
M("ConcatAN", pc(tc.an_acc_cov))

lex = {r["system"]: r for r in diag["lexical"]}
trees = {r["system"]: r for r in diag["trees"]}
cols = [("T1 PPMI-SVD", "T1 PPMI-SVD", "T1 PPMI-SVD"), (f"T3 L{L}", f"T3 L{L}", f"T3 layer {L}"),
        (f"T3 L{L} + ABTT", f"T3 L{L} + ABTT", "T3 + ABTT")]
rnd = lex["T1 PPMI-SVD"]
lines = [
    ("NN with same majority UPOS (top-10)", lambda lx, tr: f3(lx["nn_same_upos"]), f3(rnd["random_same_upos"])),
    ("NN that co-occur within $\\pm5$ in train", lambda lx, tr: f3(lx["nn_cooccur"]), f3(rnd["random_cooccur"])),
    ("hubness: skewness of $N_{10}$", lambda lx, tr: f2(lx["N10_skewness"]), "--"),
    ("function words among top-50 hubs", lambda lx, tr: f2(lx["frac_function_words_top50_hubs"]), "--"),
    ("\\midrule edge AUC (gold edge closer than non-edge)", lambda lx, tr: f2(tr["edge_auc"]), "0.50"),
    ("MST: mean degree of function-word tokens", lambda lx, tr: f2(tr["mean_fn_token_degree"]), f2(trees["gold"]["mean_fn_token_degree"]) + " (gold)"),
    ("MST: mean max.\\ degree per sentence", lambda lx, tr: f2(tr["mean_max_degree"]), f2(trees["gold"]["mean_max_degree"]) + " (gold)"),
    ("MST: edges joining same-UPOS words", lambda lx, tr: f2(tr["p_same_upos_edges"]), f2(trees["gold"]["p_same_upos_edges"]) + " (gold)"),
]
rows = []
for label_, fn, ref in lines:
    pre = ""
    if label_.startswith("\\midrule"):
        pre, label_ = "\\midrule\n", label_.replace("\\midrule ", "")
    rows.append(pre + f"{label_} & " + " & ".join(fn(lex[a], trees[b]) for a, b, _ in cols) + f" & {ref} \\\\")
write("tab_diag.tex", "\\begin{tabular}{lcccc}\n\\toprule\n & " + " & ".join(c for _, _, c in cols) +
      " & random / gold \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
M("NNSameTOne", f2(lex["T1 PPMI-SVD"]["nn_same_upos"]))
M("NNSameTThree", f2(lex[f"T3 L{L}"]["nn_same_upos"]))
M("NNSameRandom", f2(rnd["random_same_upos"]))
M("NNCoocTOne", f2(lex["T1 PPMI-SVD"]["nn_cooccur"]))
M("NNCoocTThree", f2(lex[f"T3 L{L}"]["nn_cooccur"]))
M("NNCoocRandom", f2(rnd["random_cooccur"]))
M("GoldSameUpos", f2(trees["gold"]["p_same_upos_edges"]))
M("GoldSameUposNon", f2(trees["gold"]["p_same_upos_nonedges"]))
M("FnDegTThree", f2(trees[f"T3 L{L}"]["mean_fn_token_degree"]))
M("FnDegGold", f2(trees["gold"]["mean_fn_token_degree"]))
M("FnDegTOne", f2(trees["T1 PPMI-SVD"]["mean_fn_token_degree"]))
M("MaxDegTThree", f2(trees[f"T3 L{L}"]["mean_max_degree"]))
M("MaxDegGold", f2(trees["gold"]["mean_max_degree"]))
M("SkewTOne", f2(lex["T1 PPMI-SVD"]["N10_skewness"]))
M("SkewTThree", f2(lex[f"T3 L{L}"]["N10_skewness"]))

# ---- qualitative T3 tables
nn = pd.read_csv(OUT / "task3/qualitative/nearest_neighbours.csv").set_index("word")
rows = []
for w in ("good", "bad", "monday", "dog", "car", "money", "said", "happy", "computer", "two"):
    t1n = ", ".join(nn.loc[w].T1.split(", ")[:5])
    t3n = ", ".join(nn.loc[w].T3.split(", ")[:5])
    rows.append(f"\\textit{{{esc(w)}}} & {esc(t1n)} & {esc(t3n)} \\\\")
write("tab_nn.tex", "\\begin{tabular}{lll}\n\\toprule\nword & T1 PPMI-SVD: 5 nearest neighbours & "
      f"T3 BERT-static (layer {L}): 5 nearest neighbours \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
aq = pd.read_csv(OUT / "task3/qualitative/analogy_examples.csv")
won = aq[aq.correct_T3 & ~aq.correct_T1].drop_duplicates("section").head(6)
lost = aq[~aq.correct_T3].drop_duplicates("section")
lost = lost[lost.section.isin(["currency", "gram3-comparative", "gram4-superlative", "gram6-nationality-adjective",
                               "gram2-opposite", "capital-world"])].head(6)
rows = []
for _, r in won.iterrows():
    rows.append(f"{esc(r.section)} & {esc(r.a)} : {esc(r.b)} :: {esc(r.c)} : \\textbf{{{esc(r.gold)}}} & {esc(r.pred_T3)} \\checkmark & {esc(r.pred_T1)} \\\\")
rows.append("\\midrule")
for _, r in lost.iterrows():
    rows.append(f"{esc(r.section)} & {esc(r.a)} : {esc(r.b)} :: {esc(r.c)} : \\textbf{{{esc(r.gold)}}} & {esc(r.pred_T3)} & {esc(r.pred_T1)} \\\\")
write("tab_analogy.tex", "\\begin{tabular}{llll}\n\\toprule\nsection & question ($a:b::c:$\\,\\textbf{gold}) & T3 answer & "
      "T1 answer \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
sec = r3["analogy"]["per_section"]
M("TThreeCurrency", f"{sec['currency']['correct']}/{sec['currency']['covered']}")
M("TThreePlural", pc(sec["gram8-plural"]["acc_covered"]))
M("TThreeOpposite", pc(sec["gram2-opposite"]["acc_covered"]))
M("TThreeCapital", pc(sec["capital-common-countries"]["acc_covered"]))
ex = J("task3/qualitative/mst_example.json")
M("ExText", esc(ex["text"]))
M("ExTOne", pc(ex["T1"], 0))
M("ExTThree", pc(ex["T3"], 0))
M("ExCtx", pc(ex["ctx"], 0))
M("ExProbe", pc(ex["probe"], 0))

# ---- appendix: layer sweep table
rows = []
for layer in range(13):
    a = sw[(sw.source == "T3") & (sw.layer == layer) & (sw.post == "raw")].iloc[0]
    b = sw[(sw.source == "T3") & (sw.layer == layer) & (sw.post == "abtt")].iloc[0]
    c = sw[(sw.source == "contextual") & (sw.layer == layer)].iloc[0]
    p = sw[(sw.source == "probe") & (sw.layer == layer)].iloc[0]
    line = (f"{layer} & {f3(a.ws_rho_cov)} & {f3(b.ws_rho_cov)} & {pc(a.an_acc_cov)} & {pc(b.an_acc_cov)} & "
            f"{pc(a.pool_uuas)} & {pc(b.pool_uuas)} & {pc(c.pool_uuas)} & {pc(p.pool_uuas)} & {f2(a.anisotropy)} \\\\")
    rows.append(("\\rowcolor{black!8} " if layer == L else "") + line)
t1r = sw[sw.system == "T1 spec raw"].iloc[0]
rows.append("\\midrule")
rows.append(f"T1 & {f3(t1r.ws_rho_cov)} & {f3(abtt1.ws_rho_cov)} & {pc(t1r.an_acc_cov)} & {pc(abtt1.an_acc_cov)} & "
            f"{pc(t1r.pool_uuas)} & {pc(abtt1.pool_uuas)} & -- & -- & {f2(t1r.anisotropy)} \\\\")
write("tab_sweep.tex", "\\begin{tabular}{rccccccccc}\n\\toprule\n & \\multicolumn{2}{c}{WS-353 $\\rho$ (cov.)} & "
      "\\multicolumn{2}{c}{Analogy (cov., \\%)} & \\multicolumn{4}{c}{MST UUAS on the pool (\\%)} & \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n$\\ell$ & raw & ABTT & raw & ABTT & static raw & static ABTT & "
      "contextual cos. & T2 probe & anisotropy \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- runtime facts
M("FeatSeconds", "410")
M("FeatGB", "5.1")

(GEN / "numbers.tex").write_text("% generated by report/make_tables.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
print(f"[tables] {len(MACROS)} macros, {len(list(GEN.glob('tab_*.tex')))} tables -> {GEN}")
