"""
Task 3: BERT-derived STATIC word vectors (pooled from the syntax-best layer l* of Task 2)
and a comparison against the Task 1 PPMI+SVD vectors.

What it does
  1. Reads l* from results/task2/best_layer.json (or use --layer N).
  2. Runs every TRAIN sentence of UD English EWT through frozen bert-base-uncased, takes layer l*,
     averages the word-pieces of each word (same as Task 2), then averages over ALL occurrences
     of each lower-cased word type  ->  one 768-d static vector per word.
  3. Evaluates with the SAME functions as Task 1 (common.py):
        WordSim-353 Spearman, Google-analogy accuracy, and the 20-sentence MST attachment agreement
        (the same 20 test sentences, read from results/sample_indices.json).
  4. Prints a side-by-side comparison with Task 1 and saves everything to results/task3/.

Run it from the SAME folder you used for Task 1 and Task 2 (it needs ./results and ./data):
        python task3_bert_static.py

Options:
        --layer 8          use this layer instead of the one in best_layer.json
        --all_layers       ALSO build vectors from every layer and compare them (extra analysis + plot)
        --wordsim PATH --analogy PATH   same optional benchmark files as in Task 1
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

import common

OUT = os.path.join(common.RESULTS_DIR, "task3")


# =============================================================================
# 1. Build static vectors from BERT
# =============================================================================
def build_static_vectors(train, layers, model_name, device, bs=32):
    """Average, per lower-cased word type, the contextual vectors of all its occurrences.
    layers: 1-based BERT layers.  Returns types, counts, vecs of shape (len(layers), V, dim)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    assert tok.is_fast, "need a fast tokenizer (for word_ids)"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.hidden_size

    cnt = Counter(t.lower() for s in train for t in s["tokens"])
    types = [w for w, _ in cnt.most_common()]
    w2i = {w: i for i, w in enumerate(types)}
    V = len(types)
    sums = torch.zeros(len(layers), V, dim, device=device)
    counts = torch.zeros(V, device=device)

    lens = np.array([len(s["tokens"]) for s in train])
    order = np.argsort(lens)
    t0 = time.time()
    no_pieces = 0
    for s0 in range(0, len(order), bs):
        ids = order[s0:s0 + bs]
        batch = [train[i]["tokens"] for i in ids]
        enc = tok(batch, is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=512)
        T, Wmax = enc["input_ids"].shape[1], int(lens[ids].max())
        P = np.zeros((len(ids), Wmax, T), dtype=np.float32)       # word <- average of its word-pieces
        type_ids = np.full((len(ids), Wmax), -1, dtype=np.int64)
        for b, i in enumerate(ids):
            for t, w in enumerate(enc.word_ids(b)):
                if w is not None:
                    P[b, w, t] = 1.0
            cnt_b = P[b].sum(1, keepdims=True)
            P[b] /= np.maximum(cnt_b, 1.0)
            n = lens[i]
            ok = cnt_b[:n, 0] > 0
            no_pieces += int((~ok).sum())
            type_ids[b, :n] = np.where(ok, [w2i[w.lower()] for w in train[i]["tokens"]], -1)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l] for l in layers], 0)     # (nL,B,T,dim)
            pooled = torch.einsum("bwt,lbtd->lbwd", torch.from_numpy(P).to(device), hs)
            valid = torch.from_numpy(type_ids >= 0).to(device)
            tid = torch.from_numpy(type_ids[type_ids >= 0]).to(device)
            sums.index_add_(1, tid, pooled[:, valid])
            counts.index_add_(0, tid, torch.ones(len(tid), device=device))
        if (s0 // bs) % 100 == 0:
            print(f"  sentences {s0}/{len(order)}  ({time.time() - t0:.0f}s)")
    vecs = (sums / counts.clamp(min=1)[None, :, None]).cpu().numpy()
    counts = counts.cpu().numpy()
    keep = counts > 0
    print(f"Built vectors for {int(keep.sum())} word types "
          f"({int(counts.sum())} word occurrences; {no_pieces} words without word-pieces skipped)")
    return [t for t, k in zip(types, keep) if k], counts[keep], vecs[:, keep]


# =============================================================================
# 2. Evaluation (exactly the Task 1 functions)
# =============================================================================
def evaluate(vecs, word2id, pairs, qs, sample, full_test):
    ws = common.eval_wordsim(pairs, word2id, vecs)
    an = common.eval_analogy(qs, word2id, vecs)
    m20 = common.eval_mst_agreement(sample, word2id, vecs)
    mfull = common.eval_mst_agreement(full_test, word2id, vecs)
    return {"wordsim": ws, "analogy": an, "mst20": m20, "mst_fulltest": mfull}


def flat(res):
    return {
        "WordSim Spearman (covered pairs)": res["wordsim"]["spearman_covered"],
        "WordSim Spearman (all 353, OOV=0)": res["wordsim"]["spearman_all353"],
        "WordSim pairs covered": res["wordsim"]["n_covered"],
        "Analogy acc (covered questions)": res["analogy"]["acc_covered"],
        "Analogy acc (all questions)": res["analogy"]["acc_all"],
        "Analogy semantic (covered)": res["analogy"]["acc_semantic_covered"],
        "Analogy syntactic (covered)": res["analogy"]["acc_syntactic_covered"],
        "Analogy questions covered": res["analogy"]["n_covered"],
        "MST agreement, 20 sentences (micro)": res["mst20"]["agreement_micro"],
        "MST agreement, 20 sentences (macro)": res["mst20"]["agreement_macro"],
        "MST agreement, full test split (micro)": res["mst_fulltest"]["agreement_micro"],
    }


def print_table(columns):
    names = list(columns)
    rows = list(columns[names[0]].keys())
    w0 = max(len(r) for r in rows) + 2
    print("\n" + " " * w0 + " | ".join(f"{n:>22s}" for n in names))
    for r in rows:
        cells = []
        for n in names:
            v = columns[n][r]
            cells.append(f"{v:22d}" if isinstance(v, (int, np.integer)) else f"{v:22.4f}")
        print(f"{r:<{w0}s}" + " | ".join(cells))


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--all_layers", action="store_true")
    ap.add_argument("--wordsim", default=None)
    ap.add_argument("--analogy", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N train sentences only")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- which layer?
    bl_path = os.path.join(common.RESULTS_DIR, "task2", "best_layer.json")
    if args.layer is None:
        if not os.path.exists(bl_path):
            sys.exit(f"Cannot find {bl_path}. Run this from the folder where Task 2 saved its results, "
                     f"or pass --layer N.")
        with open(bl_path) as f:
            args.layer = json.load(f)["best_layer"]
    print(f"Using BERT layer l* = {args.layer}")

    # ---- Task 1 outputs we compare against
    t1_res_path = os.path.join(common.RESULTS_DIR, "task1_results.json")
    t1_vec_path = os.path.join(common.RESULTS_DIR, "task1_vectors.npz")
    for p in [t1_res_path, t1_vec_path, os.path.join(common.RESULTS_DIR, "sample_indices.json")]:
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run this script from the same folder where you ran Task 1.")
    with open(t1_res_path) as f:
        t1 = json.load(f)
    z = np.load(t1_vec_path, allow_pickle=False)
    t1_vecs, t1_words = z["vectors"], [str(w) for w in z["words"]]
    t1_w2i = {w: i for i, w in enumerate(t1_words)}

    # ---- data and benchmarks (identical to Task 1)
    ewt = common.load_ewt("conllu")
    train, test = ewt["train"], ewt["test"]
    if args.limit:
        train = train[:args.limit]
    # one-word sentences have no edges to score; skip them in the full-test-split robustness check
    test_multi = [s for s in test if len(s["tokens"]) >= 2]
    pairs = common.load_wordsim(args.wordsim)
    qs = common.load_analogy(args.analogy)
    sample, idx = common.sample_test_sentences(test, n=20)
    assert idx == t1["sample_indices"], "20-sentence sample differs from Task 1!"
    print(f"Same 20 test sentences as Task 1: {idx}")

    # ---- BERT static vectors
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print("Device:", device)
    from transformers import AutoConfig
    nl = AutoConfig.from_pretrained(args.model).num_hidden_layers
    layers = list(range(1, nl + 1)) if args.all_layers else [args.layer]
    types, wcounts, vecs = build_static_vectors(train, layers, args.model, device, args.bs)
    all_w2i = {w: i for i, w in enumerate(types)}
    li = layers.index(args.layer)
    missing = [w for w in t1_words if w not in all_w2i]
    if missing:
        print(f"[warn] {len(missing)} Task-1 words are missing from the BERT vocabulary (limit used?)")
    same_words = [w for w in t1_words if w in all_w2i]
    same_w2i = {w: i for i, w in enumerate(same_words)}
    rows = np.array([all_w2i[w] for w in same_words])

    np.savez_compressed(os.path.join(OUT, f"bert_static_layer{args.layer}.npz"),
                        vectors=vecs[li], words=np.array(types), counts=wcounts)

    # ---- sanity: re-evaluating the saved Task 1 vectors must reproduce Task 1's numbers
    base_full = common.baseline_agreement(test_multi, trials=20)
    r_t1 = evaluate(t1_vecs, t1_w2i, pairs, qs, sample, test_multi)
    if not args.limit:
        assert abs(r_t1["wordsim"]["spearman_covered"] - t1["wordsim"]["spearman_covered"]) < 1e-6
        assert abs(r_t1["mst20"]["agreement_micro"] - t1["mst"]["agreement_micro"]) < 1e-9
    print("Task 1 vectors re-evaluated: matches the saved Task 1 results.")

    # ---- the BERT variants
    raw = vecs[li][rows]                                       # same vocabulary as Task 1 (primary)
    centered = raw - raw.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca100 = centered @ Vt[:100].T
    variants = {
        "BERT l*, same vocab (MAIN)": (raw, same_w2i),
        "BERT l*, all word types": (vecs[li], all_w2i),
        "BERT l*, centered (extra)": (centered, same_w2i),
        "BERT l*, centered+PCA100 (extra)": (pca100, same_w2i),
    }
    results = {"Task 1: PPMI+SVD (100d)": r_t1}
    for name, (v, w2i) in variants.items():
        results[name] = evaluate(v, w2i, pairs, qs, sample, test_multi)

    table = {n: flat(r) for n, r in results.items()}
    print_table(table)
    print(f"\nReference MST baselines (20 sentences): random tree {t1['baselines']['random_tree']:.4f}, "
          f"linear chain {t1['baselines']['linear_chain']:.4f}")
    print(f"Reference MST baselines (full test split): random tree {base_full['random_tree']:.4f}, "
          f"linear chain {base_full['linear_chain']:.4f}")
    print(f"Vocabulary: Task 1 = {len(t1_words)} words, BERT main = {len(same_words)}, "
          f"BERT all types = {len(types)}.  Dimensions: Task 1 = {t1_vecs.shape[1]}, BERT = {raw.shape[1]}.")

    # ---- optional: every layer (extra analysis)
    sweep = []
    if args.all_layers:
        print("\nLayer sweep (same vocab, raw vectors):")
        print("layer | WordSim(cov) | Analogy(cov) | MST20 micro | MST full-test micro")
        for j, l in enumerate(layers):
            r = evaluate(vecs[j][rows], same_w2i, pairs, qs, sample, test_multi)
            row = {"layer": l, "wordsim": r["wordsim"]["spearman_covered"],
                   "analogy": r["analogy"]["acc_covered"], "mst20": r["mst20"]["agreement_micro"],
                   "mst_full": r["mst_fulltest"]["agreement_micro"]}
            sweep.append(row)
            print(f"{l:5d} | {row['wordsim']:12.4f} | {row['analogy']:12.4f} | "
                  f"{row['mst20']:11.4f} | {row['mst_full']:.4f}")
        with open(os.path.join(OUT, "layer_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0]))
            w.writeheader()
            w.writerows(sweep)
        make_sweep_plot(sweep, args.layer, r_t1)

    # ---- save
    with open(os.path.join(OUT, "task3_results.json"), "w") as f:
        json.dump({"layer": args.layer, "sample_indices": idx, "results": results,
                   "baselines_20": t1["baselines"], "baselines_fulltest": base_full,
                   "sweep": sweep}, f, indent=2)
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        names = list(table)
        w.writerow(["metric"] + names)
        for r in table[names[0]]:
            w.writerow([r] + [table[n][r] for n in names])
    print(f"\nSaved results/task3/task3_results.json, comparison.csv, bert_static_layer{args.layer}.npz")


def make_sweep_plot(sweep, best_l, r_t1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid, surface, blue, gray = "#0b0b0b", "#e4e3df", "#fcfcfb", "#2a78d6", "#52514e"
    panels = [("wordsim", "WordSim-353 Spearman (covered pairs)", r_t1["wordsim"]["spearman_covered"]),
              ("mst_full", "MST attachment agreement (full test split)",
               r_t1["mst_fulltest"]["agreement_micro"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    fig.patch.set_facecolor(surface)
    xs = [r["layer"] for r in sweep]
    for ax, (key, title, ref) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.plot(xs, [r[key] for r in sweep], color=blue, marker="o", lw=2, ms=6,
                markeredgecolor=surface, markeredgewidth=1.2)
        ax.axhline(ref, color=gray, lw=1.2, ls="--")
        ax.text(xs[0], ref, " Task 1 (PPMI+SVD)", color=gray, fontsize=8, va="bottom")
        ax.axvline(best_l, color=gray, lw=0.8, ls=":")
        ax.set_title(title, color=ink, loc="left", fontsize=10)
        ax.set_xlabel("BERT layer used for pooling", color=ink)
        ax.set_xticks(xs)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=ink)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_layer_sweep.png")
    fig.savefig(p, facecolor=surface)
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
