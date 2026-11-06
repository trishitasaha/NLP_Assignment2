"""One evaluation suite for *static* word vectors, shared by Task 1 (PPMI+SVD) and Task 3
(BERT-pooled). Calling the same function guarantees the "identical protocol" that Task 3
requires.

evaluate_static_vectors() computes:
  1. WordSim-353: Spearman (covered + full, median imputation) with a bootstrap 95% CI
  2. Google analogy: 3CosAdd accuracy (covered + full; per section; semantic vs syntactic)
  3. 20-sentence MST check: cosine-distance MST vs gold tree (UUAS, punctuation excluded),
     with linear-chain and random-tree baselines, bootstrap CI, paired bootstrap vs the chain,
     and a per-dependency-relation recall breakdown
  4. The same MST check on the whole eligible test pool (660 sentences): a more stable estimate
  5. Nearest neighbours of a fixed list of probe words (qualitative)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import benchmarks as B
from . import config as C
from .data import Sentence, gold_edges, load_ewt, mst_eligible, sample_mst_sentences, word_counts
from .stats import bootstrap_spearman, bootstrap_uuas, paired_bootstrap_uuas
from .trees import (cosine_distance_matrix, gold_mst_edges, linear_chain_dist, mst_on_subset,
                    random_tree_edges)

PROBE_WORDS = ["good", "bad", "monday", "dog", "car", "money", "food", "said", "president",
               "google", "happy", "computer", "city", "school", "doctor", "two", "the", "of"]
N_RANDOM_TREES = 100  # random-tree baseline: expected UUAS over this many draws per sentence


# --------------------------------------------------------------------------- the shared 20-sentence sample
def get_mst_sentences() -> tuple[list[Sentence], list[Sentence]]:
    """Return (the 20 sampled test sentences, all eligible test sentences).

    The sample is created once (seed 42, decision D10) and saved to
    outputs/shared/mst_sample_ids.json. Task 3 reloads exactly the same ids.
    """
    train, test = load_ewt("train"), load_ewt("test")
    vocab = set(word_counts(train))  # lowercased train types == the T1/T3 vocabulary
    pool = [s for s in test if mst_eligible(s, vocab)]
    path = C.MST_SAMPLE_PATH
    if path.exists():
        ids = json.loads(path.read_text())["sent_ids"]
    else:
        ids = sample_mst_sentences(test, vocab)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            sent_ids=ids, seed=C.SEED, n_eligible=len(pool),
            criteria=f"test split; {C.MST_MIN_WORDS}-{C.MST_MAX_WORDS} non-PUNCT words; "
                     "all non-PUNCT words (lowercased) in the train vocabulary"), indent=2))
        print(f"[static_eval] sampled {len(ids)} MST sentences -> {path}")
    by_id = {s.sent_id: s for s in test}
    return [by_id[i] for i in ids], pool


# --------------------------------------------------------------------------- MST check
def _sentence_vectors(s: Sentence, vectors: np.ndarray, w2i: dict[str, int]) -> np.ndarray:
    """Static vector per token. Non-punct words are guaranteed in-vocab by the sampling;
    OOV punctuation (excluded from the MST anyway) gets a zero vector."""
    out = np.zeros((len(s), vectors.shape[1]), dtype=np.float64)
    for i, w in enumerate(s.lower):
        if w in w2i:
            out[i] = vectors[w2i[w]]
    return out


def mst_check(sentences: list[Sentence], dist_fn, name: str, seed: int = C.SEED,
              n_random: int = N_RANDOM_TREES) -> dict:
    """Generic structural check. dist_fn(sentence) -> [n, n] predicted distance matrix.
    n_random = 0 skips the (slow) random-tree baseline, e.g. inside layer sweeps.

    Returns aggregate UUAS (+CI), the baselines, a paired test vs the chain, per-relation
    recall and per-sentence records (with edges, for the qualitative figures).
    """
    rng = np.random.default_rng(seed)
    per_sent, rel = [], defaultdict(lambda: [0, 0])  # deprel -> [recovered, total]
    for s in sentences:
        keep = s.nonpunct_mask
        gold = gold_mst_edges(s.heads, keep)
        pred = mst_on_subset(dist_fn(s), keep)
        chain = mst_on_subset(linear_chain_dist(len(s)), keep)
        idx = np.flatnonzero(keep)
        rand_correct = np.mean([
            len(gold & {(int(idx[a]), int(idx[b])) for a, b in random_tree_edges(len(idx), rng)})
            for _ in range(n_random)]) if n_random else float("nan")
        for i, h in enumerate(s.heads):  # per-relation recall on the true head edges
            if h > 0 and keep[i] and keep[h - 1]:
                e = tuple(sorted((i, h - 1)))
                r = s.deprels[i].split(":")[0]
                rel[r][1] += 1
                rel[r][0] += e in pred
        per_sent.append(dict(
            sent_id=s.sent_id, text=s.text, n_words=int(keep.sum()), total=len(gold),
            correct=len(gold & pred), chain_correct=len(gold & chain), random_correct=float(rand_correct),
            gold_edges=sorted(gold), pred_edges=sorted(pred), tokens=s.tokens))

    corr = np.array([p["correct"] for p in per_sent])
    chain_c = np.array([p["chain_correct"] for p in per_sent])
    rand_c = np.array([p["random_correct"] for p in per_sent])
    tot = np.array([p["total"] for p in per_sent])
    return dict(
        name=name, n_sentences=len(per_sent), n_edges=int(tot.sum()),
        uuas=float(corr.sum() / tot.sum()), uuas_ci95=bootstrap_uuas(corr, tot),
        chain_uuas=float(chain_c.sum() / tot.sum()), chain_ci95=bootstrap_uuas(chain_c, tot),
        random_uuas=float(rand_c.sum() / tot.sum()),
        vs_chain=paired_bootstrap_uuas(corr, chain_c, tot),
        vs_random=paired_bootstrap_uuas(corr, rand_c, tot) if n_random else None,
        per_relation={k: dict(recovered=v[0], total=v[1], recall=v[0] / v[1])
                      for k, v in sorted(rel.items(), key=lambda kv: -kv[1][1])},
        per_sentence=per_sent,
    )


def static_mst_check(sentences, vectors, w2i, name, n_random: int = N_RANDOM_TREES) -> dict:
    return mst_check(sentences, lambda s: cosine_distance_matrix(_sentence_vectors(s, vectors, w2i)), name,
                     n_random=n_random)


def per_sentence_frame(m: dict) -> pd.DataFrame:
    """Compact per-sentence counts (for paired bootstrap tests between methods)."""
    return pd.DataFrame([dict(sent_id=p["sent_id"], correct=p["correct"], total=p["total"],
                              chain_correct=p["chain_correct"]) for p in m["per_sentence"]])


# --------------------------------------------------------------------------- qualitative
def nearest_neighbours(vectors: np.ndarray, w2i: dict[str, int], words=PROBE_WORDS, k: int = 10) -> dict:
    U = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    i2w = {i: w for w, i in w2i.items()}
    out = {}
    for w in words:
        if w not in w2i:
            continue
        sims = U @ U[w2i[w]]
        sims[w2i[w]] = -np.inf
        top = np.argsort(-sims)[:k]
        out[w] = [(i2w[int(j)], round(float(sims[j]), 3)) for j in top]
    return out


# --------------------------------------------------------------------------- full suite
def evaluate_static_vectors(vectors: np.ndarray, w2i: dict[str, int], name: str, out_dir: Path,
                            run_pool: bool = True, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    # 1. WordSim-353
    ws = B.evaluate_wordsim(vectors, w2i)
    recs = pd.DataFrame(ws.pop("records"))
    recs.to_csv(out_dir / "wordsim_pairs.csv", index=False)
    cov = recs[recs.covered]
    ws["spearman_covered_ci95"] = bootstrap_spearman(cov.model.values, cov.human.values)
    # frequency-stratified Spearman: are low scores caused by rare (poorly estimated) words?
    counts = word_counts(load_ewt("train"))
    minf = cov.apply(lambda r: min(counts[r.word1], counts[r.word2]), axis=1)
    ws["by_min_train_freq"] = {
        str(t): dict(n=int((minf >= t).sum()),
                     spearman=float(spearmanr(cov.model[minf >= t], cov.human[minf >= t]).statistic)
                     if (minf >= t).sum() >= 10 else float("nan"))
        for t in (1, 5, 10, 20, 50)}

    # 2. analogies
    an = B.evaluate_analogy(vectors, w2i)
    pd.DataFrame(an.pop("examples")).to_csv(out_dir / "analogy_examples.csv", index=False)
    pd.DataFrame([dict(section=k, **v) for k, v in an["per_section"].items()]).to_csv(
        out_dir / "analogy_by_section.csv", index=False)

    # 3 + 4. structural check
    sample, pool = get_mst_sentences()
    mst20 = static_mst_check(sample, vectors, w2i, name)
    (out_dir / "mst_20_sentences.json").write_text(json.dumps(mst20, indent=1))
    mst20_summary = {k: v for k, v in mst20.items() if k not in ("per_sentence",)}
    pool_summary = None
    if run_pool:
        mpool = static_mst_check(pool, vectors, w2i, name)
        per_sentence_frame(mpool).to_csv(out_dir / "mst_pool_per_sentence.csv", index=False)
        pool_summary = {k: v for k, v in mpool.items() if k not in ("per_sentence", "per_relation")}
        pool_summary["per_relation"] = mpool["per_relation"]

    # 5. neighbours
    nn = nearest_neighbours(vectors, w2i)
    (out_dir / "nearest_neighbours.json").write_text(json.dumps(nn, indent=1))

    results = dict(name=name, vocab_size=len(w2i), dim=int(vectors.shape[1]), wordsim=ws,
                   analogy=an, mst_20=mst20_summary, mst_pool=pool_summary)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if verbose:
        print_summary(results, nn)
    return results


def print_summary(r: dict, nn: dict | None = None) -> None:
    ws, an, m = r["wordsim"], r["analogy"], r["mst_20"]
    lo, hi = ws["spearman_covered_ci95"]
    print(f"\n================ STATIC-VECTOR EVALUATION: {r['name']} ================")
    print(f"vocab={r['vocab_size']} dim={r['dim']}")
    print(f"[WordSim-353] Spearman covered = {ws['spearman_covered']:.4f}  (95% CI {lo:.3f}..{hi:.3f}, "
          f"n={ws['n_covered']}/{ws['n_total']})   full (median-imputed) = {ws['spearman_full']:.4f}")
    if "by_min_train_freq" in ws:
        print("   by min train freq of the pair: " + ", ".join(
            f">={t}: {v['spearman']:.3f} (n={v['n']})" for t, v in ws["by_min_train_freq"].items()))
    print(f"[Analogy] acc covered = {an['acc_covered']:.4f} ({an['n_correct']}/{an['n_covered']})   "
          f"acc full = {an['acc_full']:.4f} (/{an['n_total']})")
    for k, v in an["by_kind"].items():
        print(f"   {k:9s}: covered acc = {v['acc_covered']:.4f} ({v['correct']}/{v['covered']})  full = {v['acc_full']:.4f}")
    for sec, v in an["per_section"].items():
        print(f"   {sec:30s} {v['correct']:5d}/{v['covered']:5d} = {v['acc_covered']:.3f}")
    lo, hi = m["uuas_ci95"]
    print(f"[MST 20 sents] UUAS = {m['uuas']:.4f} (95% CI {lo:.3f}..{hi:.3f}; {m['n_edges']} gold edges)  "
          f"| linear chain = {m['chain_uuas']:.4f} | random tree = {m['random_uuas']:.4f}")
    print(f"   vs chain: diff={m['vs_chain']['diff']:+.4f} CI [{m['vs_chain']['ci_low']:+.3f},{m['vs_chain']['ci_high']:+.3f}]"
          f"   vs random: diff={m['vs_random']['diff']:+.4f} CI [{m['vs_random']['ci_low']:+.3f},{m['vs_random']['ci_high']:+.3f}]")
    if r.get("mst_pool"):
        p = r["mst_pool"]
        print(f"[MST eligible pool, {p['n_sentences']} sents] UUAS = {p['uuas']:.4f} "
              f"(CI {p['uuas_ci95'][0]:.3f}..{p['uuas_ci95'][1]:.3f}) | chain = {p['chain_uuas']:.4f} | random = {p['random_uuas']:.4f}")
        top = list(p["per_relation"].items())[:12]
        print("   recall by relation (pool): " + ", ".join(f"{k}={v['recall']:.2f}(n={v['total']})" for k, v in top))
    if nn:
        print("[nearest neighbours]")
        for w, lst in nn.items():
            print(f"   {w:10s} -> " + ", ".join(x for x, _ in lst[:8]))
    print("=" * 72)
