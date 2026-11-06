"""Phase 0: download all data + model and print sanity statistics.

Usage (from the project root, inside the `nlp5` conda env):
    python code/common/prepare_data.py              # everything
    python code/common/prepare_data.py --skip-bert  # no model download / wordpiece stats
    python code/common/prepare_data.py --force      # re-download EWT even if cached

Writes:
    data/ewt/{train,dev,test}.jsonl, data/ewt/REVISION
    data/benchmarks/{wordsim353.tsv, questions-words.txt}
    outputs/shared/data_stats.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

from common import benchmarks as B  # noqa: E402
from common import config as C  # noqa: E402
from common.data import (download_ewt, gold_distance_matrix, gold_edges, load_ewt,  # noqa: E402
                         mst_eligible, tree_problems, word_counts)
from common.trees import gold_mst_edges, mst_edges  # noqa: E402


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch", "transformers",
                "datasets", "huggingface_hub", "tqdm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not installed"
    try:
        import torch
        out["cuda_available"] = torch.cuda.is_available()
        out["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    return out


def split_stats(sents) -> dict:
    lens = np.array([len(s) for s in sents])
    n_punct = sum(sum(u == "PUNCT" for u in s.upos) for s in sents)
    invalid = [(s.sent_id, tree_problems(s.heads)) for s in sents if tree_problems(s.heads)]
    # self-test 1: MST of the full gold-distance matrix must reproduce the gold tree exactly
    mst_mismatch = sum(mst_edges(gold_distance_matrix(s.heads)) != gold_edges(s.heads) for s in sents)
    # open question: punctuation that has dependents (non-leaf) -> gold MST on non-punct words
    # differs from the gold edges restricted to non-punct words
    punct_nonleaf_tokens = 0
    punct_nonleaf_sents = 0
    for s in sents:
        has_dep = Counter(h - 1 for h in s.heads if h > 0)
        k = sum(1 for i, u in enumerate(s.upos) if u == "PUNCT" and has_dep[i] > 0)
        punct_nonleaf_tokens += k
        punct_nonleaf_sents += k > 0
    return dict(
        n_sentences=len(sents), n_tokens=int(lens.sum()), mean_len=float(lens.mean()),
        median_len=float(np.median(lens)), max_len=int(lens.max()),
        n_len_5_50=int(((lens >= C.SPEARMAN_MIN_LEN) & (lens <= C.SPEARMAN_MAX_LEN)).sum()),
        n_punct_tokens=int(n_punct), sentences_with_mwt=sum(s.n_mwt > 0 for s in sents),
        sentences_with_empty_nodes=sum(s.n_empty > 0 for s in sents),
        n_invalid_trees=len(invalid), invalid_examples=invalid[:5],
        mst_selftest_mismatches=int(mst_mismatch),
        punct_nonleaf_tokens=punct_nonleaf_tokens, punct_nonleaf_sentences=punct_nonleaf_sents,
        example=dict(sent_id=sents[0].sent_id, tokens=sents[0].tokens[:15], upos=sents[0].upos[:15],
                     heads=sents[0].heads[:15], deprels=sents[0].deprels[:15]),
    )


def wordpiece_stats(splits: dict) -> dict:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.BERT_MODEL, revision=C.BERT_REVISION)
    AutoModel.from_pretrained(C.BERT_MODEL, revision=C.BERT_REVISION)  # cache weights only
    out = {}
    for name, sents in splits.items():
        enc = tok([s.tokens for s in sents], is_split_into_words=True, add_special_tokens=False)
        n_wp = np.array([len(ids) for ids in enc["input_ids"]])
        n_words = np.array([len(s) for s in sents])
        # words that the tokenizer maps to ZERO wordpieces (e.g. odd whitespace/control chars)
        zero_wp = 0
        for i, s in enumerate(sents):
            wids = set(w for w in enc.word_ids(i) if w is not None)
            zero_wp += len(s) - len(wids)
        out[name] = dict(max_wordpieces=int(n_wp.max()), mean_wordpieces=float(n_wp.mean()),
                         n_over_limit=int((n_wp > C.BERT_MAX_WORDPIECES).sum()),
                         wordpieces_per_word=float(n_wp.sum() / n_words.sum()),
                         words_with_zero_wordpieces=int(zero_wp))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    stats: dict = {"versions": versions()}

    # 1. EWT
    if args.force or not all((C.EWT_DIR / f"{s}.jsonl").exists() for s in C.SPLITS):
        stats["ewt_revision"] = download_ewt()
    else:
        stats["ewt_revision"] = (C.EWT_DIR / "REVISION").read_text().strip()
        print(f"[prepare] EWT already cached (revision {stats['ewt_revision']})")
    splits = {s: load_ewt(s) for s in C.SPLITS}
    stats["splits"] = {s: split_stats(v) for s, v in splits.items()}

    # 2. vocabulary (lowercased train word types, decision D3) shared by Task 1 and Task 3
    counts = word_counts(splits["train"])
    vocab = set(counts)
    freq = np.array(list(counts.values()))
    test_tokens = [w for s in splits["test"] for w in s.lower]
    stats["vocab"] = dict(
        n_types=len(vocab), n_tokens=int(freq.sum()), types_count_1=int((freq == 1).sum()),
        types_count_2_4=int(((freq >= 2) & (freq <= 4)).sum()), types_count_ge5=int((freq >= 5).sum()),
        test_token_oov_rate=float(np.mean([w not in vocab for w in test_tokens])),
        top20=counts.most_common(20),
    )

    # 3. benchmarks + coverage
    stats["benchmark_sha256"] = B.download_benchmarks()
    ws = B.load_wordsim()
    an = B.load_analogies()
    stats["wordsim"] = B.wordsim_coverage(ws, vocab)
    cov_words = [w for a, b, _ in ws for w in (a, b) if w in vocab]
    stats["wordsim"]["median_train_count_of_covered_words"] = float(np.median([counts[w] for w in cov_words]))
    stats["wordsim"]["oov_words_sample"] = sorted({w for a, b, _ in ws for w in (a, b) if w not in vocab})[:30]
    stats["analogy"] = B.analogy_coverage(an, vocab)

    # 4. 20-sentence MST sample eligibility (decision D10)
    eligible = [s for s in splits["test"] if mst_eligible(s, vocab)]
    stats["mst_eligible_test_sentences"] = len(eligible)

    # 5. BERT wordpieces
    if not args.skip_bert:
        stats["wordpieces"] = wordpiece_stats(splits)

    out = C.SHARED_OUT / "data_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=str))
    print_report(stats)
    print(f"\n[prepare] full stats written to {out}")


def print_report(st: dict) -> None:
    print("\n================ PHASE 0 SANITY REPORT ================")
    print("versions:", st["versions"])
    print("EWT revision:", st["ewt_revision"])
    for name, s in st["splits"].items():
        print(f"\n[{name}] sentences={s['n_sentences']} tokens={s['n_tokens']} mean_len={s['mean_len']:.1f} "
              f"max_len={s['max_len']} len5-50={s['n_len_5_50']} punct={s['n_punct_tokens']}")
        print(f"   MWT sents={s['sentences_with_mwt']} empty-node sents={s['sentences_with_empty_nodes']} "
              f"invalid trees={s['n_invalid_trees']} MST self-test mismatches={s['mst_selftest_mismatches']} "
              f"non-leaf punct tokens={s['punct_nonleaf_tokens']} (in {s['punct_nonleaf_sentences']} sents)")
    print(f"\nexample sentence: {st['splits']['train']['example']}")
    v = st["vocab"]
    print(f"\n[vocab] types={v['n_types']} tokens={v['n_tokens']} count==1: {v['types_count_1']} "
          f"2-4: {v['types_count_2_4']} >=5: {v['types_count_ge5']}  test-token OOV rate={v['test_token_oov_rate']:.3%}")
    w = st["wordsim"]
    print(f"\n[WordSim-353] pairs={w['n_pairs']} covered={w['n_covered']} ({w['pair_coverage']:.1%}); "
          f"word coverage={w['word_coverage']:.1%}; median train count of covered words="
          f"{w['median_train_count_of_covered_words']}")
    print("   OOV sample:", w["oov_words_sample"])
    a = st["analogy"]
    print(f"\n[Google analogy] questions={a['n_questions']} covered={a['n_covered']} ({a['coverage']:.1%})")
    for sec, d in a["per_section"].items():
        print(f"   {sec:32s} {d['covered']:5d}/{d['n']:5d} ({d['coverage']:.1%})")
    print(f"\n[MST sample] eligible test sentences (5-40 non-punct words, all in vocab): "
          f"{st['mst_eligible_test_sentences']}")
    if "wordpieces" in st:
        for name, d in st["wordpieces"].items():
            print(f"[wordpieces:{name}] {d}")
    print("=======================================================")


if __name__ == "__main__":
    main()
