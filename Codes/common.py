import json
import os
import random
import re
import urllib.request

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr

DATA_DIR = "data"
RESULTS_DIR = "results"

UD_URL = ("https://raw.githubusercontent.com/UniversalDependencies/"
          "UD_English-EWT/master/en_ewt-ud-{}.conllu")
WORDSIM_URL = ("https://raw.githubusercontent.com/piskvorky/gensim/develop/"
               "gensim/test/test_data/wordsim353.tsv")
ANALOGY_URL = ("https://raw.githubusercontent.com/piskvorky/gensim/develop/"
               "gensim/test/test_data/questions-words.txt")


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def read_conllu(path):
    sents, toks, heads = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    sents.append({"tokens": toks, "head": heads})
                    toks, heads = [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            toks.append(cols[1])
            heads.append(int(cols[6]))
    if toks:
        sents.append({"tokens": toks, "head": heads})
    return sents


def load_ewt(source="auto"):
    if source in ("auto", "hf"):
        try:
            from datasets import load_dataset
            ds = load_dataset("universal_dependencies", "en_ewt",
                              trust_remote_code=True)
            out = {}
            for hf_name, name in [("train", "train"),
                                  ("validation", "dev"),
                                  ("test", "test")]:
                out[name] = [{"tokens": list(ex["tokens"]),
                              "head": [int(h) for h in ex["head"]]}
                             for ex in ds[hf_name]]
            print("Loaded EWT from Hugging Face.")
            return out
        except Exception as e:  # noqa
            print(f"[info] Hugging Face loading failed ({type(e).__name__}); "
                  f"falling back to .conllu files from GitHub.")
            if source == "hf":
                raise
    out = {}
    for name in ["train", "dev", "test"]:
        path = _download(UD_URL.format(name),
                         os.path.join(DATA_DIR, f"en_ewt-ud-{name}.conllu"))
        out[name] = read_conllu(path)
    print("Loaded EWT from .conllu files.")
    return out


def load_wordsim(path=None):
    path = path or _download(WORDSIM_URL, os.path.join(DATA_DIR, "wordsim353.tsv"))
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in re.split(r"[\t,]", line.strip())]
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:      # header line
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), score))
    return pairs


def load_analogy(path=None):
    path = path or _download(ANALOGY_URL, os.path.join(DATA_DIR, "questions-words.txt"))
    qs, cat = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cat = line[1:].strip()
                continue
            w = line.lower().split()
            if len(w) == 4:
                qs.append((cat, *w))
    return qs



def normalize(M, eps=1e-12):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, eps)


def eval_wordsim(pairs, word2id, vecs):
    Wn = normalize(vecs)
    gold_cov, pred_cov, gold_all, pred_all = [], [], [], []
    for w1, w2, score in pairs:
        gold_all.append(score)
        if w1 in word2id and w2 in word2id:
            sim = float(Wn[word2id[w1]] @ Wn[word2id[w2]])
            gold_cov.append(score)
            pred_cov.append(sim)
            pred_all.append(sim)
        else:
            pred_all.append(0.0)
    return {
        "n_pairs": len(pairs),
        "n_covered": len(gold_cov),
        "spearman_covered": float(spearmanr(gold_cov, pred_cov)[0]),
        "spearman_all353": float(spearmanr(gold_all, pred_all)[0]),
    }


def eval_analogy(questions, word2id, vecs, batch=500):
    Wn = normalize(vecs).astype(np.float32)
    covered = [(cat, a, b, c, d) for cat, a, b, c, d in questions
               if all(w in word2id for w in (a, b, c, d))]
    ids = np.array([[word2id[w] for w in q[1:]] for q in covered])  # (N,4)
    cats = [q[0] for q in covered]
    correct = np.zeros(len(covered), dtype=bool)
    for s in range(0, len(covered), batch):
        A, B, C, D = (ids[s:s + batch, i] for i in range(4))
        Q = Wn[B] - Wn[A] + Wn[C]
        S = Q @ Wn.T
        rows = np.arange(len(A))
        S[rows, A] = -np.inf
        S[rows, B] = -np.inf
        S[rows, C] = -np.inf
        correct[s:s + batch] = (S.argmax(1) == D)

    def acc(mask):
        return float(correct[mask].mean()) if mask.sum() else float("nan")

    is_syn = np.array([c is not None and c.startswith("gram") for c in cats], dtype=bool)
    return {
        "n_questions": len(questions),
        "n_covered": len(covered),
        "acc_covered": float(correct.mean()) if len(covered) else float("nan"),
        "acc_all": float(correct.sum() / len(questions)),   # OOV questions count as wrong
        "acc_semantic_covered": acc(~is_syn),
        "acc_syntactic_covered": acc(is_syn),
    }


def gold_edges(heads):
    return {(min(i, h - 1), max(i, h - 1)) for i, h in enumerate(heads) if h > 0}


def mst_edges(D):
    D = np.array(D, dtype=np.float64) + 1e-9
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(D).tocoo()
    return {(int(min(i, j)), int(max(i, j))) for i, j in zip(T.row, T.col)}


def cosine_dist_matrix(tokens, word2id, Wn):
    n = len(tokens)
    E = np.zeros((n, Wn.shape[1]))
    known = np.zeros(n, dtype=bool)
    for i, t in enumerate(tokens):
        j = word2id.get(t.lower())
        if j is not None and np.any(Wn[j]):
            E[i] = Wn[j]
            known[i] = True
    D = 1.0 - E @ E.T
    D[~known, :] = 1.0
    D[:, ~known] = 1.0
    return np.clip(D, 0.0, 2.0)


def eval_mst_agreement(sents, word2id, vecs):
    Wn = normalize(vecs)
    correct = total = 0
    per_sent, unknown_frac = [], []
    for s in sents:
        D = cosine_dist_matrix(s["tokens"], word2id, Wn)
        pred, gold = mst_edges(D), gold_edges(s["head"])
        c = len(pred & gold)
        correct += c
        total += len(gold)
        per_sent.append(c / len(gold))
        unknown_frac.append(float(np.mean([t.lower() not in word2id for t in s["tokens"]])))
    return {
        "n_sentences": len(sents),
        "n_gold_edges": total,
        "agreement_micro": correct / total,
        "agreement_macro": float(np.mean(per_sent)),
        "avg_fraction_words_without_vector": float(np.mean(unknown_frac)),
    }


def baseline_agreement(sents, seed=0, trials=100):
    rng = np.random.RandomState(seed)
    lin = tot = 0
    rnd = 0.0
    for s in sents:
        g = gold_edges(s["head"])
        n = len(s["tokens"])
        tot += len(g)
        lin += len({(i, i + 1) for i in range(n - 1)} & g)
        r = 0
        for _ in range(trials):
            D = rng.rand(n, n)
            r += len(mst_edges((D + D.T) / 2) & g)
        rnd += r / trials
    return {"linear_chain": lin / tot, "random_tree": rnd / tot}


def sample_test_sentences(test_sents, n=20, seed=42, min_len=5,
                          path=os.path.join(RESULTS_DIR, "sample_indices.json")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            idx = json.load(f)["indices"]
    else:
        cand = [i for i, s in enumerate(test_sents) if len(s["tokens"]) >= min_len]
        idx = sorted(random.Random(seed).sample(cand, n))
        with open(path, "w") as f:
            json.dump({"seed": seed, "min_len": min_len, "indices": idx}, f)
    return [test_sents[i] for i in idx], idx
