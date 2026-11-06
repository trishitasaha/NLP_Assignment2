"""WordSim-353 and Google analogy benchmarks: download, load, coverage, and evaluation.

The same functions are used for Task 1 (PPMI+SVD) and Task 3 (BERT-pooled) vectors, so the
protocol is identical by construction.

Vector interface: `vectors` is a float array [V, d], `w2i` maps lowercased word -> row.
"""
from __future__ import annotations

import hashlib
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import config as C


# --------------------------------------------------------------------------- download / load
def _download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[benchmarks] downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_benchmarks() -> dict[str, str]:
    """Download both benchmark files (if missing). Returns sha256 checksums."""
    return {
        "wordsim353": _download(C.WORDSIM_URL, C.WORDSIM_PATH),
        "google_analogy": _download(C.ANALOGY_URL, C.ANALOGY_PATH),
    }


def load_wordsim(path: Path = C.WORDSIM_PATH) -> list[tuple[str, str, float]]:
    """353 (word1, word2, human_score) triples, lowercased, file order preserved."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        w1, w2, score = line.split("\t")[:3]
        pairs.append((w1.lower(), w2.lower(), float(score)))
    return pairs


def load_analogies(path: Path = C.ANALOGY_PATH) -> "OrderedDict[str, list[tuple[str, str, str, str]]]":
    """Section name -> list of (a, b, c, d) meaning a:b :: c:d. Lowercased (gensim default)."""
    sections: OrderedDict[str, list] = OrderedDict()
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith(":"):
            current = line[1:].strip()
            sections[current] = []
        else:
            a, b, c, d = line.lower().split()
            sections[current].append((a, b, c, d))
    return sections


def is_syntactic(section: str) -> bool:
    return section.startswith("gram")


# --------------------------------------------------------------------------- coverage
def wordsim_coverage(pairs, vocab) -> dict:
    covered = sum(1 for a, b, _ in pairs if a in vocab and b in vocab)
    words = {w for a, b, _ in pairs for w in (a, b)}
    return dict(n_pairs=len(pairs), n_covered=covered, pair_coverage=covered / len(pairs),
                n_words=len(words), word_coverage=sum(w in vocab for w in words) / len(words))


def analogy_coverage(sections, vocab) -> dict:
    out, tot, cov = {}, 0, 0
    for name, qs in sections.items():
        c = sum(1 for q in qs if all(w in vocab for w in q))
        out[name] = dict(n=len(qs), covered=c, coverage=c / len(qs))
        tot += len(qs)
        cov += c
    return dict(n_questions=tot, n_covered=cov, coverage=cov / tot, per_section=out)


# --------------------------------------------------------------------------- evaluation
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def evaluate_wordsim(vectors: np.ndarray, w2i: dict[str, int], pairs=None,
                     oov_policy: str = "median") -> dict:
    """Spearman between model cosine similarities and human scores.

    * covered: only pairs whose two words are in the vocabulary (main number).
    * full: all 353 pairs; OOV pairs get an imputed similarity. Default "median" (the median
      of the model's covered similarities, i.e. a neutral rank). "zero" uses 0. Zero is NOT
      neutral for anisotropic vectors like BERT, where nearly all cosines are > 0, so the
      default is "median".
    """
    pairs = pairs if pairs is not None else load_wordsim()
    U = _unit(vectors)
    sims, human, covered_mask, records = [], [], [], []
    for a, b, h in pairs:
        ok = a in w2i and b in w2i
        s = float(U[w2i[a]] @ U[w2i[b]]) if ok else np.nan
        sims.append(s)
        human.append(h)
        covered_mask.append(ok)
        records.append(dict(word1=a, word2=b, human=h, model=s, covered=ok))
    sims, human, covered_mask = np.array(sims), np.array(human), np.array(covered_mask)

    rho_cov = spearmanr(sims[covered_mask], human[covered_mask]).statistic
    fill = 0.0 if oov_policy == "zero" else float(np.median(sims[covered_mask]))
    full = np.where(covered_mask, sims, fill)
    rho_full = spearmanr(full, human).statistic
    return dict(spearman_covered=float(rho_cov), spearman_full=float(rho_full),
                n_covered=int(covered_mask.sum()), n_total=len(pairs), oov_policy=oov_policy,
                records=records)


def evaluate_analogy(vectors: np.ndarray, w2i: dict[str, int], sections=None,
                     batch_size: int = 512) -> dict:
    """3CosAdd / vector offset: argmax_x cos(x, b - a + c) over the vocabulary, excluding a, b, c.

    Vectors are unit-normalised first (same as gensim's most_similar). Questions with any OOV
    word count as wrong in the "full" accuracy and are skipped in the "covered" accuracy.
    """
    sections = sections if sections is not None else load_analogies()
    U = _unit(vectors)
    i2w = {i: w for w, i in w2i.items()}
    per_section, examples = {}, []
    tot_n = tot_cov = tot_correct = 0
    agg = {"semantic": [0, 0, 0], "syntactic": [0, 0, 0]}  # n, covered, correct

    for name, qs in sections.items():
        cov = [q for q in qs if all(w in w2i for w in q)]
        correct = 0
        for start in range(0, len(cov), batch_size):
            batch = cov[start:start + batch_size]
            ia = np.array([w2i[q[0]] for q in batch])
            ib = np.array([w2i[q[1]] for q in batch])
            ic = np.array([w2i[q[2]] for q in batch])
            target = U[ib] - U[ia] + U[ic]  # [B, d]
            scores = target @ U.T  # [B, V]; argmax unaffected by |target|
            rows = np.arange(len(batch))
            for idx in (ia, ib, ic):
                scores[rows, idx] = -np.inf
            pred = scores.argmax(1)
            for q, p in zip(batch, pred):
                hit = i2w[int(p)] == q[3]
                correct += hit
                if len(examples) < 20000:  # all covered questions (5,727) are kept
                    examples.append(dict(section=name, a=q[0], b=q[1], c=q[2], gold=q[3],
                                         pred=i2w[int(p)], correct=bool(hit)))
        per_section[name] = dict(n=len(qs), covered=len(cov), correct=correct,
                                 acc_covered=correct / len(cov) if cov else float("nan"),
                                 acc_full=correct / len(qs))
        kind = "syntactic" if is_syntactic(name) else "semantic"
        agg[kind][0] += len(qs)
        agg[kind][1] += len(cov)
        agg[kind][2] += correct
        tot_n += len(qs)
        tot_cov += len(cov)
        tot_correct += correct

    def _acc(c, d):
        return c / d if d else float("nan")

    return dict(
        acc_covered=_acc(tot_correct, tot_cov), acc_full=_acc(tot_correct, tot_n),
        n_total=tot_n, n_covered=tot_cov, n_correct=tot_correct,
        by_kind={k: dict(n=v[0], covered=v[1], correct=v[2], acc_covered=_acc(v[2], v[1]),
                         acc_full=_acc(v[2], v[0])) for k, v in agg.items()},
        per_section=per_section, examples=examples,
    )
