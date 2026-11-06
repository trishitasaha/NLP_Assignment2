"""UD English EWT loading + gold tree utilities shared by all three tasks.

Data flow:
    HF Hub (Parquet, pinned tag) --download_ewt()--> data/ewt/{split}.jsonl --load_ewt()--> list[Sentence]

Conventions
-----------
* Word indices are 0-based inside python; `heads` keeps the CoNLL-U convention
  (1-based index of the head word, 0 = ROOT).
* `tokens` are the *syntactic words* (multi-word-token surface forms such as "don't"
  are NOT included; only "do" + "n't"), so `heads` index directly into `tokens`.
* Punctuation = UPOS "PUNCT" (decision D4 in plan.md).
"""
from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config as C


@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: list[str]
    upos: list[str]
    xpos: list[str]
    heads: list[int]  # 1-based head index, 0 = root
    deprels: list[str]
    n_mwt: int = 0
    n_empty: int = 0
    lower: list[str] = field(default_factory=list)  # lowercased tokens (decision D3)

    def __post_init__(self) -> None:
        if not self.lower:
            self.lower = [t.lower() for t in self.tokens]

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def nonpunct_mask(self) -> np.ndarray:
        return np.array([u != "PUNCT" for u in self.upos], dtype=bool)


# --------------------------------------------------------------------------- download / cache
def _to_str_list(values, feature) -> list[str]:
    """Older loader schemas stored upos as ClassLabel ints; map them back to strings."""
    if values and isinstance(values[0], int):
        names = feature.feature.names  # Sequence(ClassLabel)
        return [names[v] for v in values]
    return [str(v) for v in values]


def download_ewt(out_dir: Path = C.EWT_DIR) -> str:
    """Download EWT from the HF Hub (pinned revision) and write one JSONL file per split.

    Returns the revision that was actually used.
    """
    from datasets import load_dataset  # imported lazily: only needed once

    out_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for rev in C.UD_REVISIONS:
        try:
            ds = load_dataset(C.UD_REPO, C.UD_CONFIG, revision=rev)
            break
        except Exception as e:  # noqa: BLE001 - try the next pinned revision
            print(f"[download_ewt] revision {rev!r} failed: {type(e).__name__}: {e}")
            last_err = e
    else:
        raise RuntimeError("Could not load UD English EWT from any pinned revision") from last_err

    print(f"[download_ewt] loaded revision {rev!r}; splits = {list(ds.keys())}")
    for split in C.SPLITS:
        d = ds[split]
        upos_feat = d.features["upos"]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in d:
                rec = dict(
                    sent_id=ex.get("sent_id", ""),
                    text=ex.get("text", ""),
                    tokens=list(ex["tokens"]),
                    upos=_to_str_list(list(ex["upos"]), upos_feat),
                    xpos=[str(x) if x is not None else "_" for x in ex["xpos"]],
                    heads=[int(h) for h in ex["head"]],
                    deprels=list(ex["deprel"]),
                    n_mwt=len(ex.get("mwt") or []),
                    n_empty=len(ex.get("empty_nodes") or []),
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "REVISION").write_text(rev + "\n")
    return rev


def load_ewt(split: str, data_dir: Path = C.EWT_DIR) -> list[Sentence]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python code/common/prepare_data.py` first")
    with open(path, encoding="utf-8") as f:
        return [Sentence(**json.loads(line)) for line in f]


# --------------------------------------------------------------------------- tree utilities
def tree_problems(heads: list[int]) -> list[str]:
    """Return a list of problems (empty list == valid single-rooted tree)."""
    n = len(heads)
    problems = []
    roots = [i for i, h in enumerate(heads) if h == 0]
    if len(roots) != 1:
        problems.append(f"{len(roots)} roots")
    if any(h < 0 or h > n for h in heads):
        problems.append("head index out of range")
        return problems
    # cycle check: following heads from every node must reach 0 within n steps
    for i in range(n):
        cur, steps = i + 1, 0
        while cur != 0 and steps <= n:
            cur = heads[cur - 1]
            steps += 1
        if cur != 0:
            problems.append("cycle")
            break
    return problems


def undirected_adjacency(heads: list[int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in heads]
    for i, h in enumerate(heads):
        if h > 0:
            adj[i].append(h - 1)
            adj[h - 1].append(i)
    return adj


def gold_distance_matrix(heads: list[int]) -> np.ndarray:
    """Pairwise path length (number of edges) in the undirected dependency tree. Shape [n, n]."""
    n = len(heads)
    adj = undirected_adjacency(heads)
    dist = np.zeros((n, n), dtype=np.float32)
    for s in range(n):
        seen = [-1] * n
        seen[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if seen[v] < 0:
                    seen[v] = seen[u] + 1
                    q.append(v)
        dist[s] = seen
    return dist


def gold_depths(heads: list[int]) -> np.ndarray:
    """Depth of each word = distance to the root word (root has depth 0). Shape [n]."""
    n = len(heads)
    depth = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d, cur = 0, i + 1
        while heads[cur - 1] != 0:
            cur = heads[cur - 1]
            d += 1
        depth[i] = d
    return depth


def gold_edges(heads: list[int], keep: np.ndarray | None = None) -> set[tuple[int, int]]:
    """Undirected gold edges (i<j, 0-based) among words with keep[i] True (both ends kept)."""
    edges = set()
    for i, h in enumerate(heads):
        if h > 0:
            a, b = sorted((i, h - 1))
            if keep is None or (keep[a] and keep[b]):
                edges.add((a, b))
    return edges


# --------------------------------------------------------------------------- vocabulary helpers
def word_counts(sentences: Iterable[Sentence]) -> Counter:
    c: Counter = Counter()
    for s in sentences:
        c.update(s.lower)
    return c


def nonpunct_words(s: Sentence) -> list[str]:
    return [w for w, u in zip(s.lower, s.upos) if u != "PUNCT"]


def mst_eligible(s: Sentence, vocab: set[str]) -> bool:
    """Decision D10: 5-40 non-punct words, all of them in the (shared T1/T3) train vocab."""
    words = nonpunct_words(s)
    return C.MST_MIN_WORDS <= len(words) <= C.MST_MAX_WORDS and all(w in vocab for w in words)


def sample_mst_sentences(test: list[Sentence], vocab: set[str], k: int = C.MST_SAMPLE_SIZE,
                         seed: int = C.SEED) -> list[str]:
    """Deterministically sample k eligible test sentence ids (decision D10)."""
    eligible = sorted(s.sent_id for s in test if mst_eligible(s, vocab))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=min(k, len(eligible)), replace=False)
    return [eligible[i] for i in sorted(idx)]


def to_dict(s: Sentence) -> dict:
    return asdict(s)
