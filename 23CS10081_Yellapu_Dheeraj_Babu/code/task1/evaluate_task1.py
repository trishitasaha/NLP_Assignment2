"""Task 1, steps 4-5: evaluate the PPMI+SVD vectors.

Usage (from the project root, after build_ppmi_svd.py):
    python code/task1/evaluate_task1.py                    # spec vectors in outputs/task1/
    python code/task1/evaluate_task1.py --tag <ablation>   # vectors in outputs/task1/ablations/<tag>/

Writes to the same folder: results.json, wordsim_pairs.csv, analogy_by_section.csv,
analogy_examples.csv, mst_20_sentences.json, nearest_neighbours.json.
The first run also creates outputs/shared/mst_sample_ids.json (the 20 sentences reused by Task 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.static_eval import evaluate_static_vectors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-pool", action="store_true", help="skip the 660-sentence pool MST check")
    args = ap.parse_args()
    C.set_seed()
    d = C.TASK1_OUT / "ablations" / args.tag if args.tag else C.TASK1_OUT
    vecs = np.load(d / "vectors.npy")
    vocab = json.loads((d / "vocab.json").read_text())
    w2i = {w: i for i, w in enumerate(vocab)}
    name = f"T1 PPMI+SVD{'' if not args.tag else ' [' + args.tag + ']'}"
    evaluate_static_vectors(vecs, w2i, name, d, run_pool=not args.no_pool)


if __name__ == "__main__":
    main()
