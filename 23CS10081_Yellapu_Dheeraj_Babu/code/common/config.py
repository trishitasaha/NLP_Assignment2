"""Central configuration: paths, pinned data/model sources, seeds and constants.

Every task imports from here, so all pinned versions and hyper-parameters live in one place.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
# code/common/config.py -> project root is two levels above code/
PROJECT_ROOT = Path(os.environ.get("P5_ROOT", Path(__file__).resolve().parents[2]))
# Treebank: the assigned dataset is UD English EWT (default). The two other datasets listed for
# Project 5 can be selected with P5_TREEBANK=en_gum or P5_TREEBANK=en_lines (same HF loader and
# schema); they get their own data/<name>/ and outputs_<name>/ folders so nothing is overwritten.
TREEBANK = os.environ.get("P5_TREEBANK", "en_ewt")
assert TREEBANK in ("en_ewt", "en_gum", "en_lines"), f"unknown treebank {TREEBANK}"
_TB_SHORT = TREEBANK.removeprefix("en_")
DATA_DIR = PROJECT_ROOT / "data"
EWT_DIR = DATA_DIR / _TB_SHORT  # (name kept for backward compatibility: the folder of the selected treebank)
BENCH_DIR = DATA_DIR / "benchmarks"
OUTPUT_DIR = PROJECT_ROOT / ("outputs" if TREEBANK == "en_ewt" else f"outputs_{_TB_SHORT}")
SHARED_OUT = OUTPUT_DIR / "shared"
TASK1_OUT = OUTPUT_DIR / "task1"
TASK2_OUT = OUTPUT_DIR / "task2"
TASK3_OUT = OUTPUT_DIR / "task3"
# cached BERT word features (~5 GB); override with P5_FEATURE_DIR, e.g. a local scratch disk
FEATURE_DIR = Path(os.environ.get("P5_FEATURE_DIR", TASK2_OUT / "features"))

# --------------------------------------------------------------------------- seeds
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix python / numpy / torch RNGs (torch only if it is installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- dataset (UD English EWT)
# HF repo is Parquet-only (loader v2.x, needs datasets>=4.0). We pin an immutable *tag*
# (UD release 2.18, loader schema v2.3.0: upos = strings, head = ints, MWT surface forms
# excluded from `tokens`). Fallbacks are tried in order if the tag cannot be resolved.
UD_REPO = "universal-dependencies/universal_dependencies"
UD_CONFIG = TREEBANK
UD_REVISIONS = ["ud2.18-loader-v2.3.0", "2.18", "main"]
SPLITS = ["train", "dev", "test"]

# --------------------------------------------------------------------------- benchmarks
# gensim's test-data copies, pinned to a fixed commit for reproducibility.
_GENSIM_COMMIT = "88d032bce7772939a6ca051d8e30701b4eb59c1e"
_GENSIM_RAW = f"https://raw.githubusercontent.com/piskvorky/gensim/{_GENSIM_COMMIT}/gensim/test/test_data"
WORDSIM_URL = f"{_GENSIM_RAW}/wordsim353.tsv"
ANALOGY_URL = f"{_GENSIM_RAW}/questions-words.txt"
WORDSIM_PATH = BENCH_DIR / "wordsim353.tsv"
ANALOGY_PATH = BENCH_DIR / "questions-words.txt"

# --------------------------------------------------------------------------- model
BERT_MODEL = "google-bert/bert-base-uncased"  # == "bert-base-uncased" on the HF Hub
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_MAX_WORDPIECES = 510  # 512 minus [CLS] and [SEP]
BERT_NUM_LAYERS = 12  # hidden_states[0] = embeddings, [1..12] = transformer layers
BERT_DIM = 768

# --------------------------------------------------------------------------- Task 1
WINDOW = 5  # symmetric co-occurrence window (+-5)
SVD_DIM = 100
MIN_COUNT = 1  # keep every train word type (maximises benchmark coverage)

# --------------------------------------------------------------------------- 20-sentence MST check (T1 + T3)
MST_SAMPLE_SIZE = 20
MST_MIN_WORDS = 5  # non-punctuation words
MST_MAX_WORDS = 40
MST_SAMPLE_PATH = SHARED_OUT / "mst_sample_ids.json"

# --------------------------------------------------------------------------- Task 2 (Hewitt & Manning 2019)
PROBE_LAYERS = list(range(0, BERT_NUM_LAYERS + 1))  # 0 = embedding baseline, 1..12 required
PROBE_RANKS = [32, 64, 128]
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 20
PROBE_EPOCHS = 30
SPEARMAN_MIN_LEN = 5  # macro-average Spearman over sentence lengths 5..50 (as in the paper)
SPEARMAN_MAX_LEN = 50

# --------------------------------------------------------------------------- statistics
BOOTSTRAP_RESAMPLES = 1000


def ensure_dirs() -> None:
    for d in (DATA_DIR, EWT_DIR, BENCH_DIR, SHARED_OUT, TASK1_OUT, TASK2_OUT, TASK3_OUT):
        d.mkdir(parents=True, exist_ok=True)
