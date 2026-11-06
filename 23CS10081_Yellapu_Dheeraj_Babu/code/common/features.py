"""Access to the cached BERT word features written by task2/extract_bert.py.

Layout (in config.FEATURE_DIR):
    {split}_layer{L}.npy   float16 [n_words_in_split, 768], L = 0..12
                           (L = 0: embedding layer, L = 1..12: transformer layers;
                            = HF `hidden_states[L]`)
    {split}_offsets.npy    int64 [n_sentences + 1]; words of sentence s are rows
                           offsets[s] : offsets[s+1], in the original EWT order
    {split}_meta.json      sent_ids, counts, model + revision, pooling description

Each word vector is the MEAN of its wordpiece vectors (Hewitt & Manning 2019).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


def layer_path(split: str, layer: int, feat_dir: Path = C.FEATURE_DIR) -> Path:
    return feat_dir / f"{split}_layer{layer}.npy"


def load_offsets(split: str, feat_dir: Path = C.FEATURE_DIR) -> np.ndarray:
    return np.load(feat_dir / f"{split}_offsets.npy")


def load_layer(split: str, layer: int, mmap: bool = True, feat_dir: Path = C.FEATURE_DIR) -> np.ndarray:
    """[n_words, 768] float16. With mmap=True nothing is read until it is indexed."""
    return np.load(layer_path(split, layer, feat_dir), mmap_mode="r" if mmap else None)


def load_meta(split: str, feat_dir: Path = C.FEATURE_DIR) -> dict:
    return json.loads((feat_dir / f"{split}_meta.json").read_text())


def sentence_slices(offsets: np.ndarray) -> list[slice]:
    return [slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:])]
