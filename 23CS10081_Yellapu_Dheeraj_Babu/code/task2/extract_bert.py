"""Task 2, preparation: run bert-base-uncased ONCE over EWT train/dev/test and cache every
layer's word-level vectors (decision D8). Used by the probes (Task 2) and by Task 3.

Usage (from the project root; pick a GPU with >= 3 GB free):
    CUDA_VISIBLE_DEVICES=0 python code/task2/extract_bert.py            # extract + checks
    python code/task2/extract_bert.py --check-only                     # re-run checks only

Method:
  * Input = the gold EWT tokens (no re-tokenisation), passed with is_split_into_words=True.
    The uncased tokenizer lowercases internally. All sentences fit in 512 wordpieces (Phase 0).
  * output_hidden_states=True gives 13 tensors: [0] embeddings, [1..12] transformer layers.
  * word vector = mean of its wordpiece vectors (Hewitt & Manning 2019, footnote on BERT).
  * Computation in fp32 on GPU, stored as float16 (~5 GB in total).

Checks (written to outputs/task2/feature_checks.json):
  * alignment: #word vectors == #EWT tokens in every split; no NaN/Inf
  * contextuality per layer (Ethayarajh 2019): self-similarity of frequent words across their
    occurrences vs. the anisotropy baseline (mean cosine of random word-occurrence pairs)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402
from common.data import load_ewt  # noqa: E402
from common.features import layer_path, load_layer, load_offsets  # noqa: E402

N_LAYERS = C.BERT_NUM_LAYERS + 1  # 13 incl. embeddings
CHECK_WORDS = ["the", "and", "to", "i", "it", "time", "people", "like", "right", "good", "work", "go"]


@torch.inference_mode()
def extract_split(split: str, tok, model, device, batch_size: int, out_dir: Path) -> dict:
    sents = load_ewt(split)
    lengths = np.array([len(s) for s in sents])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    n_words = int(offsets[-1])
    np.save(out_dir / f"{split}_offsets.npy", offsets)
    mm = [np.lib.format.open_memmap(layer_path(split, L, out_dir), mode="w+", dtype=np.float16,
                                    shape=(n_words, C.BERT_DIM)) for L in range(N_LAYERS)]

    order = np.argsort(-lengths, kind="stable")  # long first: similar lengths per batch, OOM early
    for start in tqdm(range(0, len(order), batch_size), desc=f"BERT {split}"):
        idx = order[start:start + batch_size]
        batch_tokens = [sents[i].tokens for i in idx]
        enc = tok(batch_tokens, is_split_into_words=True, return_tensors="pt", padding=True,
                  truncation=False)
        T = enc["input_ids"].shape[1]
        if T > C.BERT_MAX_WORDPIECES + 2:
            raise ValueError(f"sentence longer than 512 wordpieces in {split}")
        # word index of every wordpiece (-1 for [CLS]/[SEP]/[PAD])
        widx = torch.full((len(idx), T), -1, dtype=torch.long)
        for b in range(len(idx)):
            for t, w in enumerate(enc.word_ids(batch_index=b)):
                if w is not None:
                    widx[b, t] = w
        out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        H = torch.stack(out.hidden_states, 0)  # [13, B, T, 768]

        widx = widx.to(device)
        max_w = int(lengths[idx].max())
        valid = widx >= 0
        tgt = torch.where(valid, widx, torch.full_like(widx, max_w))  # dump pieces into an extra slot
        sums = torch.zeros(N_LAYERS, len(idx), max_w + 1, C.BERT_DIM, device=device)
        sums.scatter_add_(2, tgt[None, :, :, None].expand(N_LAYERS, -1, -1, C.BERT_DIM), H.float())
        cnt = torch.zeros(len(idx), max_w + 1, device=device)
        cnt.scatter_add_(1, tgt, valid.float())
        if (cnt[:, :max_w] == 0).logical_and(
                torch.arange(max_w, device=device)[None] < torch.as_tensor(lengths[idx], device=device)[:, None]).any():
            raise RuntimeError("a word received zero wordpieces - alignment problem")
        mean = (sums[:, :, :max_w] / cnt[None, :, :max_w, None].clamp(min=1)).half().cpu().numpy()

        for b, s in enumerate(idx):
            n = lengths[s]
            for L in range(N_LAYERS):
                mm[L][offsets[s]:offsets[s] + n] = mean[L, b, :n]
    for m in mm:
        m.flush()
    meta = dict(split=split, n_sentences=len(sents), n_words=n_words, sent_ids=[s.sent_id for s in sents],
                model=C.BERT_MODEL, revision=C.BERT_REVISION, layers="0=embeddings, 1..12=transformer",
                pooling="mean over wordpieces", dtype="float16")
    (out_dir / f"{split}_meta.json").write_text(json.dumps(meta))
    return dict(n_sentences=len(sents), n_words=n_words)


def run_checks(out_dir: Path) -> dict:
    rng = np.random.default_rng(C.SEED)
    checks: dict = {"alignment": {}, "contextuality": {}}
    for split in C.SPLITS:
        offs = load_offsets(split, out_dir)
        n_tok = sum(len(s) for s in load_ewt(split))
        bad = 0
        for L in range(N_LAYERS):
            X = load_layer(split, L, feat_dir=out_dir)
            assert X.shape == (n_tok, C.BERT_DIM), (split, L, X.shape, n_tok)
            sample = np.asarray(X[rng.integers(0, len(X), 5000)], dtype=np.float32)
            bad += int((~np.isfinite(sample)).sum())
        checks["alignment"][split] = dict(n_words=int(offs[-1]), n_tokens=n_tok, match=int(offs[-1]) == n_tok,
                                          nonfinite_in_sample=bad)

    # contextuality on train: self-similarity vs anisotropy (Ethayarajh 2019)
    words = [w for s in load_ewt("train") for w in s.lower]
    pos: dict[str, np.ndarray] = {}
    arr = np.array(words, dtype=object)
    for w in CHECK_WORDS:
        p = np.flatnonzero(arr == w)
        if len(p) >= 20:
            pos[w] = rng.choice(p, size=min(200, len(p)), replace=False)
    pairs = rng.integers(0, len(words), size=(5000, 2))
    print("\nlayer | anisotropy (random pairs) | mean self-sim | self-sim - anisotropy")
    for L in range(N_LAYERS):
        X = load_layer("train", L, feat_dir=out_dir)

        def unit(v):
            v = np.asarray(v, dtype=np.float32)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        a, b = unit(X[np.sort(pairs[:, 0])]), unit(X[np.sort(pairs[:, 1])])
        aniso = float(np.mean(np.sum(a * b, 1)))
        selfsim = {}
        for w, p in pos.items():
            U = unit(X[np.sort(p)])
            G = U @ U.T
            k = len(U)
            selfsim[w] = float((G.sum() - k) / (k * (k - 1)))
        ms = float(np.mean(list(selfsim.values())))
        checks["contextuality"][L] = dict(anisotropy=aniso, self_sim=selfsim, mean_self_sim=ms,
                                          adjusted_self_sim=ms - aniso)
        print(f"{L:5d} | {aniso:25.3f} | {ms:13.3f} | {ms - aniso:8.3f}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--splits", nargs="+", default=C.SPLITS)
    args = ap.parse_args()
    C.set_seed()
    out_dir = C.FEATURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.check_only:
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[extract] device={device} model={C.BERT_MODEL}@{C.BERT_REVISION[:7]} -> {out_dir}")
        tok = AutoTokenizer.from_pretrained(C.BERT_MODEL, revision=C.BERT_REVISION)
        model = AutoModel.from_pretrained(C.BERT_MODEL, revision=C.BERT_REVISION).to(device).eval()
        t0 = time.time()
        for split in args.splits:
            info = extract_split(split, tok, model, device, args.batch_size, out_dir)
            print(f"[extract] {split}: {info}")
        print(f"[extract] done in {time.time() - t0:.0f} s")

    checks = run_checks(out_dir)
    print("\n[alignment]", json.dumps(checks["alignment"], indent=1))
    (C.TASK2_OUT / "feature_checks.json").write_text(json.dumps(checks, indent=1))
    size = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1e9
    print(f"[extract] feature dir size: {size:.2f} GB; checks -> {C.TASK2_OUT / 'feature_checks.json'}")


if __name__ == "__main__":
    main()
