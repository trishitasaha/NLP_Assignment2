> **Repository context.** This folder is **Yellapu Dheeraj Babu's (23CS10081) implementation** inside the shared repository
> `trishitasaha/NLP_Assignment2`. Trishita Saha's implementation is at the repository root (`Codes/`, `results/`), and the
> common report and slides are in `../Combined/`.
>
> **Included here:** all code, `outputs/` with all results, figures, logs and the 78 + 26 probe checkpoints (`.pt`), and
> the individual `report/` and `presentation/` sources and PDFs.
> **Not included** (size): `data/` (downloaded by `code/common/prepare_data.py`), the BERT feature cache
> `outputs/task2/features/` (4.8 GB, rebuilt by `code/task2/extract_bert.py` in ~7 min) and the pooled vectors
> `outputs/task3/static/` (0.6 GB, rebuilt by `code/task3/pool_static.py` in seconds). Run every command from inside this folder.

# Project 5: From Static to Contextual. Distributional Semantics and What's Inside BERT

**NLP CS60075 Assignment II** · dataset: **UD English EWT** · Yellapu Dheeraj Babu (23CS10081)

The project has three tasks:
- **Task 1:** classical PPMI + SVD word vectors.
- **Task 2:** a reproduction of the Hewitt & Manning (2019) structural probe on `bert-base-uncased`.
- **Task 3:** static word vectors pooled from the most syntactic BERT layer, compared with Task 1 on semantic and structural tests.

The full write-up is in `report.pdf`.

| Result (EWT test) | Value |
|---|---|
| T1 PPMI-SVD: WS-353 ρ (all 353 / covered 248) · analogy acc. (all / covered) · MST UUAS (20 sentences) | 0.048 / 0.104 · 0.15% / 0.51% · 20.5% |
| T2 probe: best layer ℓ\* (dev) · UUAS · depth Spearman (NSpr) | **8** · **79.5%** · **0.859** (paper, BERT-base: 79.8%, 0.87) |
| T3 BERT-static (layer 8): WS-353 ρ · analogy acc. · MST UUAS (20 sentences) | 0.452 / 0.577 · 19.8% / 67.7% · 18.4% |

---

## 1. Layout

```
code/
  common/                shared: config + seed, EWT loading, benchmarks, trees/MST, bootstrap stats,
                         prepare_data.py (downloads + sanity report), static_eval.py (the T1/T3 evaluation suite)
  task1/                 build_ppmi_svd.py, evaluate_task1.py, summarize_task1.py
  task2/                 extract_bert.py, probes.py, probe_data.py, train_probes.py, evaluate_task2.py,
                         plot_task2.py, layer_significance.py
  task3/                 pool_static.py, t3_common.py, evaluate_task3.py, extensions_task3.py,
                         diagnostics_task3.py, plot_task3.py
  run_all.py             runs everything (or selected tasks) in order
  run_all_colab.ipynb    Google Colab notebook: end-to-end run
README.md  requirements.txt  report.pdf  presentation.pdf (individual slides)
```

Data, the BERT feature cache and probe checkpoints are **not** in the archive. They are downloaded or produced by the scripts (§3, §5).

## 2. Environment

Python **3.12** (developed with 3.12.16). Every package version is pinned in `requirements.txt`.

```bash
conda create -n nlp5 python=3.12 -y && conda activate nlp5     # or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is needed for Task 2 (any GPU with ≥ 8 GB; we used an NVIDIA L40S). Tasks 1 and 3 run on CPU.

**Pretrained checkpoint:** `google-bert/bert-base-uncased` from the Hugging Face Hub, revision `86b5e0934494bd15c9632b12f734a8a67f723594`. It is downloaded automatically.

## 3. Data (download and preparation)

```bash
python code/common/prepare_data.py          # add --skip-bert to skip caching the BERT model
```

This script does the following:
- **UD English EWT** (the assigned dataset): from HF `universal-dependencies/universal_dependencies`, config `en_ewt`, at the immutable tag `ud2.18-loader-v2.3.0`. Written to `data/ewt/{train,dev,test}.jsonl`.
- **WordSim-353** and the **Google analogy set**: the copies shipped with gensim, at the pinned commit `88d032b`. Written to `data/benchmarks/`, with SHA-256 checksums recorded.
- **BERT:** caches the model and runs a sanity report (tree validity, MST self-test, vocabulary and benchmark coverage, word-piece lengths). The report is written to `outputs/shared/data_stats.json`.

**The other two Project-5 datasets** (not used in our results) are prepared the same way:

```bash
P5_TREEBANK=en_gum   python code/common/prepare_data.py     # UD English GUM   -> data/gum/,   outputs_gum/
P5_TREEBANK=en_lines python code/common/prepare_data.py     # UD English LinES -> data/lines/, outputs_lines/
```

Setting `P5_TREEBANK` for any later command runs the whole pipeline on that treebank, in separate folders.

## 4. Running

### Everything, in sequence (one command)

```bash
python code/run_all.py                    # data → Task 1 → Task 2 → Task 3   (~2–4 h on one GPU)
python code/run_all.py --fast             # Task 2 with rank 64 only (26 instead of 78 probes)
python code/run_all.py --with-control     # also the spec-literal Task 2 control run (report §4.2)
python code/run_all.py --dry-run          # just print the commands
```

**Colab:** open `code/run_all_colab.ipynb` and choose *Runtime → GPU*. Upload this archive when asked, then *Run all*.

### Each task on its own

Run from the project root. Task 2 needs the data, and Task 3 needs Task 1 and Task 2.

| Task | Commands | Time |
|---|---|---|
| **1** | `python code/task1/build_ppmi_svd.py && python code/task1/evaluate_task1.py`<br>ablations: `build_ppmi_svd.py --eig-power 0 --tag p0` + `evaluate_task1.py --tag p0` (likewise `p05`, `cds075`, `cds075_p05`), then `summarize_task1.py` | ~1 min, CPU |
| **2** | `python code/task2/extract_bert.py` (features of all 13 layers, ~5 GB, 7 min)<br>`python code/task2/train_probes.py` (78 probes; resumable; 2 GPUs: `--layers 0 1 2 3 4 5 6` and `--layers 7 8 9 10 11 12` with `CUDA_VISIBLE_DEVICES=0/1`)<br>`python code/task2/evaluate_task2.py && python code/task2/plot_task2.py && python code/task2/layer_significance.py` | ~1 h on 2 GPUs |
| **3** | `python code/task3/pool_static.py && python code/task3/evaluate_task3.py && python code/task3/extensions_task3.py && python code/task3/diagnostics_task3.py && python code/task3/plot_task3.py` | ~5 min |

Task 3 uses **ℓ\*** from `outputs/task2/best_layer.json`, the cached features, and (for the reference rows) the trained probes. It also uses the Task 1 `p0` ablation as the "best classical variant" row.

**Spec-literal control for Task 2** (constant lr, last-epoch probe):

```bash
python code/task2/train_probes.py --ranks 128 --no-decay --keep-last --tag spec_literal
python code/task2/evaluate_task2.py --probe-dir outputs/task2/probes_spec_literal --out-dir outputs/task2/spec_literal
```

## 5. Where results and checkpoints go

| Path | Contents |
|---|---|
| `outputs/shared/` | `data_stats.json` (sanity report), `mst_sample_ids.json` (the 20 sampled test-sentence IDs, reused by T1 and T3) |
| `outputs/task1/` | `vectors.npy`, `vocab.json`, `results.json` (all T1 metrics), per-item CSVs, `ablations/<tag>/`, `summary_table.csv` |
| `outputs/task2/features/` | cached BERT word vectors, `{split}_layer{0..12}.npy` (float16) |
| `outputs/task2/probes/` | probe checkpoints `L{layer}_{distance,depth}_k{rank}.pt` (best-dev state + training history) |
| `outputs/task2/` | `results_task2.csv` (every layer × rank × probe, dev + test), `best_layer.json` (ℓ\*), `paper_comparison.csv`, `per_relation_test.csv`, `layer_significance.csv`, `figures/`, `spec_literal/` |
| `outputs/task3/` | `static/layer{L}.npy` (pooled vectors), `main/` + `main_abtt/` (T3 results, same files as T1), `comparison_table.csv`, `paired_tests.json`, `extensions/` (layer sweep, ladder), `diagnostics/`, `qualitative/`, `figures/` |
| `outputs/run_all_logs/` | one log per step when using `run_all.py` |

## 6. Reproducibility

- **Seed:** `SEED = 42` (`code/common/config.py`) seeds Python, NumPy and PyTorch, and sets `PYTHONHASHSEED`. It is used for the following:
  - the 20-sentence sample: `numpy.random.default_rng(42)` over the eligible test sentences;
  - the random-tree baseline;
  - all bootstrap CIs (1000 resamples);
  - `TruncatedSVD(random_state=42)`;
  - the anisotropy sampling.
- **Probe seeds:** each probe is initialised with seed `42 + 1000·layer + rank (+500 for depth)`, and batch order uses seed `42 + layer`.
- **Fixed inputs:** the dataset tag, benchmark commit and model revision are all pinned (§3).
- **Report numbers:** every number in the report is generated from `outputs/` by `report/make_tables.py`. That script is not part of the submission archive.
