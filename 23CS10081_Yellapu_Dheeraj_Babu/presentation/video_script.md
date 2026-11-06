# Video script: code walkthrough and results (≈ 5 min)

**Speaker:** Yellapu Dheeraj Babu (23CS10081) · **Format:** screen recording with voice-over · **Slides:** `presentation/presentation.pdf` (18 slides; the video uses 2, 3, 5, 7, 9, 11, 13, 14 and 17)

| Segment | Time | On screen |
|---|---|---|
| 1. Problem | 0:00 – 0:30 | slides 1 → 2 → 3 |
| 2. Code walkthrough | 0:30 – 3:00 | slide 5, then VS Code + terminal |
| 3. Result analysis | 3:00 – 5:00 | slides 7, 9, 11, 13, 14, 17 |

Each narration block below is sized to its time slot at a normal speaking pace (about 140 words per minute). Lines in *italics* tell you what to do on screen; you don't read them aloud.

---

## Before you hit record

1. **Slides:** open `presentation/presentation.pdf` in full-screen mode, on slide 1.
2. **VS Code:** open the project folder, set the zoom to about 150% (Ctrl + = twice), and close the sidebar (Ctrl+B). Open these tabs in this order, so Ctrl+Tab or a click moves to the next one:
   1. `code/task1/build_ppmi_svd.py`
   2. `code/common/static_eval.py`
   3. `code/task2/extract_bert.py`
   4. `code/task2/probes.py`
   5. `code/task2/train_probes.py`
   6. `code/task2/evaluate_task2.py`
   7. `code/task3/pool_static.py`
3. **Terminal:** in the VS Code terminal, run `cd ~/nlp_23cs10081 && conda activate nlp5 && clear`, so it's ready for the last 10 seconds of the walkthrough.
4. **Line numbers:** use Ctrl+G and a line number to jump. The numbers below are for the current code.
5. **Recording:** record at 1080p (OBS, or Zoom in recording mode) and do one silent practice run of the tab switching. If you run long, trim the optional lines marked **[cut if long]**.

---

## 1. Problem: 0:00 – 0:30 (slides 1 → 2 → 3)

*Slide 1 (title), about 3 s. Say the first two lines on slide 2 (static vs. contextual), then the rest on slide 3 (the three tasks).*

> Hi, I'm Dheeraj, and this is Project 5: *From Static to Contextual*.
> We compare two kinds of word representations on one treebank, UD English EWT.
> Task 1 builds classical count-based vectors with PPMI and SVD: one fixed vector per word.
> Task 2 trains a structural probe on frozen BERT to find which layer encodes syntax best.
> Task 3 averages that layer's contextual vectors into new static vectors.
> We test both kinds of vectors on word meaning and on dependency-tree structure, and the key question is whether those two evaluations agree.

*(about 75 words)*

---

## 2. Code walkthrough: 0:30 – 3:00

### 0:30 – 0:50 · Slide 5, code organisation

> The code mirrors the three tasks: a folder each for task 1, 2 and 3, plus `common` for data loading, the tree code and statistics.
> The most important design choice is that Tasks 1 and 3 call the *same* evaluation function, so the comparison uses an identical protocol by construction.
> The data flows downward. Task 2 caches BERT features for all 13 layers once, and Task 3 reuses them.

### 0:50 – 1:10 · `task1/build_ppmi_svd.py`

*Ctrl+G **37** (`cooccurrence_matrix`), then **77**, then **83**.*

> Task 1 is a short pipeline. Here we count co-occurrences in a symmetric window of 5 words inside each sentence. It's vectorised: one pass per offset instead of a Python loop.
> Line 77 is PPMI: log of P of w and c over P of w times P of c, keeping only the positive values.
> And here is truncated SVD to 100 dimensions with seed 42. The word vectors are U times Sigma.

### 1:10 – 1:25 · `common/static_eval.py`

*Ctrl+G **146** (`evaluate_static_vectors`), scroll a little, then **72** (`mst_check`).*

> This is the shared evaluation. It computes WordSim-353 Spearman on all 353 pairs and on the covered ones, analogy accuracy with the three query words excluded, and the structural check: a minimum spanning tree over cosine distances, scored against the gold tree, with random-tree and linear-chain baselines and bootstrap confidence intervals.

### 1:25 – 1:45 · `task2/extract_bert.py`

*Ctrl+G **55**, then **66**, then **74**.*

> For Task 2 we feed the gold EWT words to bert-base-uncased, already split into words, so the alignment is exact.
> `output_hidden_states` gives us all 13 layers.
> A word made of several word pieces gets the *mean* of its pieces, as in the paper, computed on the GPU with a scatter-add. Everything is cached once as float16, about 5 gigabytes.

### 1:45 – 2:10 · `task2/probes.py`

*Ctrl+G **21** (`DistanceProbe`), point at line **33**, then **48** (`distance_loss`).*

> These are the two probes from Hewitt and Manning.
> The distance probe predicts the tree distance between two words as the squared distance after a linear projection B, of rank k.
> Line 33 computes it with the identity "norm a squared plus norm b squared minus two a dot b", which gives the same value with far less memory.
> The depth probe is the squared norm of B times h.
> The loss is the paper's: L1 error, divided by n squared for distances, or by n for depths.

### 2:10 – 2:30 · `task2/train_probes.py`

*Ctrl+G **43** (`ProbeRun`), then **74** (`end_epoch`), and point at line **87**.*

> We train 13 layers times 3 ranks times 2 probe types, 78 probes in all, with Adam at learning rate 10⁻³, batches of 20 sentences and 30 epochs.
> Following the paper, if the dev loss stops improving, the learning rate is cut tenfold and the best dev checkpoint is kept.
> **[cut if long]** We also ran the literal constant-rate schedule as a control: it picks the same best layer, but it never converges.

### 2:30 – 2:40 · `task2/evaluate_task2.py`

*Ctrl+G **208**.*

> Evaluation follows the official reporter: UUAS, distance and depth Spearman, and root accuracy. The best layer, ℓ*, is chosen on the *dev* set, never on test.

### 2:40 – 2:50 · `task3/pool_static.py`

*Ctrl+G **50** (`pooling_matrix`), then **82**.*

> Task 3 averages the layer-ℓ* vectors of every occurrence of a word in the training set. Here that is one sparse matrix product. It uses exactly Task 1's vocabulary, so the two cover the same items, which allows paired significance tests.

### 2:50 – 3:00 · Terminal

*Type `python code/run_all.py --dry-run | head -20` and press Enter.*

> Everything runs with one command, `run_all.py`, or with the Colab notebook, and all seeds are fixed at 42.

*(about 360 words in total)*

---

## 3. Result analysis: 3:00 – 5:00 (slides 7 → 9 → 11 → 13 → 14 → 17)

### 3:00 – 3:20 · Slide 7, Task 1 results

> Task 1 is limited by data. EWT gives us only 200 thousand words, and almost half of the word types appear once.
> WordSim correlation is 0.10 overall, but it rises to 0.32 on frequent words, so the method works where it has data.
> Analogies stay below one percent in every setting we tried.
> On trees, it beats a random tree but not the trivial chain baseline.

### 3:20 – 3:50 · Slide 9, Task 2 results

> The structural probe reproduces the paper.
> We get 79.5 percent UUAS on EWT test, against 79.8 in the paper, and depth Spearman of 0.86 against 0.87.
> The curve rises, peaks in the middle layers and falls. Layers 6 to 8 are statistically tied, and dev picks layer 8.
> Rank 64 is already as good as 128, so syntax lives in a small subspace.
> Layer 0, with no context, already gets 59 percent. Context adds about 21 points, mostly for function words.

### 3:50 – 4:20 · Slides 11 → 13, Task 3 vs Task 1 (meaning, then trees)

> Now the main comparison. On word meaning, the BERT-static vectors win by a huge margin: WordSim goes from 0.10 to 0.58, and analogies from half a percent to 68 percent. Both differences are significant.
> *(Go to slide 13.)* But on the tree check, BERT-static is *worse*: 17.9 against 24.8 percent on 660 sentences, barely above a random tree.
> On the required 20 sentences the difference isn't significant. That's why we also report the larger pool.
> So the two evaluations disagree.

### 4:20 – 4:50 · Slide 14, why

> We measured why.
> First, the tree lives in the context. Static vectors get 18 percent, the same vectors seen through the probe's projection get 36, contextual vectors get 38, and the probe gets 83.
> Second, similar is not connected. BERT-static neighbours are substitutes, like good and excellent; PPMI neighbours are partners that occur together, like said and spokesman.
> And inside a sentence, BERT-static closeness carries no information about which words are attached: its edge AUC is 0.46, where 0.5 is chance.

### 4:50 – 5:00 · Slide 17, conclusions

> To conclude: pooling from the syntax layer gives much better word-meaning vectors, but worse syntax, because benchmarks compare word types while trees need tokens in context. Thank you.

*(about 290 words in total)*

---

## Slides not used in the video (for the 10-minute talk / questions)

| Slide | Topic |
|---|---|
| 12 | Layer sweep: static semantics is best at layers 1–2, not layer 8 |
| 15 | One of the 20 sentences, with all four trees |
| 16 | Reproducibility, deviations, constant-lr control run |

## If you run long or short

- **Long:** trim the **[cut if long]** line, then drop the 2:30–2:40 evaluation segment by saying that sentence over the `train_probes.py` tab.
- **Short:** on slide 9, add "the depth probe peaks at the same layer, 8". On slide 14, add "and the anisotropy fix helps semantics but not trees".
