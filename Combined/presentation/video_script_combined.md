# Common video script (≈ 5 min): two implementations compared

**Speakers:** Yellapu Dheeraj Babu (A) and Trishita Saha (B). One person can read every part; if you both record, each reads their own implementation's code segment.
**Slides:** `presentation_combined.pdf` (14 slides) · **Format:** screen recording with voice-over

| Segment | Time | On screen |
|---|---|---|
| 1. Problem | 0:00 – 0:30 | slides 1 → 2 |
| 2. Code walkthrough | 0:30 – 3:00 | slides 3 → 4, then A's code, then B's code |
| 3. Result analysis | 3:00 – 5:00 | slides 5, 7, 8, 9, 10, 11, 13 |

Narration is sized for about 140 words per minute. Lines in *italics* are on-screen actions, not to be read aloud.

---

## Before recording

1. **Slides:** `presentation_combined.pdf` in full-screen mode, on slide 1.
2. **Editor:** open the cloned repository `NLP_Assignment2` in VS Code at about 150% zoom. Open these tabs in order:
   - **A:** `23CS10081_Yellapu_Dheeraj_Babu/code/task1/build_ppmi_svd.py`, `…/code/task2/probes.py`, `…/code/task2/train_probes.py`, `…/code/task3/pool_static.py`
   - **B:** `Codes/common.py`, `Codes/Task1.py`, `Codes/Task2.py`, `Codes/Task3.py`
3. Use Ctrl+G and a line number to jump.

---

## 1. Problem: 0:00 – 0:30 (slides 1 → 2)

> This is Project 5, *From Static to Contextual*, on the UD English EWT treebank, by Dheeraj and Trishita.
> Task 1 builds classical PPMI and SVD word vectors. Task 2 trains a structural probe on frozen BERT to find the layer that encodes syntax best. Task 3 averages that layer into static vectors and compares them with Task 1, on word meaning and on dependency trees.
> We each implemented the whole pipeline independently, so we can also ask which conclusions survive a change of implementation.

---

## 2. Code walkthrough: 0:30 – 3:00

### 0:30 – 0:50 · Slides 3 → 4

> Both implementations share the data, the benchmarks, the frozen BERT model and the probe definitions from the paper. They differ in a few choices, listed here. The most important is the probe's learning-rate schedule.
> *(Slide 4.)* Our repository keeps the two cleanly separated: Trishita's code at the root, in `Codes` and `results`, and Dheeraj's code in its own folder, one sub-folder per task.

### 0:50 – 1:50 · Implementation A (Dheeraj)

*`build_ppmi_svd.py`, line 37, then 77.*
> In A, Task 1 counts co-occurrences in a plus-or-minus-5-word window, vectorised by offset. Line 77 is PPMI, and the SVD keeps 100 dimensions.

*`probes.py`, line 21, then 33.*
> The distance probe is the squared distance after a projection B. Line 33 computes it with the "norm a squared plus norm b squared minus two a dot b" identity, which saves memory.

*`train_probes.py`, line 74, then 87.*
> The key choice is here: when the dev loss stops improving, the learning rate is cut tenfold, as in the paper, and the best dev checkpoint is kept.

*`pool_static.py`, line 50.*
> Task 3 averages layer-8 vectors over every occurrence of a word, as one sparse matrix product, and evaluates them with exactly the function Task 1 used. Everything runs with `run_all.py`.

### 1:50 – 2:50 · Implementation B (Trishita)

*`Codes/common.py`, line 52.*
> In B, the data loader tries the Hugging Face dataset first and falls back to the official `conllu` files from GitHub.

*`Codes/Task1.py`, lines 35, 56, 74.*
> Task 1 builds the co-occurrence matrix over words seen at least twice, applies PPMI, and runs a truncated SVD with scipy, using a fixed start vector for reproducibility.

*`Codes/Task2.py`, line 248, then 283.*
> `probe_loss` computes both probes for several layers at once, with the L1 loss divided by n squared for distances and by n for depths.
> `train_probe` uses Adam with a constant learning rate of 10⁻³ for 30 epochs and keeps the best dev checkpoint. This is the literal schedule from the assignment.

*`Codes/Task3.py`, line 42, then 217.*
> Task 3 pools layer-8 vectors per word type and also tests mean-centred and PCA-reduced variants.

### 2:50 – 3:00

> Before comparing results, we checked the two Task 3 pipelines against each other.

---

## 3. Result analysis: 3:00 – 5:00

### 3:00 – 3:15 · Slide 5, cross-check

> When we score A's outputs with B's settings, we get B's numbers exactly: 0.648 on the same 203 word pairs, and 0.288 with B's missing-pair rule. The two pipelines are equivalent, so the remaining differences come from settings.

### 3:15 – 3:35 · Slide 7, Task 2

> Both implementations find the same best layer, 8, and the same curve: up through the middle layers, then down. But the UUAS differs: 79.5 percent in A, close to the paper's 79.8, against 68.9 in B.

### 3:35 – 4:00 · Slide 8, the schedule

> The cause is the learning rate. When A is retrained with B's constant rate, it drops to 62 percent. The probe isn't over-fitting: its training loss stays high too. With an L1 loss, Adam keeps taking full-size steps around the minimum, and the paper's decay is what lets it settle. This also explains why B found the smallest rank best.

### 4:00 – 4:20 · Slides 9 → 10, Task 3

> On word meaning, both implementations find a large gain from BERT-static vectors: WordSim goes from about 0.18 to 0.65, and analogies from under 1 percent to around 70.
> *(Slide 10.)* On trees, both find that BERT-static is *worse* than PPMI, close to a random tree. So meaning and syntax disagree, in both implementations.

### 4:20 – 4:45 · Slide 11, why

> Trishita proposed three reasons, and Dheeraj measured them. Averaging removes the context, where the tree lives: static vectors get 18 percent, contextual vectors 38, the probe 83. BERT-static neighbours are substitutes, while PPMI neighbours are partners that occur together. And inside a sentence, BERT-static closeness says nothing about which words are connected.

### 4:45 – 5:00 · Slide 13, conclusions

> So pooling the syntax layer gives better meaning vectors but worse syntax, and two independent implementations agree on every one of these conclusions. Thank you.

---

## If you run long or short

- **Long:** shorten the B Task 3 sentence and the A `pool_static` sentence.
- **Short:** show slide 12 (robust vs. implementation-sensitive findings) for 10 seconds before the conclusions.
