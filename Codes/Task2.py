"""
Task 2: Structural probe (Hewitt & Manning, NAACL 2019) on frozen bert-base-uncased,
trained / evaluated on UD English EWT.

What it does
  1. Extracts word vectors from EVERY BERT layer (1..12) for train/dev/test
     (word-piece vectors are averaged per word) and caches them on disk (fp16).
  2. For each layer, each rank k in {32, 64, 128}, trains
        * a DISTANCE probe  (squared L2 distance of B(h_i - h_j)  ~ tree distance)
        * a DEPTH probe     (squared L2 norm of B h_i              ~ tree depth)
     with Adam, lr 1e-3, batch of 20 sentences, 30 epochs (L1 loss, as in the paper).
  3. Decodes an undirected minimum spanning tree from the distance probe -> UUAS.
     Also reports depth-probe Spearman (and a few extras).
  4. Saves a table, two plots (UUAS vs layer, depth Spearman vs layer) and
     results/task2/best_layer.json (the layer l* that Task 3 needs).

Run (all defaults = the assignment settings):
        python task2_structural_probe.py

Useful options:
        --plot_only            only rebuild the table/plots from saved runs
        --layers 6 7 8         only these layers (1-based)
        --ranks 128            only this rank
        --epochs 30 --lr 1e-3 --bs 20
        --chunk 4              how many layers are trained together on the GPU (lower it if GPU memory is short)
        --redo                 ignore saved runs and train again
The script is RESUMABLE: every finished (probe, rank, layer) is saved, so if Colab
disconnects just run the same command again.
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.stats import rankdata

import common

DEFAULT_OUT = os.path.join(common.RESULTS_DIR, "task2")


# =============================================================================
# 1. Data: tokens, heads, upos from the .conllu files (same files/order as Task 1)
# =============================================================================
def read_conllu_upos(path):
    sents, toks, heads, upos = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    sents.append({"tokens": toks, "head": heads, "upos": upos})
                    toks, heads, upos = [], [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:      # multi-word token / empty node
                continue
            toks.append(cols[1])
            upos.append(cols[3])
            heads.append(int(cols[6]))
    if toks:
        sents.append({"tokens": toks, "head": heads, "upos": upos})
    return sents


def load_data(limit=None):
    out = {}
    for name in ["train", "dev", "test"]:
        path = common._download(common.UD_URL.format(name),
                                os.path.join(common.DATA_DIR, f"en_ewt-ud-{name}.conllu"))
        s = read_conllu_upos(path)
        out[name] = s[:limit] if limit else s
    return out


def gold_tree(heads):
    """Gold tree distance matrix D[i,j] (number of edges on the path) and depth[i]
    (number of edges from word i up to the root), computed from the head field."""
    n = len(heads)
    adj = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h > 0:
            adj[i].append(h - 1)
            adj[h - 1].append(i)
    D = np.full((n, n), -1, dtype=np.int16)
    for s in range(n):
        D[s, s] = 0
        queue = [s]
        for u in queue:                    # BFS (the list grows while we loop)
            for v in adj[u]:
                if D[s, v] < 0:
                    D[s, v] = D[s, u] + 1
                    queue.append(v)
    assert (D >= 0).all(), "gold tree is not connected"
    root = heads.index(0)
    return D, D[root].copy(), root


# =============================================================================
# 2. BERT feature extraction (all layers, cached)
# =============================================================================
def feat_path(out, split):
    return os.path.join(out, "features", f"{split}.npy")


def extract_features(data, args, device):
    from transformers import AutoModel, AutoTokenizer
    fdir = os.path.join(args.out, "features")
    os.makedirs(fdir, exist_ok=True)
    meta_path = os.path.join(fdir, "meta.json")
    totals = {sp: int(sum(len(s["tokens"]) for s in ss)) for sp, ss in data.items()}
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    ok = (meta.get("model") == args.model and meta.get("totals") == totals
          and all(os.path.exists(feat_path(args.out, sp)) for sp in data))
    if ok:
        print(f"Using cached BERT features in {fdir}")
        return meta["n_layers"], meta["dim"]

    print(f"Extracting features with {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.is_fast, "need a fast tokenizer (for word_ids)"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    nl, dim = model.config.num_hidden_layers, model.config.hidden_size
    zero_words = 0
    t0 = time.time()
    for split, sents in data.items():
        lens = np.array([len(s["tokens"]) for s in sents])
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        mm = np.lib.format.open_memmap(feat_path(args.out, split), mode="w+",
                                       dtype=np.float16, shape=(nl, int(lens.sum()), dim))
        order = np.argsort(lens)                     # similar lengths together = less padding
        for s0 in range(0, len(order), args.extract_bs):
            ids = order[s0:s0 + args.extract_bs]
            batch = [sents[i]["tokens"] for i in ids]
            enc = tok(batch, is_split_into_words=True, return_tensors="pt",
                      padding=True, truncation=True, max_length=512)
            T, Wmax = enc["input_ids"].shape[1], int(lens[ids].max())
            P = np.zeros((len(ids), Wmax, T), dtype=np.float32)   # word <- word-piece averaging
            for b in range(len(ids)):
                for t, w in enumerate(enc.word_ids(b)):
                    if w is not None:
                        P[b, w, t] = 1.0
                cnt = P[b].sum(1, keepdims=True)
                zero_words += int((cnt[:lens[ids[b]]] == 0).sum())
                P[b] /= np.maximum(cnt, 1.0)
            with torch.no_grad():
                out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
                hs = torch.stack(out.hidden_states[1:], 0)        # layers 1..nl (0 = embeddings)
                pooled = torch.einsum("bwt,lbtd->lbwd", torch.from_numpy(P).to(device), hs)
            pooled = pooled.half().cpu().numpy()
            for b, i in enumerate(ids):
                mm[:, offs[i]:offs[i] + lens[i]] = pooled[:, b, :lens[i]]
            if (s0 // args.extract_bs) % 50 == 0:
                print(f"  {split}: {s0}/{len(order)} sentences  ({time.time() - t0:.0f}s)")
        mm.flush()
        del mm
    meta = {"model": args.model, "n_layers": nl, "dim": dim, "totals": totals,
            "words_without_wordpieces": zero_words}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Features saved ({time.time() - t0:.0f}s). Words with no word-piece (zero vector): {zero_words}")
    del model
    return nl, dim


# =============================================================================
# 3. Splits on the device: features + padded gold distances / depths
# =============================================================================
class Split:
    def __init__(self, name, sents, device, out):
        self.name, self.sents, self.device, self.out = name, sents, device, out
        self.n = len(sents)
        self.lens = np.array([len(s["tokens"]) for s in sents])
        self.offs = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.Lmax = int(self.lens.max())
        dt = np.uint8 if self.Lmax <= 255 else np.int16
        Dg = np.zeros((self.n, self.Lmax, self.Lmax), dtype=dt)
        dep = np.zeros((self.n, self.Lmax), dtype=dt)
        self.gold = []
        for i, s in enumerate(sents):
            D, d, root = gold_tree(s["head"])
            n = len(d)
            Dg[i, :n, :n] = D
            dep[i, :n] = d
            self.gold.append((D, d, root))
        self.Dg = torch.from_numpy(Dg).to(device)
        self.dep = torch.from_numpy(dep).to(device)
        self.feats = None
        self._eval_ready = False

    def load_layers(self, layers):            # layers = 0-based consecutive list
        arr = np.load(feat_path(self.out, self.name), mmap_mode="r")
        self.feats = torch.from_numpy(np.array(arr[layers[0]:layers[-1] + 1])).to(self.device)

    def free(self):
        self.feats = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def batch(self, idx):                      # idx: numpy array of sentence ids
        lens = self.lens[idx]
        L = int(lens.max())
        ar = np.arange(L)
        valid = ar[None, :] < lens[:, None]
        pos = np.where(valid, self.offs[idx][:, None] + ar[None, :], 0)
        dev = self.device
        pos_t = torch.from_numpy(pos).to(dev)
        valid_t = torch.from_numpy(valid).to(dev)
        idx_t = torch.from_numpy(idx).to(dev)
        X = self.feats[:, pos_t].float() * valid_t[None, :, :, None]       # (layers,B,L,dim)
        return (X, torch.from_numpy(lens).float().to(dev), valid_t,
                self.Dg[idx_t][:, :L, :L].float(), self.dep[idx_t][:, :L].float())

    def prepare_eval(self):
        """Pre-compute everything about the gold trees that evaluation needs."""
        if self._eval_ready:
            return
        self.ge, self.ge_np, self.nonpunct, self.drank, self.deprank = [], [], [], [], []
        for s, (D, d, root) in zip(self.sents, self.gold):
            n = len(d)
            ge = common.gold_edges(s["head"])
            npm = np.array([u != "PUNCT" for u in s["upos"]])
            self.ge.append(ge)
            self.ge_np.append({e for e in ge if npm[e[0]] and npm[e[1]]})
            self.nonpunct.append(np.where(npm)[0])
            if 5 <= n <= 50:
                self.drank.append(rankdata(D.astype(float), axis=1))
                self.deprank.append(rankdata(d.astype(float)))
            else:
                self.drank.append(None)
                self.deprank.append(None)
        self._eval_ready = True


# =============================================================================
# 4. The probes (several layers are trained at once; each has its own B and loss,
#    and Adam works per-parameter, so this equals training them separately)
# =============================================================================
def probe_loss(kind, Bm, X, lens, valid, Dg, dep):
    """Bm: (layers,k,dim). Returns per-layer loss (layers,) and predictions."""
    T = torch.matmul(X, Bm.transpose(1, 2).unsqueeze(1))          # (layers,B,L,k) = B h
    sq = (T * T).sum(-1)                                          # (layers,B,L)   = ||B h||^2
    if kind == "depth":
        err = (sq - dep.unsqueeze(0)).abs() * valid.unsqueeze(0)
        per_sent = err.sum(-1) / lens.unsqueeze(0)                # L1 loss / n
        return per_sent.mean(1), sq
    gram = torch.matmul(T, T.transpose(-1, -2))
    pred = (sq.unsqueeze(-1) + sq.unsqueeze(-2) - 2 * gram).clamp_min(0)   # ||B(h_i-h_j)||^2
    mask = (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(0)
    err = (pred - Dg.unsqueeze(0)).abs() * mask
    per_sent = err.sum((-1, -2)) / (lens ** 2).unsqueeze(0)       # L1 loss / n^2
    return per_sent.mean(1), pred


@torch.no_grad()
def evaluate_split(kind, Bm, split, want_preds=False, bs=50):
    nl = Bm.shape[0]
    tot = torch.zeros(nl, device=Bm.device)
    preds = [[None] * split.n for _ in range(nl)] if want_preds else None
    for s in range(0, split.n, bs):
        idx = np.arange(s, min(s + bs, split.n))
        X, lens, valid, Dg, dep = split.batch(idx)
        loss, pred = probe_loss(kind, Bm, X, lens, valid, Dg, dep)
        tot += loss * len(idx)
        if want_preds:
            p = pred.cpu().numpy()
            for l in range(nl):
                for j, i in enumerate(idx):
                    n = split.lens[i]
                    preds[l][i] = p[l, j, :n, :n] if kind == "distance" else p[l, j, :n]
    return (tot / split.n).cpu().numpy(), preds


def train_probe(kind, k, nl_chunk, dim, train, dev, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Bm = torch.empty(nl_chunk, k, dim, device=device).uniform_(-0.05, 0.05).requires_grad_()
    opt = torch.optim.Adam([Bm], lr=args.lr)
    best = Bm.detach().clone()
    best_dev = np.full(nl_chunk, np.inf)
    best_ep = np.zeros(nl_chunk, dtype=int)
    train_loss = np.zeros(nl_chunk)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(train.n)
        tot = torch.zeros(nl_chunk, device=device)
        for s in range(0, train.n, args.bs):
            idx = perm[s:s + args.bs]
            X, lens, valid, Dg, dep = train.batch(idx)
            loss, _ = probe_loss(kind, Bm, X, lens, valid, Dg, dep)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            tot += loss.detach() * len(idx)
        train_loss = (tot / train.n).cpu().numpy()
        dev_loss, _ = evaluate_split(kind, Bm.detach(), dev)
        improved = dev_loss < best_dev
        if improved.any():
            m = torch.from_numpy(improved).to(device)
            best[m] = Bm.detach()[m]
            best_dev[improved] = dev_loss[improved]
            best_ep[improved] = ep
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"    epoch {ep:2d}/{args.epochs}  train loss {np.round(train_loss, 3)}  "
                  f"dev loss {np.round(dev_loss, 3)}  ({time.time() - t0:.0f}s)")
    return best, best_dev, best_ep, train_loss


# =============================================================================
# 5. Metrics
# =============================================================================
def rowwise_spearman(P, Grank):
    """Spearman between each row of P and the matching (already ranked) row of Grank."""
    P = np.atleast_2d(P)
    Pr = rankdata(P, axis=1)
    a = Pr - Pr.mean(1, keepdims=True)
    b = Grank - Grank.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def distance_metrics(preds, sp):
    sp.prepare_eval()
    c_all = t_all = c_np = t_np = 0
    dspr = []
    for i, P in enumerate(preds):
        P = (P + P.T) / 2
        pred = common.mst_edges(P)
        c_all += len(pred & sp.ge[i])
        t_all += len(sp.ge[i])
        idx = sp.nonpunct[i]                                  # MST over non-punctuation words only
        if len(idx) >= 2:
            sub = common.mst_edges(P[np.ix_(idx, idx)])
            pred_np = {(int(min(idx[a], idx[b])), int(max(idx[a], idx[b]))) for a, b in sub}
            c_np += len(pred_np & sp.ge_np[i])
        t_np += len(sp.ge_np[i])
        if sp.drank[i] is not None:
            dspr.append(np.nanmean(rowwise_spearman(P, sp.drank[i])))
    return {"uuas_all": c_all / t_all, "uuas_nopunct": c_np / max(t_np, 1),
            "dist_spearman": float(np.nanmean(dspr))}


def depth_metrics(preds, sp):
    sp.prepare_eval()
    spr, root_ok = [], []
    for i, p in enumerate(preds):
        root_ok.append(int(np.argmin(p) == sp.gold[i][2]))
        if sp.deprank[i] is not None:
            spr.append(rowwise_spearman(p, sp.deprank[i][None, :])[0])
    return {"depth_spearman": float(np.nanmean(spr)), "root_acc": float(np.mean(root_ok))}


# =============================================================================
# 6. Main training sweep
# =============================================================================
def run_path(out, kind, k, layer):
    return os.path.join(out, "runs", f"{kind}_k{k}_L{layer:02d}.json")


def sweep(args, data, nl, dim, device):
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    splits = {n: Split(n, s, device, args.out) for n, s in data.items()}
    layers_all = args.layers or list(range(1, nl + 1))
    chunks = [layers_all[i:i + args.chunk] for i in range(0, len(layers_all), args.chunk)]
    # chunks must be consecutive layers (features are sliced); split non-consecutive input
    fixed = []
    for ch in chunks:
        cur = [ch[0]]
        for l in ch[1:]:
            if l == cur[-1] + 1:
                cur.append(l)
            else:
                fixed.append(cur)
                cur = [l]
        fixed.append(cur)
    t_start = time.time()
    for ch in fixed:
        for k in args.ranks:
            for kind in ["distance", "depth"]:
                if not args.redo and all(os.path.exists(run_path(args.out, kind, k, l)) for l in ch):
                    print(f"[skip] {kind} k={k} layers {ch} already done")
                    continue
                print(f"\n=== {kind} probe, rank k={k}, BERT layers {ch}  "
                      f"(elapsed {(time.time() - t_start) / 60:.1f} min) ===")
                for sp in splits.values():
                    sp.load_layers([ch[0] - 1, ch[-1] - 1])
                Bm, best_dev, best_ep, train_loss = train_probe(
                    kind, k, len(ch), dim, splits["train"], splits["dev"], args, device,
                    seed=args.seed + 1000 * k + (0 if kind == "distance" else 7) + ch[0])
                res = {}
                for name in ["dev", "test"]:
                    _, preds = evaluate_split(kind, Bm, splits[name], want_preds=True)
                    res[name] = [(distance_metrics if kind == "distance" else depth_metrics)(
                        preds[j], splits[name]) for j in range(len(ch))]
                for j, l in enumerate(ch):
                    r = {"kind": kind, "k": k, "layer": l, "epochs": args.epochs,
                         "best_epoch": int(best_ep[j]), "best_dev_loss": float(best_dev[j]),
                         "final_train_loss": float(train_loss[j]),
                         "dev": res["dev"][j], "test": res["test"][j]}
                    with open(run_path(args.out, kind, k, l), "w") as f:
                        json.dump(r, f, indent=2)
                    key = "uuas_all" if kind == "distance" else "depth_spearman"
                    print(f"    layer {l:2d}: dev {key}={r['dev'][key]:.4f}  test {key}={r['test'][key]:.4f}"
                          f"  (best epoch {r['best_epoch']})")
                for sp in splits.values():
                    sp.free()


# =============================================================================
# 7. Summary table, best layer, plots
# =============================================================================
def load_runs(out):
    runs = {}
    for fn in glob.glob(os.path.join(out, "runs", "*.json")):
        with open(fn) as f:
            r = json.load(f)
        runs[(r["kind"], r["k"], r["layer"])] = r
    return runs


def summarize(args):
    runs = load_runs(args.out)
    if not runs:
        print("No finished runs found.")
        return
    ks = sorted({k for (_, k, _) in runs})
    layers = sorted({l for (_, _, l) in runs})

    def get(kind, k, l, split, key):
        r = runs.get((kind, k, l))
        return r[split][key] if r else float("nan")

    # ---- tables
    def table(title, kind, key, split="test"):
        print(f"\n{title}")
        print("layer | " + " | ".join(f"k={k:<4d}" for k in ks))
        for l in layers:
            print(f"{l:5d} | " + " | ".join(f"{get(kind, k, l, split, key):.4f}" for k in ks))

    table("Test UUAS (all words) - distance probe", "distance", "uuas_all")
    table("Test UUAS (punctuation excluded) - distance probe", "distance", "uuas_nopunct")
    table("Test depth-probe Spearman (sentences of length 5-50)", "depth", "depth_spearman")

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "k", "layer", "split", "uuas_all", "uuas_nopunct",
                    "dist_spearman", "depth_spearman", "root_acc", "best_epoch"])
        for (kind, k, l), r in sorted(runs.items()):
            for sp in ["dev", "test"]:
                m = r[sp]
                w.writerow([kind, k, l, sp, m.get("uuas_all", ""), m.get("uuas_nopunct", ""),
                            m.get("dist_spearman", ""), m.get("depth_spearman", ""),
                            m.get("root_acc", ""), r["best_epoch"]])

    # ---- best layer l*  (chosen on DEV so the test set is not used for selection)
    dist = [(r["dev"]["uuas_all"], -l, k) for (kind, k, l), r in runs.items() if kind == "distance"]
    if dist:
        dv, nl_, bk = max(dist)
        best_l = -nl_
        best_test = max(((r["test"]["uuas_all"], -l, k) for (kind, k, l), r in runs.items()
                         if kind == "distance"))
        dep = [(r["test"]["depth_spearman"], -l, k) for (kind, k, l), r in runs.items() if kind == "depth"]
        info = {"best_layer": best_l, "best_k": bk,
                "selected_by": "highest DEV UUAS over layers and ranks",
                "dev_uuas": dv, "test_uuas": runs[("distance", bk, best_l)]["test"]["uuas_all"],
                "best_layer_by_test_uuas": -best_test[1], "best_k_by_test_uuas": best_test[2],
                "best_layer_by_test_depth_spearman": (-max(dep)[1] if dep else None)}
        with open(os.path.join(args.out, "best_layer.json"), "w") as f:
            json.dump(info, f, indent=2)
        print("\nBEST LAYER l* =", best_l, f"(rank k={bk}, dev UUAS {dv:.4f}, "
              f"test UUAS {info['test_uuas']:.4f})")
        print("  best layer by test UUAS:", info["best_layer_by_test_uuas"],
              "| best layer by test depth Spearman:", info["best_layer_by_test_depth_spearman"])
        print(f"  saved {os.path.join(args.out, 'best_layer.json')}")
    else:
        best_l = None
    make_plots(args, runs, ks, layers, best_l)


def make_plots(args, runs, ks, layers, best_l):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}          # blue, orange, aqua
    markers = ["o", "s", "^"]
    ink, grid, surface = "#0b0b0b", "#e4e3df", "#fcfcfb"

    def one(kind, key, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        for ci, k in enumerate(ks):
            xs = [l for l in layers if (kind, k, l) in runs]
            ys = [runs[(kind, k, l)]["test"][key] for l in xs]
            ax.plot(xs, ys, color=colors[ci % 3], marker=markers[ci % 3], lw=2, ms=6,
                    markeredgecolor=surface, markeredgewidth=1.2, label=f"k = {k}")
        if best_l is not None and kind == "distance":
            ax.axvline(best_l, color="#52514e", lw=1, ls="--")
            ax.text(best_l + 0.1, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"l* = {best_l}", color="#52514e", fontsize=9)
        ax.set_xticks(layers)
        ax.set_xlabel("BERT layer", color=ink)
        ax.set_ylabel(ylabel, color=ink)
        ax.set_title(title, color=ink, loc="left", fontsize=12)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#9a9992")
        ax.tick_params(colors=ink)
        ax.legend(frameon=False, labelcolor=ink)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, fname), facecolor=surface)
        plt.close(fig)
        print("saved", os.path.join(args.out, fname))

    one("distance", "uuas_all", "UUAS (test)", "Distance probe: UUAS vs BERT layer (EWT test)",
        "fig_uuas_vs_layer.png")
    one("depth", "depth_spearman", "Spearman correlation (test)",
        "Depth probe: Spearman vs BERT layer (EWT test)", "fig_depth_spearman_vs_layer.png")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="1-based layers (default: all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--extract_bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only the first N sentences per split")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--extract_only", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.plot_only:
        summarize(args)
        return
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: no GPU found - this will be very slow. In Colab: Runtime > Change runtime type > GPU.")
    data = load_data(args.limit)
    print({k: len(v) for k, v in data.items()}, "sentences")
    nl, dim = extract_features(data, args, device)
    if args.extract_only:
        return
    sweep(args, data, nl, dim, device)
    summarize(args)


if __name__ == "__main__":
    main()
