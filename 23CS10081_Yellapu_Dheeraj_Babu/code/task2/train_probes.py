"""Task 2: train structural probes for every BERT layer x rank x {distance, depth}.

Usage (from the project root):
    # smoke test (seconds)
    CUDA_VISIBLE_DEVICES=0 python code/task2/train_probes.py --layers 7 --ranks 64 --epochs 1 --tag smoke
    # full sweep: layers 0..12 x ranks {32,64,128} x {distance, depth} = 78 probes, 30 epochs each
    CUDA_VISIBLE_DEVICES=0 nohup python code/task2/train_probes.py > outputs/task2_train.log 2>&1 &

Training (assignment step 3 + paper App. A.2):
  * Adam, lr 1e-3, batch 20 sentences (random order, re-shuffled every epoch), 30 epochs, train split.
  * Paper schedule (decision D7): after each epoch the dev loss is computed. If it does not reach a
    new minimum (by > 1e-4), the learning rate is multiplied by 0.1 and the optimizer is reset (no
    momentum kept). The checkpoint with the lowest dev loss is kept.
  * All 6 probes of a layer are trained side by side on the same batches (independent parameters,
    optimizers and schedules), so each batch is built only once.
  * Resumable: probes whose finished checkpoint exists are skipped.

Outputs: outputs/task2/probes[/<tag>]/L{layer}_{probe}_k{rank}.pt  (best state + training history)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402

from probe_data import GoldData, LayerFeatures  # noqa: E402
from probes import PROBES  # noqa: E402

MIN_DELTA = 1e-4


def ckpt_path(out: Path, layer: int, probe: str, rank: int) -> Path:
    return out / f"L{layer}_{probe}_k{rank}.pt"


class ProbeRun:
    """One probe with its own optimizer, LR schedule and best-dev checkpoint."""

    def __init__(self, layer: int, probe: str, rank: int, device, lr: float):
        torch.manual_seed(C.SEED + 1000 * layer + rank + (0 if probe == "distance" else 500))
        cls, self.loss_fn = PROBES[probe]
        self.model = cls(C.BERT_DIM, rank).to(device)
        self.layer, self.probe, self.rank, self.lr = layer, probe, rank, lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.best, self.best_epoch, self.best_state = float("inf"), -1, None
        self.history: list[dict] = []
        self._train_sum, self._train_n = 0.0, 0

    def step(self, h, dist, dep, lens) -> None:
        pred = self.model(h)
        loss = self.loss_fn(pred, dist if self.probe == "distance" else dep, lens)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._train_sum = self._train_sum + loss.detach() * len(lens)  # stays on GPU: no per-step sync
        self._train_n += len(lens)

    @torch.no_grad()
    def dev_loss(self, dev_batches) -> float:
        tot, n = 0.0, 0
        for h, dist, dep, lens in dev_batches():
            pred = self.model(h)
            tot += self.loss_fn(pred, dist if self.probe == "distance" else dep, lens).item() * len(lens)
            n += len(lens)
        return tot / n

    def end_epoch(self, epoch: int, dev: float, decay: bool = True, keep_last: bool = False) -> None:
        improved = dev < self.best - MIN_DELTA
        self.history.append(dict(epoch=epoch, train_loss=float(self._train_sum) / max(self._train_n, 1),
                                 dev_loss=dev, lr=self.lr, improved=improved))
        self._train_sum, self._train_n = 0.0, 0
        if improved:
            self.best, self.best_epoch = dev, epoch
            if not keep_last:
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if keep_last:  # literal assignment: the saved probe is the final-epoch one (overwritten each epoch)
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.last_epoch, self.last_dev = epoch, dev
        if not improved and decay:  # paper: reset optimizer with lr * 0.1
            self.lr *= 0.1
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

    def save(self, path: Path, args) -> None:
        # best_* = the saved state: the best-dev epoch, or (keep_last) the final epoch
        saved_epoch, saved_dev = ((self.last_epoch, self.last_dev) if args.keep_last
                                  else (self.best_epoch, self.best))
        torch.save(dict(state_dict=self.best_state, layer=self.layer, probe=self.probe, rank=self.rank,
                        best_dev_loss=saved_dev, best_epoch=saved_epoch, min_dev_loss=self.best,
                        min_dev_epoch=self.best_epoch, history=self.history,
                        epochs=args.epochs, batch_size=args.batch_size, lr0=args.lr, seed=C.SEED,
                        lr_decay=not args.no_decay, keep_last=args.keep_last, finished=True), path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=C.PROBE_LAYERS)
    ap.add_argument("--ranks", type=int, nargs="+", default=C.PROBE_RANKS)
    ap.add_argument("--probes", nargs="+", default=["distance", "depth"])
    ap.add_argument("--epochs", type=int, default=C.PROBE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.PROBE_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.PROBE_LR)
    ap.add_argument("--tag", default=None, help="write to outputs/task2/probes_<tag>/ (e.g. smoke)")
    ap.add_argument("--max-train-sents", type=int, default=None, help="debug: subsample train")
    ap.add_argument("--no-decay", action="store_true",
                    help="constant lr (no x0.1 decay on dev plateau): the literal assignment schedule")
    ap.add_argument("--keep-last", action="store_true",
                    help="save the final-epoch probe instead of the best-dev one (literal assignment)")
    args = ap.parse_args()
    C.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = C.TASK2_OUT / (f"probes_{args.tag}" if args.tag else "probes")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} layers={args.layers} ranks={args.ranks} probes={args.probes} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} decay={not args.no_decay} "
          f"keep_last={args.keep_last} -> {out}", flush=True)

    t0 = time.time()
    train_gold, dev_gold = GoldData("train", device), GoldData("dev", device)
    print(f"[train] gold targets ready ({time.time() - t0:.0f} s)", flush=True)
    n_train = len(train_gold) if args.max_train_sents is None else min(args.max_train_sents, len(train_gold))

    for layer in args.layers:
        todo = [(p, k) for p in args.probes for k in args.ranks if not ckpt_path(out, layer, p, k).exists()]
        if not todo:
            print(f"[train] layer {layer}: all probes done, skipping", flush=True)
            continue
        tl = time.time()
        train_x, dev_x = LayerFeatures(train_gold, layer), LayerFeatures(dev_gold, layer)
        runs = [ProbeRun(layer, p, k, device, args.lr) for p, k in todo]

        dev_idx = torch.arange(len(dev_gold), device=device)

        def dev_batches():
            for s in range(0, len(dev_idx), 100):
                idx = dev_idx[s:s + 100]
                dist, dep, lens = dev_gold.targets(idx)
                yield dev_x.inputs(idx), dist, dep, lens

        gen = torch.Generator().manual_seed(C.SEED + layer)
        for epoch in range(1, args.epochs + 1):
            perm = torch.randperm(n_train, generator=gen).to(device)
            for r in runs:
                r.model.train()
            for s in range(0, n_train, args.batch_size):
                idx = perm[s:s + args.batch_size]
                dist, dep, lens = train_gold.targets(idx)
                h = train_x.inputs(idx)
                for r in runs:
                    r.step(h, dist, dep, lens)
            msg = []
            for r in runs:
                r.model.eval()
                r.end_epoch(epoch, r.dev_loss(dev_batches), decay=not args.no_decay, keep_last=args.keep_last)
                hst = r.history[-1]
                msg.append(f"{r.probe[:4]}-k{r.rank}: tr={hst['train_loss']:.4f} dev={hst['dev_loss']:.4f}"
                           f"{'*' if hst['improved'] else ''} lr={hst['lr']:.0e}")
            print(f"[L{layer:02d} ep{epoch:02d}] " + " | ".join(msg), flush=True)

        for r in runs:
            r.save(ckpt_path(out, layer, r.probe, r.rank), args)
        print(f"[train] layer {layer} done in {time.time() - tl:.0f} s; best dev: " +
              ", ".join(f"{r.probe}-k{r.rank}={r.best:.4f}@ep{r.best_epoch}" for r in runs), flush=True)
        del train_x, dev_x, runs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=1))
    print(f"[train] all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
