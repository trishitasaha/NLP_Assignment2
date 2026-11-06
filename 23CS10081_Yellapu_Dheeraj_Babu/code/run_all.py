"""Run the whole project (or one task) end to end, in the right order.

Usage (from the project root):
    python code/run_all.py                 # data + Task 1 + Task 2 + Task 3 (full; ~2-4 h on one GPU)
    python code/run_all.py --tasks 1       # one task (its inputs must exist: Task 3 needs Task 2's outputs)
    python code/run_all.py --tasks 2 3 --fast
    python code/run_all.py --dry-run       # print the commands only

Steps (each is an ordinary script that can also be run on its own; see README.md):
  data   common/prepare_data.py           EWT + WS-353 + Google analogies + BERT cache, sanity report
  task1  task1/build_ppmi_svd.py + evaluate_task1.py (+ Levy et al. ablations, summary table)
  task2  task2/extract_bert.py -> train_probes.py -> evaluate_task2.py -> plot_task2.py -> layer_significance.py
  task3  task3/pool_static.py -> evaluate_task3.py -> extensions_task3.py -> diagnostics_task3.py -> plot_task3.py

--fast trains the probes for rank 64 only (26 instead of 78 probes); every other step is identical.
--with-control also runs the spec-literal Task 2 control (constant lr, last epoch, k = 128).
Steps whose outputs already exist are skipped by the scripts themselves where that is cheap to check
(probe checkpoints); re-running is otherwise safe (outputs are overwritten deterministically, seed 42).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

T1_ABLATIONS = {  # tag -> extra flags (Levy, Goldberg & Dagan 2015); "p0" is used by Task 3 as the best T1 variant
    "p0": ["--eig-power", "0"],
    "p05": ["--eig-power", "0.5"],
    "cds075": ["--cds-alpha", "0.75"],
    "cds075_p05": ["--cds-alpha", "0.75", "--eig-power", "0.5"],
}


def steps(args) -> list[tuple[str, list[str]]]:
    s: list[tuple[str, list[str]]] = []
    if "data" in args.tasks:
        s.append(("data", ["code/common/prepare_data.py"]))
    if "1" in args.tasks:
        s.append(("task1", ["code/task1/build_ppmi_svd.py"]))
        s.append(("task1", ["code/task1/evaluate_task1.py"]))
        for tag, flags in T1_ABLATIONS.items():
            if args.skip_ablations and tag != "p0":
                continue
            s.append(("task1", ["code/task1/build_ppmi_svd.py", *flags, "--tag", tag]))
            s.append(("task1", ["code/task1/evaluate_task1.py", "--tag", tag]))
        s.append(("task1", ["code/task1/summarize_task1.py"]))
    if "2" in args.tasks:
        s.append(("task2", ["code/task2/extract_bert.py"]))
        s.append(("task2", ["code/task2/train_probes.py", *(["--ranks", "64"] if args.fast else [])]))
        s.append(("task2", ["code/task2/evaluate_task2.py"]))
        s.append(("task2", ["code/task2/plot_task2.py"]))
        s.append(("task2", ["code/task2/layer_significance.py"]))
        if args.with_control:
            s.append(("task2", ["code/task2/train_probes.py", "--ranks", "128", "--no-decay", "--keep-last",
                                "--tag", "spec_literal"]))
            s.append(("task2", ["code/task2/evaluate_task2.py", "--probe-dir", "outputs/task2/probes_spec_literal",
                                "--out-dir", "outputs/task2/spec_literal"]))
    if "3" in args.tasks:
        s.append(("task3", ["code/task3/pool_static.py"]))
        s.append(("task3", ["code/task3/evaluate_task3.py"]))
        s.append(("task3", ["code/task3/extensions_task3.py"]))
        s.append(("task3", ["code/task3/diagnostics_task3.py"]))
        s.append(("task3", ["code/task3/plot_task3.py"]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["data", "1", "2", "3"], choices=["data", "1", "2", "3"])
    ap.add_argument("--fast", action="store_true", help="Task 2: rank 64 only (26 probes)")
    ap.add_argument("--skip-ablations", action="store_true", help="Task 1: only the p0 variant (needed by Task 3)")
    ap.add_argument("--with-control", action="store_true", help="Task 2: also the spec-literal control run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = steps(args)
    log_dir = ROOT / "outputs" / "run_all_logs"
    t_all = time.time()
    for i, (task, cmd) in enumerate(plan, 1):
        line = " ".join(shlex.quote(c) for c in [Path(PY).name, *cmd])
        print(f"\n[{i}/{len(plan)}] ({task}) {line}", flush=True)
        if args.dry_run:
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{i:02d}_{Path(cmd[0]).stem}{'_' + cmd[-1] if '--tag' in cmd else ''}.log"
        t0 = time.time()
        with open(log, "w") as fh:
            proc = subprocess.Popen([PY, *cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for out_line in proc.stdout:  # stream to the console and to the log
                sys.stdout.write(out_line)
                fh.write(out_line)
            rc = proc.wait()
        print(f"    -> exit {rc} in {time.time() - t0:.0f} s (log: {log.relative_to(ROOT)})", flush=True)
        if rc != 0:
            sys.exit(f"[run_all] step {i} failed; fix the problem and re-run (finished probes are kept).")
    if not args.dry_run:
        print(f"\n[run_all] all {len(plan)} steps done in {(time.time() - t_all) / 60:.1f} min. "
              f"Results: outputs/task1, outputs/task2, outputs/task3 (see README.md).")


if __name__ == "__main__":
    main()
