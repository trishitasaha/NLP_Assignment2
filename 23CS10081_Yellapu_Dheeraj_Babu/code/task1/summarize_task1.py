"""Collect Task 1 results (spec run + any ablations) into one table.

Usage: python code/task1/summarize_task1.py   ->  outputs/task1/summary_table.csv (+ printed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config as C  # noqa: E402


def row(tag: str, d: Path) -> dict | None:
    if not (d / "results.json").exists():
        return None
    r = json.loads((d / "results.json").read_text())
    m = json.loads((d / "build_meta.json").read_text())
    ws, an, m20, mp = r["wordsim"], r["analogy"], r["mst_20"], r.get("mst_pool") or {}
    return dict(
        run=tag, cds_alpha=m["cds_alpha"], eig_power=m["eig_power"], min_count=m["min_count"],
        ws_rho_cov=ws["spearman_covered"], ws_ci=f"{ws['spearman_covered_ci95'][0]:.3f}..{ws['spearman_covered_ci95'][1]:.3f}",
        ws_rho_full=ws["spearman_full"], ws_rho_freq20=ws.get("by_min_train_freq", {}).get("20", {}).get("spearman"),
        an_acc_cov=an["acc_covered"], an_sem_cov=an["by_kind"]["semantic"]["acc_covered"],
        an_syn_cov=an["by_kind"]["syntactic"]["acc_covered"], an_acc_full=an["acc_full"],
        mst20_uuas=m20["uuas"], mst20_chain=m20["chain_uuas"], mst20_random=m20["random_uuas"],
        pool_uuas=mp.get("uuas"))


def main() -> None:
    rows = [row("spec (alpha=1, p=1)", C.TASK1_OUT)]
    abl = C.TASK1_OUT / "ablations"
    if abl.exists():
        rows += [row(d.name, d) for d in sorted(abl.iterdir()) if d.is_dir()]
    df = pd.DataFrame([r for r in rows if r])
    df.to_csv(C.TASK1_OUT / "summary_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        print(df)


if __name__ == "__main__":
    main()
