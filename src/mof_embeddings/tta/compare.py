#!/usr/bin/env python
"""Summarise test-time adaptation runs: accuracy on the target before and after fine-tuning.

For every target, selection (fgw / random / global) and scored epoch, and for the foundation model:
  force MAE and RMSE over all target frames (eV/A) and on the query frame alone,
  energy MAE per atom, raw and after removing the mean per-atom offset over the target's frames
    (MOF-off PBE is a D3BJ dataset without +U, OMat24 is PBE(+U) without dispersion, so a constant
    offset is expected and fine-tuning on OMat24 cannot remove it; the offset-free error measures
    the shape of the energy surface),
  stress MAE (meV/A^3), and the same force/energy metrics on the selection's validation frames.
The target-blind choice is the epoch with the lowest validation loss logged by mace_run_train.

  $PY -m mof_embeddings.tta.compare [--targets Q6 Q3 Q4 Q1]
  -> data/tta/summary.csv, data/tta/target_force_mae.png
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import SELECTIONS, TARGETS, run_dir, target_dir

OUT = DATA / "tta"


def metrics(df, prefix):
    if df.empty:
        return {}
    n = 3 * df.natoms.sum()
    err = (df.energy_pred - df.energy_ref) / df.natoms
    return {f"{prefix}_force_mae": df.force_abs_sum.sum() / n, f"{prefix}_force_rmse": np.sqrt(df.force_sq_sum.sum() / n),
            f"{prefix}_energy_mae_meV": float(np.abs(err).mean() * 1e3),
            f"{prefix}_energy_mae_offset_free_meV": float(np.abs(err - err.mean()).mean() * 1e3),
            f"{prefix}_energy_offset_meV": float(err.mean() * 1e3),
            f"{prefix}_stress_mae_meV_A3": float(df.stress_mae.mean() * 1e3)}


def best_epoch(target, sel):
    path = next((run_dir(target, sel) / "run" / "results").glob("*_train.txt"), None)
    if path is None:
        return None
    ev = [json.loads(l) for l in path.read_text().splitlines()]
    ev = [r for r in ev if r.get("mode") == "eval" and r.get("epoch") is not None]
    return min(ev, key=lambda r: r["loss"])["epoch"] if ev else None


def summarise(targets):
    rows = []
    for t in targets:
        tgt = pd.read_csv(target_dir(t) / "target.csv")
        query_frame = int(np.flatnonzero(tgt.is_query.values)[0])
        found = pd.read_csv(target_dir(t) / "foundation_eval.csv")
        for sel in SELECTIONS:
            evals = {"foundation": found.assign(set=found.set.replace({f"valid_{sel}": "valid", f"train_{sel}": "train"}))}
            for p in sorted(run_dir(t, sel).glob("eval_epoch*.csv")):
                evals[int(re.search(r"epoch(\d+)", p.name).group(1))] = pd.read_csv(p)
            best = best_epoch(t, sel)
            for epoch, df in evals.items():
                tg = df[df.set == "target"]
                row = dict(target=t, selection=sel, epoch=epoch, n_target_frames=len(tg),
                           best_validation_epoch=best,
                           **metrics(tg, "target"), **metrics(tg[tg.frame == query_frame], "query"),
                           **metrics(df[df.set == "valid"], "valid"), **metrics(df[df.set == "train"], "train"))
                rows.append(row)
    return pd.DataFrame(rows)


def plot(df, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    COL = {"fgw": "#2a78d6", "global": "#eb6834", "random": "#898781"}
    LAB = {"fgw": "20 nearest by FGW", "global": "20 nearest by global vector", "random": "20 random OMat24"}
    fig, axes = plt.subplots(2, len(targets), figsize=(3.6 * len(targets), 6.2), facecolor=SURFACE, squeeze=False)
    for c, t in enumerate(targets):
        for r, (metric, label) in enumerate([("target_force_mae", "target force MAE (eV/Å)"),
                                             ("target_energy_mae_offset_free_meV", "target energy MAE,\noffset removed (meV/atom)")]):
            ax = axes[r, c]
            sub = df[df.target == t]
            base = sub[sub.epoch == "foundation"][metric].iloc[0]
            ax.axhline(base, color=INK, lw=1, ls="--", label="MACE-OMAT-0 (no fine-tuning)")
            for sel in SELECTIONS:
                s = sub[(sub.selection == sel) & (sub.epoch != "foundation")].copy()
                if s.empty:
                    continue
                s["e"] = s.epoch.astype(int) + 1
                s = s.sort_values("e")
                ax.plot(s.e, s[metric], marker="o", ms=3.5, color=COL[sel], label=LAB[sel])
                b = s.best_validation_epoch.iloc[0]
                if pd.notna(b) and (s.epoch.astype(int) == int(b)).any():
                    v = s[s.epoch.astype(int) == int(b)][metric].iloc[0]
                    ax.scatter([int(b) + 1], [v], s=70, facecolor="none", edgecolor=COL[sel], lw=1.6, zorder=4)
            ax.set_xscale("log"); ax.set_xlabel("training epochs", fontsize=9, color=INK2)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=INK2)
            if r == 0:
                q = df[df.target == t]
                ax.set_title(f"{t}  ({int(q.n_target_frames.iloc[0])} PBE frames)", fontsize=10, color=INK)
            ax.tick_params(labelsize=8, colors=INK2); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
            for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(OUT / "target_force_mae.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = summarise(args.targets)
    df.to_csv(OUT / "summary.csv", index=False)
    plot(df, args.targets)
    cols = ["target", "selection", "epoch", "best_validation_epoch", "target_force_mae", "query_force_mae",
            "target_energy_mae_meV", "target_energy_mae_offset_free_meV", "valid_force_mae", "train_force_mae"]
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
