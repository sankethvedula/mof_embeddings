#!/usr/bin/env python
"""Summarise test-time adaptation runs: accuracy on the target before and after fine-tuning.

For every target, selection and scored epoch, and for the foundation model:
  force MAE and RMSE over all target frames (eV/A) and on the query frame alone,
  energy MAE per atom, raw and after removing the mean per-atom offset over the target's frames
    (MOF-off PBE is a D3BJ dataset without +U, OMat24 is PBE(+U) without dispersion, so a constant
    offset is expected and fine-tuning on OMat24 cannot remove it; the offset-free error measures
    the shape of the energy surface),
  stress MAE (meV/A^3), and the same force/energy metrics on the selection's validation frames.
The target-blind choice is the epoch with the lowest validation loss logged by mace_run_train.

  $PY -m mof_embeddings.tta.compare [--targets Q6 Q3 ...]
  -> data/tta/summary.csv (every scored epoch), final.csv (before vs last epoch),
     target_force_mae.png, target_energy_offset_free.png, target_energy_raw.png, final_force_change.png,
     convergence.png (training loss, validation and target errors per epoch; convergence_100.png for 100,
     convergence_lr1e-05.png for another learning rate), target_force_mae_lr.png (learning rates compared)
Selections are found on disk: fgw, random, global (20 structures) and e.g. fgw100 (100 structures).
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import (LR, METHODS, N_TRAIN, SEED, TARGETS, out_dir, parse_selection, run_params,
                                       selection_name, target_dir)

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


def best_epoch(target, sel, lr=LR, seed=SEED, variant=""):
    path = next((out_dir(target, sel, lr, seed, variant) / "run" / "results").glob("*_train.txt"), None)
    if path is None:
        return None
    ev = [json.loads(l) for l in path.read_text().splitlines()]
    ev = [r for r in ev if r.get("mode") == "eval" and r.get("epoch") is not None]
    return min(ev, key=lambda r: r["loss"])["epoch"] if ev else None


def runs(target):
    """(selection, learning rate, seed, variant) of every run with scored epochs, 20-structure sets first."""
    found = []
    for d in target_dir(target).iterdir():
        if not d.is_dir() or d.name == "run":
            continue
        for r in [d] + sorted(x for x in d.iterdir() if x.is_dir() and x.name != "run"):
            if any(r.glob("eval_epoch*.csv")):
                found.append((d.name, *run_params(r)))
    return sorted(found, key=lambda x: (parse_selection(x[0])[1], METHODS.index(parse_selection(x[0])[0]), -x[1], x[2]))


def summarise(targets):
    rows = []
    for t in targets:
        tgt = pd.read_csv(target_dir(t) / "target.csv")
        query_frame = int(np.flatnonzero(tgt.is_query.values)[0])
        if not (target_dir(t) / "foundation_eval.csv").exists():
            continue
        found = pd.read_csv(target_dir(t) / "foundation_eval.csv")
        for sel, lr, seed, variant in runs(t):
            method, n_train = parse_selection(sel)
            evals = {"foundation": found.assign(set=found.set.replace({f"valid_{sel}": "valid", f"train_{sel}": "train"}))}
            for p in sorted(out_dir(t, sel, lr, seed, variant).glob("eval_epoch*.csv")):
                evals[int(re.search(r"epoch(\d+)", p.name).group(1))] = pd.read_csv(p)
            best = best_epoch(t, sel, lr, seed, variant)
            for epoch, df in evals.items():
                tg = df[df.set == "target"]
                row = dict(target=t, selection=sel, method=method, n_train=n_train, lr=lr, seed=seed,
                           variant=variant, epoch=epoch,
                           n_target_frames=len(tg), best_validation_epoch=best,
                           **metrics(tg, "target"), **metrics(tg[tg.frame == query_frame], "query"),
                           **metrics(df[df.set == "valid"], "valid"), **metrics(df[df.set == "train"], "train"))
                rows.append(row)
    return pd.DataFrame(rows)


INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
COL = {"fgw": "#2a78d6", "global": "#eb6834", "random": "#898781", "fgwmid": "#1baf7a", "fgwlow": "#7c4dc4"}
LAB = {"fgw": "nearest by FGW (α 0.5)", "global": "nearest by global vector", "random": "random OMat24",
       "fgwmid": "nearest by FGW (α 0.25)", "fgwlow": "nearest by FGW (α 0.1)"}


def style(ax):
    ax.tick_params(labelsize=8, colors=INK2); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)


LS_N = {N_TRAIN: "-", 100: "--"}
LS_LR = {1e-4: "-", 3e-5: "--", 1e-5: ":"}


def plot_curves(df, targets, metric, ylabel, fname, vary="n_train"):
    """Target error along training, one panel per target, colour by selection method.
    vary="n_train": default learning rate, solid 20 structures, dashed 100;
    vary="lr": 20 structures, line style by learning rate."""
    df = df[df.lr == LR] if vary == "n_train" else df[df.n_train == N_TRAIN]
    styles = LS_N if vary == "n_train" else LS_LR
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = min(5, len(targets)); nrow = -(-len(targets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow + 0.5), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(targets):]:
        ax.set_visible(False)
    for ax, t in zip(axes.flat, targets):
        sub = df[df.target == t]
        if sub.empty:
            ax.set_visible(False); continue
        ax.axhline(sub[sub.epoch == "foundation"][metric].iloc[0], color=INK, lw=1, ls="--")
        for (sel, lr), s in sub[sub.epoch != "foundation"].groupby(["selection", "lr"], sort=False):
            s = s.copy()
            method, n = parse_selection(sel)
            ls = styles.get(n if vary == "n_train" else lr, "-.")
            s["e"] = s.epoch.astype(int) + 1
            s = s.sort_values("e")
            ax.plot(s.e, s[metric], marker="o", ms=3, color=COL[method], ls=ls, lw=1.4)
            b = s.best_validation_epoch.iloc[0]
            if pd.notna(b) and (s.epoch.astype(int) == int(b)).any():
                v = s[s.epoch.astype(int) == int(b)][metric].iloc[0]
                ax.scatter([int(b) + 1], [v], s=60, facecolor="none", edgecolor=COL[method], lw=1.4, zorder=4)
        ax.set_xscale("log")
        q = sub.iloc[0]
        ax.set_title(f"{t}  ({int(q.n_target_frames)} PBE frames)", fontsize=9.5, color=INK)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel("training epochs", fontsize=8.5, color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, ls="--", lw=1, label="MACE-OMAT-0 (no fine-tuning)")]
    handles += [Line2D([], [], color=COL[m], lw=1.6, label=LAB[m]) for m in METHODS]
    if vary == "n_train":
        handles += [Line2D([], [], color=INK2, ls=ls, lw=1.4, label=f"{n} training structures") for n, ls in LS_N.items()]
    else:
        handles += [Line2D([], [], color=INK2, ls=ls, lw=1.4, label=f"lr {lr:g}") for lr, ls in LS_LR.items()
                    if lr in set(df.lr)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / fname, dpi=150, facecolor=SURFACE); plt.close(fig)


def final_table(df):
    """Relative change of the target errors after the last scored epoch, per target and selection."""
    rows = []
    for (t, sel, lr, seed, variant), g in df[df.epoch != "foundation"].groupby(
            ["target", "selection", "lr", "seed", "variant"], sort=False):
        last = g.loc[g.epoch.astype(int).idxmax()]
        base = df[(df.target == t) & (df.selection == sel) & (df.epoch == "foundation")].iloc[0]
        rows.append(dict(target=t, selection=sel, method=last.method, n_train=last.n_train, lr=lr, seed=seed,
                         variant=variant,
                         last_epoch=int(last.epoch),
                         force_before=base.target_force_mae, force_after=last.target_force_mae,
                         force_change_pct=100 * (last.target_force_mae / base.target_force_mae - 1),
                         energy_before=base.target_energy_mae_meV, energy_after=last.target_energy_mae_meV,
                         energy_offset_free_before=base.target_energy_mae_offset_free_meV,
                         energy_offset_free_after=last.target_energy_mae_offset_free_meV))
    return pd.DataFrame(rows)


def plot_final(fin, targets):
    """Change of the target force MAE after training (default learning rate): one row per target,
    one marker per selection."""
    fin = fin[fin.lr == LR]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(targets) + 1.2), facecolor=SURFACE)
    sels = [(m, n) for n in sorted(fin.n_train.unique()) for m in METHODS]
    for i, t in enumerate(targets):
        for k, (m, n) in enumerate(sels):
            r = fin[(fin.target == t) & (fin.method == m) & (fin.n_train == n)]
            if r.empty:
                continue
            y = i + (k - (len(sels) - 1) / 2) * 0.11
            ax.scatter(r.force_change_pct, [y], s=34, color=COL[m], marker="o" if n == N_TRAIN else "D",
                       facecolor=COL[m] if n == N_TRAIN else "none", lw=1.4, zorder=3,
                       label=f"{LAB[m]}, {n}" if i == 0 else None)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets); ax.invert_yaxis()
    ax.set_xlabel("change in target force MAE after 60 epochs (%)", fontsize=9, color=INK2)
    style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); fig.savefig(OUT / "final_force_change.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def training_log(target, sel, lr=LR, seed=SEED):
    """Per-epoch training loss (mean over batches) and validation errors logged by mace_run_train."""
    path = next((out_dir(target, sel, lr, seed) / "run" / "results").glob("*_train.txt"), None)
    if path is None:
        return pd.DataFrame()
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    opt = pd.DataFrame([r for r in recs if r.get("mode") == "opt"]).groupby("epoch").loss.mean().rename("train_loss")
    ev = pd.DataFrame([r for r in recs if r.get("mode") == "eval" and r.get("epoch") is not None]).set_index("epoch")
    ev = ev[["loss", "mae_f", "mae_e_per_atom"]].rename(columns={"loss": "valid_loss", "mae_f": "valid_force_mae",
                                                               "mae_e_per_atom": "valid_energy_mae"})
    return ev.join(opt, how="outer").reset_index()


def plot_convergence(df, targets, n_train=N_TRAIN, fname="convergence.png", lr=LR):
    """Rows: training loss, validation force and energy MAE (held-out neighbours, from the trainer),
    target force MAE (scored epochs). One column per target, one colour per selection method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = [("train_loss", "training loss\n(mean over batches)"),
            ("valid_force_mae", "validation force MAE\n(eV/Å, held-out neighbours)"),
            ("valid_energy_mae", "validation energy MAE\n(meV/atom, held-out neighbours)"),
            ("target_force_mae", "target force MAE\n(eV/Å, MOF frames)")]
    fig, axes = plt.subplots(len(rows), len(targets), figsize=(2.75 * len(targets), 2.3 * len(rows) + 0.5),
                             facecolor=SURFACE, squeeze=False)
    for c, t in enumerate(targets):
        sub = df[(df.target == t) & (df.n_train == n_train) & (df.lr == lr)]
        base = sub[sub.epoch == "foundation"]
        for method in METHODS:
            sel = selection_name(method, n_train)
            log = training_log(t, sel, lr)
            if log.empty:
                continue
            for r, (col, _) in enumerate(rows[:3]):
                y = log[col] * (1e3 if col == "valid_energy_mae" else 1)
                axes[r, c].plot(log.epoch + 1, y, color=COL[method], lw=1.3)
            s = sub[(sub.selection == sel) & (sub.epoch != "foundation")].copy()
            s["e"] = s.epoch.astype(int) + 1
            s = s.sort_values("e")
            axes[3, c].plot(s.e, s.target_force_mae, marker="o", ms=3, color=COL[method], lw=1.3)
            b = s.best_validation_epoch.iloc[0] if len(s) else None
            if b is not None and pd.notna(b):
                for r in range(3):
                    axes[r, c].axvline(int(b) + 1, color=COL[method], lw=0.8, ls=":")
        if len(base):
            axes[3, c].axhline(base.target_force_mae.iloc[0], color=INK, lw=1, ls="--")
        axes[0, c].set_yscale("log")
        axes[0, c].set_title(f"{t}  ({int(sub.n_target_frames.iloc[0])} PBE frames)" if len(sub) else t,
                             fontsize=9.5, color=INK)
        for r in range(len(rows)):
            style(axes[r, c])
        axes[-1, c].set_xlabel("epoch", fontsize=8.5, color=INK2)
    for r, (_, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab, fontsize=8, color=INK2)
    handles = [Line2D([], [], color=COL[m], lw=1.6, label=f"{n_train} {LAB[m]}, lr {lr:g}") for m in METHODS]
    handles += [Line2D([], [], color=INK, ls="--", lw=1, label="MACE-OMAT-0 (no fine-tuning)"),
                Line2D([], [], color=INK2, ls=":", lw=1, label="best validation epoch")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / fname, dpi=150, facecolor=SURFACE); plt.close(fig)


def seed_summary(fin):
    """Mean and spread over seeds of the target force-MAE change, for runs repeated with several seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = fin[fin.last_epoch == fin.last_epoch.max()].groupby(["target", "selection", "method", "lr"])
    s = g.force_change_pct.agg(n_seeds="count", mean="mean", sd="std", lo="min", hi="max").reset_index()
    s = s[s.n_seeds > 1]
    s.to_csv(OUT / "seeds.csv", index=False)
    if s.empty:
        return
    for lr, h in s.groupby("lr"):
        targets = [t for t in TARGETS if t in set(h.target)]
        methods = [m for m in METHODS if m in set(h.method)]
        fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(targets) + 1.3), facecolor=SURFACE)
        for i, t in enumerate(targets):
            for k, m in enumerate(methods):
                r = h[(h.target == t) & (h.method == m)]
                runs_ = fin[(fin.target == t) & (fin.method == m) & np.isclose(fin.lr, lr)
                            & (fin.selection == m)]
                if r.empty:
                    continue
                y = i + (k - (len(methods) - 1) / 2) * 0.14
                ax.plot([r.lo.iloc[0], r.hi.iloc[0]], [y, y], color=COL[m], lw=1.2, alpha=0.6)
                ax.scatter(runs_.force_change_pct, [y] * len(runs_), s=10, color=COL[m], alpha=0.6, lw=0)
                ax.scatter(r["mean"], [y], s=30, color=COL[m], zorder=3, label=LAB[m] if i == 0 else None)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets, fontsize=7.5); ax.invert_yaxis()
        ax.set_xlabel(f"change in target force MAE after 60 epochs (%), lr {lr:g}; dots = seeds, "
                      "large dot = mean", fontsize=8.5, color=INK2)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout(); fig.savefig(OUT / f"seeds_lr{lr:g}.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    every = summarise(args.targets)
    targets = [t for t in args.targets if t in set(every.target)]
    every.to_csv(OUT / "summary.csv", index=False)
    fin_all = final_table(every)
    fin_all.to_csv(OUT / "final.csv", index=False)
    seed_summary(fin_all[fin_all.variant == ""])
    df = every[(every.seed == SEED) & (every.variant == "")]              # curves: default seed and protocol
    fin = fin_all[(fin_all.seed == SEED) & (fin_all.variant == "")]
    plot_curves(df, targets, "target_force_mae", "target force MAE (eV/Å)", "target_force_mae.png")
    plot_curves(df, targets, "target_energy_mae_offset_free_meV", "target energy MAE,\noffset removed (meV/atom)",
                "target_energy_offset_free.png")
    plot_curves(df, targets, "target_energy_mae_meV", "target energy MAE (meV/atom)", "target_energy_raw.png")
    plot_final(fin, targets)
    for n, lr in df[["n_train", "lr"]].drop_duplicates().itertuples(index=False):
        name = "convergence" + ("" if n == N_TRAIN else f"_{n}") + ("" if lr == LR else f"_lr{lr:g}")
        plot_convergence(df, targets, int(n), name + ".png", lr)
    if df.lr.nunique() > 1:
        plot_curves(df, targets, "target_force_mae", "target force MAE (eV/Å)", "target_force_mae_lr.png", vary="lr")
        plot_curves(df, targets, "target_energy_mae_offset_free_meV", "target energy MAE,\noffset removed (meV/atom)",
                    "target_energy_offset_free_lr.png", vary="lr")
    print(fin.assign(lr=fin.lr.map("{:g}".format)).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
