#!/usr/bin/env python
"""How the target's errors evolve during fine-tuning, and how much a better stopping rule would gain.

For every finished 20-structure run, the change of each error versus MACE-OMAT-0 at every scored
epoch (after 1, 2, 3, 6, 11, 21, 41 and 60 epochs) on
  the target (all MOF-off PBE frames): force MAE, offset-free energy MAE, stress MAE,
  the held-out neighbours (validation set, ranks 21-25 of the selection) and the training set: force MAE.
Runs with several seeds are averaged per target, so each target counts once.
Stopping rules compared (change of target force MAE, and the other target errors at the same epoch):
  fixed      the same epoch for every run
  validation the scored epoch with the lowest validation force MAE (target-blind)
  oracle     the scored epoch with the lowest target force MAE (uses the target; an upper bound)

Training-variant sweep (tta.common.VARIANTS: readouts only, products + readouts, LoRA, stress weight
1000; 30 epochs, seed 123, the 10 search targets): each variant against the default protocol (all
weights, lr 3e-5 and 1e-4) at the same number of epochs, per training set.

  $PY -m mof_embeddings.tta.epochs [--lr 3e-5 1e-4] [--sweep]
  -> data/tta/epochs/curves.csv, stopping.csv, epochs_lr<lr>.png;
     with --sweep: sweep.csv, sweep_paired.csv, sweep.png
"""
import argparse

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import N_TRAIN, SEARCH_TARGETS, SEED, TARGETS, VARIANTS

OUT = DATA / "tta" / "epochs"
TARGET_METRICS = [("target_force_mae", "target force MAE"),
                  ("target_energy_mae_offset_free_meV", "target energy MAE (offset-free)"),
                  ("target_stress_mae_meV_A3", "target stress MAE")]
OTHER = [("valid_force_mae", "held-out neighbours: force MAE"), ("train_force_mae", "training set: force MAE")]
ORDER = ["fgw", "fgwmid", "fgwlow", "global", "random"]


def curves(targets=TARGETS, variants=("",)):
    """Relative change per finished run and scored epoch, for runs of the given training variants."""
    from mof_embeddings.tta.compare import summarise
    from mof_embeddings.tta.prepare import epochs_of
    s = summarise(targets)
    s = s[(s.n_train == N_TRAIN) & s.variant.isin(variants)]
    rows = []
    for (t, sel, lr, seed, variant), g in s.groupby(["target", "selection", "lr", "seed", "variant"]):
        base = g[g.epoch == "foundation"]
        done = g[g.epoch != "foundation"]
        if base.empty or done.epoch.astype(int).max() != epochs_of(variant, 60) - 1:
            continue
        b = base.iloc[0]
        for r in done.itertuples():
            row = dict(target=t, method=r.method, lr=lr, seed=seed, variant=variant, epochs=int(r.epoch) + 1)
            for m, _ in TARGET_METRICS + OTHER:
                row[m] = 100 * (getattr(r, m) / b[m] - 1)
            rows.append(row)
    return pd.DataFrame(rows)


def stopping(c):
    """Per target (seeds averaged): change of the target errors under each stopping rule."""
    rows = []
    for (t, method, lr, seed), g in c.groupby(["target", "method", "lr", "seed"]):
        g = g.sort_values("epochs")
        picks = {f"fixed {e}": g[g.epochs == e] for e in sorted(g.epochs.unique())}
        picks["validation"] = g.loc[[g.valid_force_mae.idxmin()]]
        picks["oracle"] = g.loc[[g.target_force_mae.idxmin()]]
        for rule, r in picks.items():
            r = r.iloc[0]
            rows.append(dict(target=t, method=method, lr=lr, seed=seed, rule=rule, epochs=int(r.epochs),
                             **{m: r[m] for m, _ in TARGET_METRICS}))
    d = pd.DataFrame(rows)
    d = d.groupby(["target", "method", "lr", "rule"]).mean(numeric_only=True).reset_index()
    out = []
    for (method, lr, rule), g in d.groupby(["method", "lr", "rule"]):
        row = dict(method=method, lr=lr, rule=rule, n_targets=len(g), median_epochs=g.epochs.median())
        for m, lab in TARGET_METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_improved"] = int((g[m] < 0).sum())
        out.append(row)
    return pd.DataFrame(out)


def plot(c, lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.tta.compare import COL, INK, INK2, LAB, SURFACE, style
    g = c[np.isclose(c.lr, lr)]
    g = g.groupby(["target", "method", "epochs"]).mean(numeric_only=True).reset_index()
    panels = TARGET_METRICS + OTHER
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), facecolor=SURFACE)
    for ax, (m, lab) in zip(axes, panels):
        for method in [x for x in ORDER if x in set(g.method)]:
            h = g[g.method == method]
            a = h.groupby("epochs")[m]
            mean, lo, hi = a.mean(), a.quantile(0.25), a.quantile(0.75)
            x = np.r_[0.8, mean.index.values]          # starting point: 0 change before training
            ax.plot(x, np.r_[0, mean.values], color=COL[method], lw=1.6, marker="o", ms=3,
                    label=f"{LAB[method]} (n={h.target.nunique()})")
            ax.fill_between(mean.index.values, lo.values, hi.values, color=COL[method], alpha=0.10, lw=0)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 60]); ax.set_xticklabels(["1", "2", "5", "10", "20", "60"])
        ax.set_title(lab, fontsize=9, color=INK)
        ax.set_xlabel("epochs", fontsize=8.5, color=INK2)
        style(ax)
    axes[0].set_ylabel("change vs base model (%)\nmean over targets, band = IQR", fontsize=8.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5, frameon=False)
    fig.suptitle(f"Errors along fine-tuning, 20 structures, lr {lr:g}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / f"epochs_lr{lr:g}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def protocol(variant, lr):
    return f"{VARIANTS[variant]['desc'] if variant else 'all weights'}, lr {lr:g}"


def sweep():
    """Each variant against the default protocol at the same number of epochs (search targets, seed 123)."""
    from scipy.stats import wilcoxon
    c = curves(SEARCH_TARGETS, tuple(VARIANTS))
    c = c[c.seed == SEED]
    c["protocol"] = [protocol(v, lr) for v, lr in zip(c.variant, c.lr)]
    keep = c.variant.ne("") | np.isclose(c.lr, 3e-5) | np.isclose(c.lr, 1e-4)
    c = c[keep]
    metrics = [m for m, _ in TARGET_METRICS]
    tab = c.groupby(["protocol", "method", "epochs"])[metrics].agg(["mean", lambda x: int((x < 0).sum()), "count"])
    tab.columns = [f"{m}_{k}" for m, k in zip(tab.columns.get_level_values(0),
                                             ["mean", "improved", "n"] * len(metrics))]
    tab = tab.reset_index()
    tab.to_csv(OUT / "sweep.csv", index=False)
    ref = protocol("", 3e-5)
    rows = []
    for (proto, method, e), g in c[c.protocol != ref].groupby(["protocol", "method", "epochs"]):
        r = c[(c.protocol == ref) & (c.method == method) & (c.epochs == e)].set_index("target")
        g = g.set_index("target")
        both = g.index.intersection(r.index)
        if len(both) < 5:
            continue
        for m in metrics:
            d = g.loc[both, m] - r.loc[both, m]
            rows.append(dict(protocol=proto, method=method, epochs=e, metric=m, n=len(d), mean_diff=d.mean(),
                             better=int((d < 0).sum()), p=wilcoxon(d).pvalue if (d != 0).any() else 1.0))
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "sweep_paired.csv", index=False)
    plot_sweep(c)
    return tab, pr


def plot_sweep(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.tta.compare import INK, INK2, LAB, SURFACE, style
    protos = sorted(c.protocol.unique(), key=lambda p: (not p.startswith("all weights"), p))
    colors = ["#0b0b0b", "#898781", "#2a78d6", "#1baf7a", "#7c4dc4", "#eb6834", "#e34948"]
    methods = [m for m in ["fgwmid", "fgw", "random"] if m in set(c.method)]
    fig, axes = plt.subplots(len(methods), len(TARGET_METRICS), figsize=(3.6 * len(TARGET_METRICS), 2.9 * len(methods) + 0.6),
                             facecolor=SURFACE, squeeze=False)
    for i, method in enumerate(methods):
        for j, (m, lab) in enumerate(TARGET_METRICS):
            ax = axes[i, j]
            for k, p in enumerate(protos):
                h = c[(c.protocol == p) & (c.method == method)]
                if h.empty:
                    continue
                a = h.groupby("epochs")[m].mean()
                ax.plot(np.r_[0.8, a.index.values], np.r_[0, a.values], color=colors[k % len(colors)], lw=1.5,
                        marker="o", ms=3, ls="--" if p.startswith("all weights") else "-",
                        label=p if (i, j) == (0, 0) else None)
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30, 60]); ax.set_xticklabels(["1", "2", "5", "10", "30", "60"])
            ax.set_title(f"{LAB[method]}: {lab}", fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel("change vs base model (%)", fontsize=8, color=INK2)
            if i == len(methods) - 1:
                ax.set_xlabel("epochs", fontsize=8, color=INK2)
            style(ax)
    fig.legend(loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle("Training variants on the 10 search targets (mean over targets)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "sweep.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5, 1e-4])
    parser.add_argument("--sweep", action="store_true", help="compare the training variants instead")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        tab, pr = sweep()
        pd.set_option("display.width", 220)
        last = tab.groupby(["protocol", "method"]).epochs.transform("max") == tab.epochs
        for m, _ in TARGET_METRICS:
            t = tab[last | tab.epochs.isin([1, 6])].copy()
            t["cell"] = [f"{x:+.1f} ({k}/{n})" for x, k, n in zip(t[f"{m}_mean"], t[f"{m}_improved"], t[f"{m}_n"])]
            print(f"\n== {m}: mean change % (targets improved)")
            print(t.pivot_table(index=["protocol", "epochs"], columns="method", values="cell", aggfunc="first").to_string())
        return
    c = curves()
    c.to_csv(OUT / "curves.csv", index=False)
    st = stopping(c)
    st.to_csv(OUT / "stopping.csv", index=False)
    for lr in args.lr:
        plot(c, lr)
    pd.set_option("display.width", 220)
    for lr in args.lr:
        g = st[np.isclose(st.lr, lr)].copy()
        g["cell"] = [f"{f:+.1f} ({k}/{n})" for f, k, n in zip(g.target_force_mae_mean, g.target_force_mae_improved,
                                                              g.n_targets)]
        rules = [r for r in ["fixed 1", "fixed 2", "fixed 3", "fixed 6", "fixed 11", "fixed 21", "fixed 41",
                             "fixed 60", "validation", "oracle"] if r in set(g.rule)]
        print(f"\n== lr {lr:g}: target force MAE change % (targets improved) by stopping rule")
        print(g.pivot(index="rule", columns="method", values="cell").reindex(rules).to_string())
        h = g[g.rule.isin(["validation", "oracle", "fixed 60"])].copy()
        h["cell"] = [f"{e:+.1f} / {s:+.1f}" for e, s in zip(h.target_energy_mae_offset_free_meV_mean,
                                                           h.target_stress_mae_meV_A3_mean)]
        print("offset-free energy / stress change %:")
        print(h.pivot(index="rule", columns="method", values="cell").to_string())
        v = g[g.rule == "validation"].set_index("method").median_epochs
        print("median epochs chosen by validation:", v.to_dict())


if __name__ == "__main__":
    main()
