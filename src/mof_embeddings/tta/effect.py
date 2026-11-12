#!/usr/bin/env python
"""Do FGW-selected neighbours reduce the fine-tuned model's error on the target? All metrics, with controls.

For every finished 20-structure run (last epoch scored) the relative change of each target metric
after 60 epochs versus MACE-OMAT-0 is computed:
  force MAE, force RMSE (all target frames), force MAE on the query frame,
  energy MAE raw (dominated by the level-of-theory offset) and offset-free (frame-to-frame energies),
  stress MAE.
Runs with several seeds are averaged per target first, so each target counts once. Per selection
method and learning rate: mean change with a 95% bootstrap interval over targets, median, and the
number of targets improved. Paired comparisons (same targets): FGW (alpha 0.5 or 0.1) against random
structures, against the global vector, and alpha 0.1 against alpha 0.5, with the Wilcoxon
signed-rank p-value and the number of targets where the first set gives the lower error.

+U labels. OMat24 follows Materials Project settings: PBE+U for oxides and fluorides of Co, Cr, Fe,
Mn, Mo, Ni, V and W, plain PBE otherwise; the MOF-off target labels have no +U. Each training set is
characterised by the share of structures whose formula falls under that rule (frac_plus_u) and the
share that also contain the target's metal (frac_plus_u_target_metal, from the formula; OMat24's
per-structure settings are not read). Results are also reported separately for targets whose metal
is a +U metal, and the force change is related to frac_plus_u_target_metal for those targets.

  $PY -m mof_embeddings.tta.effect [--lr 3e-5]
  -> data/tta/effect/changes.csv (per run), summary.csv, paired.csv, effect_lr<lr>.png,
     by_metal_group.csv, plus_u_lr<lr>.png
"""
import argparse
import re

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import METHODS, N_TRAIN, TARGETS, run_dir, targets_table

OUT = DATA / "tta" / "effect"
METRICS = [("target_force_mae", "force MAE"), ("target_force_rmse", "force RMSE"),
           ("query_force_mae", "force MAE, query frame"), ("target_energy_mae_meV", "energy MAE (raw)"),
           ("target_energy_mae_offset_free_meV", "energy MAE (offset-free)"),
           ("target_stress_mae_meV_A3", "stress MAE")]
PAIRS = [("fgw", "random"), ("fgwlow", "random"), ("fgw", "global"), ("fgwlow", "global"), ("fgwlow", "fgw")]
U_METALS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}


def elements(formula):
    return set(re.findall(r"[A-Z][a-z]?", formula))


def plus_u(formula):
    e = elements(formula)
    return bool(e & U_METALS) and bool(e & {"O", "F"})


def set_features():
    """+U share of every 20-structure training set (by formula, Materials Project rule)."""
    metal = targets_table().set_index("query").metal
    rows = []
    for t in TARGETS:
        for m in METHODS:
            path = run_dir(t, m) / "sets.csv"
            if not path.exists():
                continue
            f = pd.read_csv(path).query("split == 'train'").formula
            rows.append(dict(target=t, method=m, target_metal=metal[t], target_metal_u=metal[t] in U_METALS,
                             frac_plus_u=f.map(plus_u).mean(),
                             frac_plus_u_target_metal=f.map(lambda x: plus_u(x) and metal[t] in elements(x)).mean()))
    return pd.DataFrame(rows)


def changes(epochs=60):
    from mof_embeddings.tta.compare import summarise
    s = summarise(TARGETS)
    s = s[s.n_train == N_TRAIN]
    base = s[s.epoch == "foundation"].drop_duplicates("target").set_index("target")
    last = s[s.epoch.astype(str) == str(epochs - 1)]
    rows = []
    for r in last.itertuples():
        b = base.loc[r.target]
        rows.append(dict(target=r.target, method=r.method, lr=r.lr, seed=r.seed,
                         **{m: 100 * (getattr(r, m) / b[m] - 1) for m, _ in METRICS}))
    return pd.DataFrame(rows)


def per_target(ch):
    keys = ["target", "method", "lr", "target_metal_u", "frac_plus_u", "frac_plus_u_target_metal"]
    return ch.groupby(keys)[[m for m, _ in METRICS]].mean().reset_index()


def by_metal_group(pt):
    rows = []
    for (u, method, lr), g in pt.groupby(["target_metal_u", "method", "lr"]):
        for m, lab in METRICS:
            x = g[m].dropna()
            lo, hi = bootstrap_ci(x)
            rows.append(dict(target_metal="+U metal (Co Cr Fe Mn Mo Ni V W)" if u else "other metal",
                             method=method, lr=lr, metric=lab, n_targets=len(x), mean=x.mean(), ci_lo=lo,
                             ci_hi=hi, improved=int((x < 0).sum())))
    return pd.DataFrame(rows)


def plot_plus_u(pt, lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from mof_embeddings.tta.compare import COL, INK, INK2, LAB, SURFACE, style
    g = pt[np.isclose(pt.lr, lr) & pt.target_metal_u & (pt.method != "random")]
    if len(g) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, (m, lab) in zip(axes, [("target_force_mae", "force MAE"),
                                   ("target_energy_mae_offset_free_meV", "energy MAE (offset-free)")]):
        for meth, h in g.groupby("method"):
            ax.scatter(h.frac_plus_u_target_metal, h[m], s=28, color=COL[meth], label=LAB[meth], zorder=3,
                       edgecolor="white", lw=0.5)
            for x, y, t in zip(h.frac_plus_u_target_metal, h[m], h.target):
                ax.annotate(t, (x, y), fontsize=6, color=INK2, xytext=(2, 2), textcoords="offset points")
        rho = spearmanr(g.frac_plus_u_target_metal, g[m]).statistic
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{lab}: Spearman ρ = {rho:+.2f} (n = {len(g)})", fontsize=9, color=INK)
        ax.set_xlabel("training structures labelled with +U on the target's metal", fontsize=8.5, color=INK2)
        ax.set_ylabel("change vs base model (%)", fontsize=8.5, color=INK2)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Targets with a +U metal, neighbour sets, lr {lr:g}", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(OUT / f"plus_u_lr{lr:g}.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summary(pt):
    rows = []
    for (method, lr), g in pt.groupby(["method", "lr"]):
        for m, lab in METRICS:
            x = g[m].dropna()
            lo, hi = bootstrap_ci(x)
            rows.append(dict(method=method, lr=lr, metric=lab, n_targets=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                             median=x.median(), improved=int((x < 0).sum())))
    return pd.DataFrame(rows)


def paired(pt):
    from scipy.stats import wilcoxon
    rows = []
    for lr, g in pt.groupby("lr"):
        w = g.pivot_table(index="target", columns="method", values=[m for m, _ in METRICS])
        for a, b in PAIRS:
            for m, lab in METRICS:
                if (m, a) not in w or (m, b) not in w:
                    continue
                d = (w[(m, a)] - w[(m, b)]).dropna()
                if len(d) < 3:
                    continue
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append(dict(lr=lr, a=a, b=b, metric=lab, n_targets=len(d), mean_diff=d.mean(),
                                 median_diff=d.median(), a_better=int((d < 0).sum()), p_wilcoxon=p))
    return pd.DataFrame(rows)


def plot(sm, lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.tta.compare import COL, INK, INK2, LAB, SURFACE, style
    g = sm[np.isclose(sm.lr, lr)]
    methods = [m for m in ["fgw", "fgwlow", "fgwmid", "global", "random"] if m in set(g.method)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), facecolor=SURFACE)
    for ax, (_, lab) in zip(axes.flat, METRICS):
        h = g[g.metric == lab]
        for i, m in enumerate(methods):
            r = h[h.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=COL[m], lw=2, solid_capstyle="round")
            ax.scatter([r["mean"]], [i], s=40, color=COL[m], zorder=3)
            ax.annotate(f"{r['mean']:+.1f}%  ({r.improved}/{r.n_targets})", (r.ci_hi, i), xytext=(5, -3),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([LAB[m] for m in methods], fontsize=7.5)
        ax.invert_yaxis(); ax.set_title(lab, fontsize=9.5, color=INK)
        ax.set_xlabel("change vs base model (%), mean and 95% CI", fontsize=7.5, color=INK2)
        style(ax)
        x0, x1 = ax.get_xlim(); ax.set_xlim(x0, x1 + 0.35 * (x1 - x0))
    fig.suptitle(f"Fine-tuning on 20 structures, lr {lr:g}, 60 epochs: change of target errors "
                 "(negative = better; in brackets: targets improved)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(OUT / f"effect_lr{lr:g}.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ch = changes().merge(set_features(), on=["target", "method"], how="left")
    ch.to_csv(OUT / "changes.csv", index=False)
    pt = per_target(ch)
    grp = by_metal_group(pt)
    grp.to_csv(OUT / "by_metal_group.csv", index=False)
    sm, pr = summary(pt), paired(pt)
    sm.to_csv(OUT / "summary.csv", index=False)
    pr.to_csv(OUT / "paired.csv", index=False)
    for lr in args.lr:
        plot(sm, lr)
        plot_plus_u(pt, lr)
    pd.set_option("display.width", 220)
    for lr in args.lr:
        print(f"\n== lr {lr:g}: mean change % [95% CI], targets improved")
        g = sm[np.isclose(sm.lr, lr)].copy()
        g["cell"] = [f"{m:+.1f} [{lo:+.1f},{hi:+.1f}] {k}/{n}" for m, lo, hi, k, n in
                     zip(g["mean"], g.ci_lo, g.ci_hi, g.improved, g.n_targets)]
        print(g.pivot(index="metric", columns="method", values="cell").reindex([l for _, l in METRICS]).to_string())
        q = pr[np.isclose(pr.lr, lr)].copy()
        q["cell"] = [f"{d:+.1f} ({k}/{n}) p={p:.3f}" for d, k, n, p in zip(q.mean_diff, q.a_better, q.n_targets,
                                                                         q.p_wilcoxon)]
        q["pair"] = q.a + " vs " + q.b
        print(q.pivot(index="metric", columns="pair", values="cell").reindex([l for _, l in METRICS]).to_string())
        h = grp[np.isclose(grp.lr, lr)].copy()
        h["cell"] = [f"{m:+.1f} [{lo:+.1f},{hi:+.1f}] {k}/{n}" for m, lo, hi, k, n in
                     zip(h["mean"], h.ci_lo, h.ci_hi, h.improved, h.n_targets)]
        for metric in ["force MAE", "energy MAE (offset-free)", "energy MAE (raw)"]:
            print(f"-- {metric} by target metal")
            print(h[h.metric == metric].pivot(index="target_metal", columns="method", values="cell").to_string())


if __name__ == "__main__":
    main()
