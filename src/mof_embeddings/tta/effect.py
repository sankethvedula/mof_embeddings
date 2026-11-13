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

One selection on many targets (--cohorts fgwmid): every target with that selection at --lr, the
change after 6 and after 60 epochs, by the round that added the target (search queries Q1-Q10,
round 3: S targets with all five selections, round 4: further S targets with this selection only)
and by target metal (+U metal or not).

  $PY -m mof_embeddings.tta.effect [--lr 3e-5]
  -> data/tta/effect/changes.csv (per run), summary.csv, paired.csv, effect_lr<lr>.png,
     by_metal_group.csv, plus_u_lr<lr>.png
  $PY -m mof_embeddings.tta.effect --cohorts fgwmid [--lr 3e-5]
  -> data/tta/effect/cohorts_<method>.csv (per target), cohorts_<method>_summary.csv,
     cohorts_<method>.png

Weight interpolation (--interp fgwmid): the models of tta.interpolate, w = (1 - lambda) w_foundation +
lambda w_finetuned(epoch), scored like the runs; change per lambda and epoch over all targets and by
target metal. lambda = 0 and 1 are checked against the foundation and fine-tuned evaluations.
  $PY -m mof_embeddings.tta.effect --interp fgwmid [--lr 3e-5]
  -> data/tta/effect/interp_<method>.csv (per target), interp_<method>_summary.csv, interp_<method>.png
"""
import argparse

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import METHODS, N_TRAIN, TARGETS, U_METALS, elements, plus_u, run_dir, targets_table

OUT = DATA / "tta" / "effect"
METRICS = [("target_force_mae", "force MAE"), ("target_force_rmse", "force RMSE"),
           ("query_force_mae", "force MAE, query frame"), ("target_energy_mae_meV", "energy MAE (raw)"),
           ("target_energy_mae_offset_free_meV", "energy MAE (offset-free)"),
           ("target_stress_mae_meV_A3", "stress MAE")]
PAIRS = [("fgw", "random"), ("fgwlow", "random"), ("fgw", "global"), ("fgwlow", "global"), ("fgwlow", "fgw"),
         ("fgwmidnou", "fgwmid"), ("moffgw", "fgwmid"), ("moffgw", "fgwmidnou"),
         ("moffgw", "mofrandom"), ("moffgw", "mofglobal"), ("moffgwnf", "mofrandom"), ("mofglobal", "mofrandom")]


def set_features():
    """+U share of every 20-structure training set (by formula, Materials Project rule)."""
    metal = targets_table().set_index("query").metal
    rows = []
    for t in TARGETS:
        for m in METHODS:
            path = run_dir(t, m) / "sets.csv"
            same = run_dir(t, m) / "same_as.txt"
            if same.exists():                    # identical set stored under another selection
                path = run_dir(t, same.read_text().strip()) / "sets.csv"
            if not path.exists():
                continue
            f = pd.read_csv(path).query("split == 'train'").formula
            rows.append(dict(target=t, method=m, target_metal=metal[t], target_metal_u=metal[t] in U_METALS,
                             frac_plus_u=f.map(plus_u).mean(),
                             frac_plus_u_target_metal=f.map(lambda x: plus_u(x) and metal[t] in elements(x)).mean()))
    return pd.DataFrame(rows)


def changes(epochs=60, s=None):
    """% change of every metric after `epochs` epochs versus the foundation model, per run."""
    from mof_embeddings.tta.compare import summarise
    s = summarise(TARGETS) if s is None else s
    s = s[s.n_train == N_TRAIN]
    base = s[s.epoch == "foundation"].drop_duplicates("target").set_index("target")
    last = s[s.epoch.astype(str) == str(epochs - 1)]
    rows = []
    for r in last.itertuples():
        b = base.loc[r.target]
        rows.append(dict(target=r.target, method=r.method, lr=r.lr, seed=r.seed,
                         **{m: 100 * (getattr(r, m) / b[m] - 1) for m, _ in METRICS}))
        for m in METHODS:                        # a selection whose set is identical (same_as.txt) shares the run
            same = run_dir(r.target, m) / "same_as.txt"
            if r.n_train == N_TRAIN and same.exists() and same.read_text().strip() == r.selection:
                rows.append({**rows[-1], "method": m})
    return pd.DataFrame(rows, columns=["target", "method", "lr", "seed"] + [m for m, _ in METRICS])


def cohorts(method, lr):
    """Change of every metric for one selection on all its targets, after 6 and 60 epochs."""
    from mof_embeddings.tta.compare import summarise
    s = summarise(TARGETS)
    t = targets_table().set_index("query")
    rnd = t["round"] if "round" in t else pd.Series(3, index=t.index)
    per = []
    for ep in [6, 60]:
        ch = changes(ep, s)
        ch = ch[(ch.method == method) & np.isclose(ch.lr, lr)]
        ch = ch.groupby("target")[[m for m, _ in METRICS]].mean().reset_index()
        ch["epochs"] = ep
        per.append(ch)
    per = pd.concat(per, ignore_index=True)
    per["cohort"] = ["search queries" if q.startswith("Q") else f"round {int(rnd[q])} S targets"
                     for q in per.target]
    per["metal"] = t.metal.reindex(per.target).values
    per["natoms"] = t.natoms.reindex(per.target).values
    per["metal_group"] = np.where(per.metal.isin(U_METALS), "+U metal", "other metal")
    rows = []
    for ep, g in per.groupby("epochs"):
        for key, col in [("all", None), ("cohort", "cohort"), ("metal", "metal_group")]:
            for name, h in ([("all targets", g)] if col is None else g.groupby(col)):
                for m, lab in METRICS:
                    x = h[m].dropna()
                    lo, hi = bootstrap_ci(x)
                    rows.append(dict(epochs=ep, group=key, subset=name, metric=lab, n_targets=len(x),
                                     mean=x.mean(), ci_lo=lo, ci_hi=hi, median=x.median(),
                                     improved=int((x < 0).sum())))
    return per, pd.DataFrame(rows)


def interp(method, lr):
    """% change versus the foundation model of every interpolated model, per target, lambda and epoch."""
    import re as _re
    from mof_embeddings.tta.compare import metrics
    from mof_embeddings.tta.common import out_dir, target_dir
    t = targets_table().set_index("query")
    rows, check = [], []
    for target in TARGETS:
        d = out_dir(target, method, lr) / "interp"
        f = target_dir(target) / "foundation_eval.csv"
        if not d.exists() or not f.exists():
            continue
        found = pd.read_csv(f)
        b = metrics(found[found.set == "target"], "target")
        for path in sorted(d.glob("epoch*_l*.csv")):
            ep, lam = _re.fullmatch(r"epoch(\d+)_l([0-9.]+)\.csv", path.name).groups()
            m = metrics(pd.read_csv(path), "target")
            rows.append(dict(target=target, epochs=int(ep) + 1, lam=float(lam), metal=t.metal[target],
                             **{k: 100 * (m[k] / b[k] - 1) for k, _ in METRICS if k.startswith("target")}))
            ref = out_dir(target, method, lr) / f"eval_epoch{ep}.csv"
            if float(lam) in (0.0, 1.0) and ref.exists():
                r = pd.read_csv(ref)
                r = b if float(lam) == 0 else metrics(r[r.set == "target"], "target")
                check.append(abs(m["target_force_mae"] / r["target_force_mae"] - 1))
    per = pd.DataFrame(rows)
    per["metal_group"] = np.where(per.metal.isin(U_METALS), "+U metal", "other metal")
    print(f"check: lambda 0 / 1 reproduce the stored force MAE to {max(check) * 100:.3f}% (max over {len(check)})")
    out = []
    for (ep, lam), g in per.groupby(["epochs", "lam"]):
        for name, h in [("all targets", g)] + list(g.groupby("metal_group")):
            for m, lab in METRICS:
                if m not in h:
                    continue
                x = h[m].dropna()
                lo, hi = bootstrap_ci(x)
                out.append(dict(epochs=ep, lam=lam, subset=name, metric=lab, n_targets=len(x), mean=x.mean(),
                                ci_lo=lo, ci_hi=hi, improved=int((x < 0).sum()), worse_1pct=int((x > 1).sum())))
    return per, pd.DataFrame(out)


def plot_interp(sm, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.tta.compare import COL, INK, INK2, LAB, SURFACE, style
    metrics = ["force MAE", "energy MAE (offset-free)", "stress MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax, lab in zip(axes, metrics):
        for ep, ls in [(6, "-"), (60, "--")]:
            for sub, a in [("all targets", 1.0), ("other metal", 0.55), ("+U metal", 0.3)]:
                g = sm[(sm.metric == lab) & (sm.epochs == ep) & (sm.subset == sub)].sort_values("lam")
                ax.plot(g.lam, g["mean"], ls, marker="o", ms=3.5, color=COL[method], alpha=a,
                        label=f"{sub}, {ep} epochs")
                if sub == "all targets":
                    ax.fill_between(g.lam, g.ci_lo, g.ci_hi, color=COL[method], alpha=0.08 if ep == 60 else 0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(lab, fontsize=9.5, color=INK)
        ax.set_xlabel("λ  (0 = MACE-OMAT-0, 1 = fine-tuned)", fontsize=8, color=INK2)
        ax.set_ylabel("change vs MACE-OMAT-0 (%)", fontsize=8, color=INK2)
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(f"Weight interpolation, {LAB[method]}: mean change of target errors (band: 95% CI, all targets)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / f"interp_{method}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cohorts(sm, method, lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.tta.compare import COL, INK, INK2, LAB, SURFACE, style
    metrics = ["force MAE", "energy MAE (offset-free)", "stress MAE", "energy MAE (raw)"]
    subsets = list(dict.fromkeys(sm.subset))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.42 * len(subsets) + 1.6),
                             facecolor=SURFACE, sharey=True)
    for ax, lab in zip(axes, metrics):
        for k, (ep, mk) in enumerate([(6, "o"), (60, "s")]):
            h = sm[(sm.metric == lab) & (sm.epochs == ep)].set_index("subset")
            for i, sub in enumerate(subsets):
                if sub not in h.index:
                    continue
                r = h.loc[sub]
                y = i + (k - 0.5) * 0.3
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=COL[method], lw=1.6, alpha=0.5 + 0.5 * k)
                ax.scatter([r["mean"]], [y], marker=mk, s=26, color=COL[method], alpha=0.5 + 0.5 * k, zorder=3,
                           label=f"after {ep} epochs" if i == 0 else None)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(range(len(subsets)))
        ax.set_yticklabels([f"{s} ({int(sm[(sm.subset == s)].n_targets.max())})" for s in subsets], fontsize=7.5)
        ax.set_title(lab, fontsize=9.5, color=INK)
        ax.set_xlabel("change vs base model (%), mean, 95% CI", fontsize=7.5, color=INK2)
        style(ax)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{LAB[method]}, 20 structures, lr {lr:g}: change of target errors on every target "
                 "(negative = better; in brackets: targets)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / f"cohorts_{method}.png", dpi=150, facecolor=SURFACE); plt.close(fig)


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
    parser.add_argument("--cohorts", metavar="METHOD", help="one selection on all its targets, by round and metal")
    parser.add_argument("--interp", metavar="METHOD", help="weight interpolation of this selection's runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.interp:
        per, sm = interp(args.interp, args.lr[0])
        per.to_csv(OUT / f"interp_{args.interp}.csv", index=False)
        sm.to_csv(OUT / f"interp_{args.interp}_summary.csv", index=False)
        plot_interp(sm, args.interp)
        sm["cell"] = [f"{m:+.1f} [{lo:+.1f},{hi:+.1f}] {k}/{n} w{w}" for m, lo, hi, k, n, w in
                      zip(sm["mean"], sm.ci_lo, sm.ci_hi, sm.improved, sm.n_targets, sm.worse_1pct)]
        pd.set_option("display.width", 250)
        for sub in ["all targets", "other metal", "+U metal"]:
            g = sm[(sm.subset == sub) & sm.metric.isin(["force MAE", "energy MAE (offset-free)", "stress MAE"])]
            print(f"\n== {sub}: mean change % [95% CI] improved/targets, w = targets worse by > 1%")
            print(g.pivot_table(index=["epochs", "lam"], columns="metric", values="cell", aggfunc="first").to_string())
        return
    if args.cohorts:
        per, sm = cohorts(args.cohorts, args.lr[0])
        per.to_csv(OUT / f"cohorts_{args.cohorts}.csv", index=False)
        sm.to_csv(OUT / f"cohorts_{args.cohorts}_summary.csv", index=False)
        plot_cohorts(sm, args.cohorts, args.lr[0])
        sm["cell"] = [f"{m:+.1f} [{lo:+.1f},{hi:+.1f}] {k}/{n}" for m, lo, hi, k, n in
                      zip(sm["mean"], sm.ci_lo, sm.ci_hi, sm.improved, sm.n_targets)]
        pd.set_option("display.width", 250)
        for ep in [6, 60]:
            print(f"\n== {args.cohorts}, lr {args.lr[0]:g}, after {ep} epochs: mean change % [95% CI], improved/targets")
            g = sm[sm.epochs == ep]
            print(g.pivot(index="subset", columns="metric", values="cell")
                  .reindex(columns=[l for _, l in METRICS]).reindex(list(dict.fromkeys(g.subset))).to_string())
        return
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
