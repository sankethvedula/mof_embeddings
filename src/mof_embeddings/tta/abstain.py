#!/usr/bin/env python
"""Adapt or keep the foundation model? Target-blind rules evaluated on the finished runs.

Each rule decides before fine-tuning, from information available without the target's labels,
whether to fine-tune (and use the fine-tuned model) or keep MACE-OMAT-0 (change 0%):
  always            fine-tune every target
  no +U metal       skip targets whose metal OMat24 labels with PBE+U (tta.common.U_METALS)
  feature < tau     fine-tune only if the training set's mean feature term (FGW at alpha 0.25, from
                    the candidate-pool ranking tta.select_alpha) is below tau
  both              no +U metal and feature < tau
tau is chosen by leave-one-target-out: for each target, the tau that minimises the mean force change
on the other targets (over the observed feature values) decides that target, so the reported result
is not fitted to the target it is scored on.

Input: data/tta/effect/cohorts_<method>.csv (tta.effect --cohorts). Reported per rule and epoch:
mean change of force MAE, offset-free energy MAE and stress MAE over all targets, number of targets
fine-tuned, and targets made worse (force change > +1%).

  $PY -m mof_embeddings.tta.abstain [--method fgwmid]
  -> data/tta/effect/abstain_<method>.csv
"""
import argparse

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA
from mof_embeddings.tta.common import N_TRAIN, U_METALS

OUT = DATA / "tta" / "effect"
METRICS = [("target_force_mae", "force"), ("target_energy_mae_offset_free_meV", "energy offset-free"),
           ("target_stress_mae_meV_A3", "stress")]
COL = {"fgwmid": "a0_25", "fgwlow": "a0_1", "fgw": "a0_5"}


def set_feature(target, method):
    """Mean feature term of the training structures (pool ranking at the selection's alpha)."""
    tag = COL[method]
    p = pd.read_csv(DATA / "tta" / "alpha_rank" / f"{target}.csv").sort_values(f"fgw_{tag}")
    return p[f"feature_{tag}"].head(N_TRAIN).mean()


def loo_threshold(x, y):
    """Leave-one-out tau: for target i, the threshold on the others minimising their mean change."""
    out = np.empty(len(x))
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        cands = np.r_[np.sort(x[m]), np.inf]
        means = [np.where(x[m] < c, y[m], 0).mean() for c in cands]
        out[i] = cands[int(np.argmin(means))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="fgwmid")
    args = parser.parse_args()
    d = pd.read_csv(OUT / f"cohorts_{args.method}.csv")
    d["feature"] = [set_feature(t, args.method) for t in d.target]
    d["u_metal"] = d.metal.isin(U_METALS)
    rows = []
    for ep, g in d.groupby("epochs"):
        g = g.dropna(subset=[m for m, _ in METRICS]).reset_index(drop=True)
        tau = loo_threshold(g.feature.values, g.target_force_mae.values)
        rules = {"always": np.ones(len(g), bool), "no +U metal": ~g.u_metal.values,
                 "feature < tau (LOO)": g.feature.values < tau,
                 "both": ~g.u_metal.values & (g.feature.values < tau)}
        for name, adapt in rules.items():
            r = dict(epochs=ep, rule=name, targets=len(g), fine_tuned=int(adapt.sum()),
                     tau_median=float(np.median(tau)) if "tau" in name or name == "both" else np.nan,
                     worse_force=int(((g.target_force_mae.values > 1) & adapt).sum()))
            for m, lab in METRICS:
                r[f"{lab} mean %"] = float(np.where(adapt, g[m].values, 0).mean())
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"abstain_{args.method}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    print("\nfeature term vs force change (Spearman):",
          {ep: round(g[["feature", "target_force_mae"]].corr("spearman").iloc[0, 1], 2) for ep, g in d.groupby("epochs")})


if __name__ == "__main__":
    main()
