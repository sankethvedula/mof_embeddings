#!/usr/bin/env python
"""Which differences between atom-matching methods hold up on a held-out set?

Each method is scored per material (exact site, same element, median distance of
the matched site, fraction of neighbour bonds kept, bond-length distortion), then
compared in pairs across materials: bootstrap 95% CI of the mean per-material
difference (10,000 resamples of materials) and a Wilcoxon signed-rank test, with
Holm correction over every comparison. A difference "holds up" when the CI
excludes zero and the corrected p < 0.05.

Inputs: ot_mapping/, fgw_mapping_periodic/ and gw_embedding/ under --data-dir.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mof_embeddings.config import DATA

METRICS = {"exact": "exact site", "same_element": "same element", "error_A": "median distance to matched site (Å)",
           "neighbours_kept": "bonds kept", "distortion_A": "bond-length change (Å)"}
HIGHER_IS_BETTER = {"exact": True, "same_element": True, "error_A": False, "neighbours_kept": True, "distortion_A": False}


def per_material(data_dir):
    fgw_atoms = pd.read_csv(data_dir / "fgw_mapping_periodic/atom_matches.csv")
    fgw_bonds = pd.read_csv(data_dir / "fgw_mapping_periodic/neighbour_scores.csv")
    gw_atoms = pd.read_csv(data_dir / "gw_embedding/atom_matches.csv")
    gw_bonds = pd.read_csv(data_dir / "gw_embedding/neighbour_scores.csv")
    frames = pd.read_csv(data_dir / "frames.csv").pivot(index="material", columns="family", values="natoms")
    ratio = (frames.r300 / frames.relax).round().astype(int)

    def method_name(model, alpha):
        return f"OT · {model}" if alpha == 0 else f"FGW α={alpha:g} · {model}"

    fgw_atoms["method"] = [method_name(m, a) for m, a in zip(fgw_atoms.model, fgw_atoms.alpha)]
    gw_atoms["method"] = "GW · " + gw_atoms.model.str.replace(">", "→")
    atoms = pd.concat([fgw_atoms, gw_atoms])
    scores = atoms.groupby(["method", "material"]).agg(exact=("exact", "mean"), same_element=("same_element", "mean"),
                                                       error_A=("error_A", "median"))
    fgw_bonds["method"] = [m if pd.isna(a) else method_name(m, a) for m, a in zip(fgw_bonds.model, fgw_bonds.alpha)]
    gw_bonds["method"] = "GW · " + gw_bonds.model.str.replace(">", "→")
    bonds = pd.concat([fgw_bonds, gw_bonds]).set_index(["method", "material"])[["neighbours_kept", "distortion_A"]]
    scores = scores.join(bonds, how="outer").reset_index()
    scores["ratio"] = scores.material.map(ratio)
    return scores


def paired(scores, a, b, metric, rng, n_boot=10_000):
    x = scores[scores.method == a].set_index("material")[metric]
    y = scores[scores.method == b].set_index("material")[metric]
    d = (x - y).dropna().to_numpy()
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(a=a, b=b, metric=metric, n=len(d), mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(),
                ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), p=p,
                a_better=int(np.sum(d > 1e-9) if HIGHER_IS_BETTER[metric] else np.sum(d < -1e-9)),
                b_better=int(np.sum(d < -1e-9) if HIGHER_IS_BETTER[metric] else np.sum(d > 1e-9)))


def holm(p):
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(f"{DATA}/similar_structures_heldout"))
    args = parser.parse_args()
    out = args.data_dir / "comparison"
    out.mkdir(exist_ok=True)
    scores = per_material(args.data_dir)
    scores.to_csv(out / "per_material_scores.csv", index=False)

    order = ["truth", "random_same_element"] + [f"{m} · {fs}" for m in ["OT", "FGW α=0.1", "FGW α=0.25", "GW"]
                                                for fs in ["UMA", "MH1", "UMA+MH1"]] + ["GW · UMA→MH1", "GW · MH1→UMA"]
    table = scores.groupby("method")[list(METRICS)].mean().reindex(order)
    table.to_csv(out / "method_means.csv")

    fs = ["UMA", "MH1", "UMA+MH1"]
    comparisons = []
    for f in fs:                                                       # does geometry help, and how much?
        comparisons += [(f"FGW α=0.25 · {f}", f"OT · {f}"), (f"FGW α=0.1 · {f}", f"FGW α=0.25 · {f}"),
                        (f"GW · {f}", f"OT · {f}"), (f"FGW α=0.25 · {f}", "random_same_element"),
                        (f"OT · {f}", "random_same_element")]
    for m in ["OT", "FGW α=0.25"]:                                     # which features?
        comparisons += [(f"{m} · UMA", f"{m} · MH1"), (f"{m} · UMA+MH1", f"{m} · UMA"), (f"{m} · UMA+MH1", f"{m} · MH1")]
    rng = np.random.default_rng(0)
    rows = []
    for a, b in comparisons:
        for metric in METRICS:
            if b == "random_same_element" and metric not in ("neighbours_kept", "distortion_A"):
                continue
            rows.append(paired(scores, a, b, metric, rng))
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm(tests.p.to_numpy())
    tests["holds_up"] = (tests.p_holm < 0.05) & ((tests.ci_low > 0) | (tests.ci_high < 0))
    tests.to_csv(out / "paired_tests.csv", index=False)

    by_ratio = scores.groupby(["ratio", "method"])[["exact", "neighbours_kept"]].mean().unstack("ratio")
    by_ratio.reindex(order).to_csv(out / "means_by_supercell_ratio.csv")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(f"{scores.material.nunique()} materials\n")
        print(table.round(3).to_string()); print()
        cols = ["a", "b", "metric", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "a_better", "b_better", "p_holm", "holds_up"]
        print(tests[cols].round(3).to_string(index=False)); print()
        print(by_ratio.reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
