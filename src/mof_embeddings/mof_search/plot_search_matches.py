#!/usr/bin/env python
"""Best OMat24 sr_10m match for each MOF query after the FGW rerank. Matches are
de-duplicated by OMat24 formula + subset (consecutive frames of one AIMD run
count once). Each row: the MOF, its best match, and where each MOF element's
atoms are sent by the FGW plan. Also writes the distinct top-5 table."""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect

from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.plot_closest_pairs import FLOW, INK, INK2, MUTED, SURFACE, draw, element_flow
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel_fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="*", help="only these queries, in this order (e.g. Q3 Q6 Q2)")
    parser.add_argument("--out", default=str(OUT / "best_matches.png"))
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    const = json.loads((OUT / "constants.json").read_text())
    top = pd.read_csv(OUT / "top20_per_query.csv")
    distinct = top.drop_duplicates(["query", "formula", "subset"]).groupby("query", sort=False).head(5)
    distinct.to_csv(OUT / "top5_distinct_per_query.csv", index=False)
    best = distinct.groupby("query", sort=False).head(1).reset_index(drop=True)
    best = best.iloc[np.argsort([int(q[1:]) for q in best["query"]])].reset_index(drop=True)
    if args.queries:
        best = best.set_index("query").loc[args.queries].reset_index()

    mu, sd = uma_stats()
    q, qf, qa = query_features(mu, sd)
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)

    def kernel(at):
        C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, const["sigma"])
        return len(C) * C / const["k0"]

    fig, axes = plt.subplots(len(best), 3, figsize=(12, 3.55 * len(best)), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.25, hspace=0.85))
    for r, row in best.iterrows():
        j = int(row["query"][1:]) - 1
        f = int(row.frame)
        at = db.get_atoms(id=f + 1)
        x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
        value, plan = fgw_distance(qf[j], kernel(qa[j]), x, kernel(at), const["alpha"], const["m0"], return_plan=True)
        F, e1, e2 = element_flow(plan, qa[j].get_chemical_symbols(), at.get_chemical_symbols())
        short = row.subset.replace("aimd-from-PBE-", "AIMD ").replace("-npt", " K NPT").replace("-nvt", " K NVT")
        draw(axes[r, 0], qa[j], f"MOF · {q.formula[j]} ({len(qa[j])})")
        draw(axes[r, 1], at, f"OMat24 {short} · {row.formula} ({len(at)})")
        ax = axes[r, 2]
        ax.imshow(F, cmap=FLOW, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(e2)), e2, fontsize=8.5, color=INK2); ax.set_yticks(range(len(e1)), e1, fontsize=8.5, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        for a in range(len(e1)):
            for b in range(len(e2)):
                if F[a, b] >= 0.1:
                    ax.text(b, a, f"{F[a, b]:.0%}", ha="center", va="center", fontsize=7.5,
                            color=SURFACE if F[a, b] > 0.55 else INK)
        ax.set_xlabel("OMat24 element", fontsize=8.5, color=MUTED); ax.set_ylabel("MOF element", fontsize=8.5, color=MUTED)
        ax.set_title("where each MOF element's atoms go", fontsize=9.5, color=INK, pad=4)
        axes[r, 0].text(-0.02, 1.32, f"{row['query']} ({q.metal[j]})  ·  best of 10M OMat24 structures  ·  FGW {value:.3f}",
                        transform=axes[r, 0].transAxes, fontsize=10.5, color=INK, fontweight="semibold")
    fig.subplots_adjust(top=0.96)
    if not args.no_title:
        fig.suptitle("Best OMat24 (sr_10m, 10M structures) match for each MOF query · structure FGW, UMA features, σ = 3 Å, α = 0.5",
                     fontsize=13, color=INK, y=0.99)
    fig.savefig(args.out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    print(distinct[["query", "formula_query", "fgw_rank", "formula", "subset", "natoms", "fgw", "prefilter_rank"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
