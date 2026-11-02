#!/usr/bin/env python
"""Pairwise FGW distance matrices grouped by dataset / OMat24 subset.

For a matrix directory (D_alpha*.npy) and its sample (structures.csv), draws:
  left   the full matrix with rows/columns ordered by group (within each group by
         average-linkage clustering), thin separators between groups
  right  the group x group median distance, annotated
Groups come from the sample's `subdataset` column mapped to families.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from mof_embeddings.mof_search.compare_subset_terms import FAMILY

FAMILY = dict(FAMILY, **{"MOF-off R2SCAN": "MOF-off"})
ORDER = ["MOF-off", "AIMD 1000 K", "AIMD 3000 K", "rattled relax", "rattled 300", "rattled 500", "rattled 1000"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#2a78d6", "#9ec5f4", SURFACE])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="D_alpha*.npy")
    parser.add_argument("--sample", type=Path, required=True, help="directory with structures.csv")
    parser.add_argument("--title", default="", help="figure title; empty for none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    D = np.load(args.matrix)
    meta = pd.read_csv(args.sample / "structures.csv")
    group = meta.subdataset.map(FAMILY).to_numpy()
    groups = [g for g in ORDER if g in set(group)]
    keep = np.ones(len(D), bool)
    row_med = np.median(D, 1)                                    # drop only extreme rows (unphysical frames such as
    keep &= np.isfinite(D).all(1) & (row_med < 5 * np.percentile(row_med, 95))   # MoNaO6P, median 72), not large MOFs
    order, bounds = [], [0]
    for g in groups:
        idx = np.flatnonzero((group == g) & keep)
        if len(idx) > 2:
            idx = idx[leaves_list(linkage(squareform(D[np.ix_(idx, idx)], checks=False), "average"))]
        order += list(idx); bounds.append(len(order))
    order = np.array(order)
    B = np.array([[np.median(D[np.ix_(order[bounds[a]:bounds[a + 1]], order[bounds[b]:bounds[b + 1]])]
                   [~np.eye(bounds[a + 1] - bounds[a], dtype=bool)] if a == b else
                   D[np.ix_(order[bounds[a]:bounds[a + 1]], order[bounds[b]:bounds[b + 1]])])
                   for b in range(len(groups))] for a in range(len(groups))])
    pd.DataFrame(B, index=groups, columns=groups).round(3).to_csv(args.out.with_suffix(".group_medians.csv"))

    fig = plt.figure(figsize=(15.5, 8.2), facecolor=SURFACE)
    ax = fig.add_axes([0.11, 0.17, 0.47, 0.74])
    off = D[np.ix_(order, order)][~np.eye(len(order), dtype=bool)]
    lo, hi = np.percentile(off, [1, 99])
    im = ax.imshow(D[np.ix_(order, order)], cmap=SEQ, vmin=lo, vmax=hi, interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color=SURFACE, lw=1.6); ax.axvline(b - 0.5, color=SURFACE, lw=1.6)
    mids = [(bounds[i] + bounds[i + 1]) / 2 - 0.5 for i in range(len(groups))]
    labels = [f"{g} ({bounds[i + 1] - bounds[i]})" for i, g in enumerate(groups)]
    ax.set_yticks(mids, labels, fontsize=9, color=INK2); ax.set_xticks(mids, labels, fontsize=8.5, color=INK2, rotation=35, ha="right")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("All pairs, grouped (darker = more similar)", fontsize=10.5, color=INK)
    cax = fig.add_axes([0.595, 0.17, 0.01, 0.3]); cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=8); cb.set_label("FGW distance", color=INK2, fontsize=8.5)

    ax2 = fig.add_axes([0.70, 0.17, 0.28, 0.74])
    ax2.imshow(B, cmap=SEQ, vmin=lo, vmax=hi)
    for a in range(len(groups)):
        for b in range(len(groups)):
            v = (B[a, b] - lo) / (hi - lo)
            ax2.text(b, a, f"{B[a, b]:.2f}", ha="center", va="center", fontsize=8.5, color=SURFACE if v < 0.45 else INK)
    ax2.set_yticks(range(len(groups)), groups, fontsize=9, color=INK2)
    ax2.set_xticks(range(len(groups)), groups, fontsize=8.5, color=INK2, rotation=35, ha="right")
    ax2.tick_params(length=0)
    for s in ax2.spines.values(): s.set_visible(False)
    ax2.set_title("Median distance between groups", fontsize=10.5, color=INK)
    if args.title:
        fig.suptitle(args.title, fontsize=12.5, color=INK, y=0.985)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE, **({} if args.title else dict(bbox_inches="tight")))
    print("saved", args.out)
    print(pd.DataFrame(B, index=groups, columns=groups).round(2).to_string())


if __name__ == "__main__":
    main()
