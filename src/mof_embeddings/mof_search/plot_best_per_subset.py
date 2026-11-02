#!/usr/bin/env python
"""Grids of the best OMat24 match per (MOF query, subset family): left any
structure, right structures containing both C and H. Cell colour and number =
FGW distance relative to that query's best match anywhere; cell text = formula."""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mof_embeddings.mof_search.common import OUT

ORDER = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax", "rattled 300", "rattled 500", "rattled 1000"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("gap", ["#0d366b", "#2a78d6", "#9ec5f4", SURFACE])


def pretty(formula):
    return re.sub(r"(\d+)", lambda m: m.group(1).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")), formula)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "best_per_subset.png")
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    q = pd.read_csv(OUT / "queries.csv")
    label = {r["query"]: f"{r['query']} {r.metal} · {pretty(r.formula)}" for _, r in q.iterrows()}
    tables = [(pd.read_csv(OUT / "best_per_subset.csv"), "Best match in each subset"),
              (pd.read_csv(OUT / "best_organic_per_subset.csv"), "Best match containing C and H")]
    queries = list(q["query"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), facecolor=SURFACE, gridspec_kw=dict(wspace=0.05))
    for k, (ax, (df, title)) in enumerate(zip(axes, tables)):
        R = df.pivot(index="query", columns="family", values="vs_overall_best").reindex(index=queries, columns=ORDER)
        Fm = df.pivot(index="query", columns="family", values="formula").reindex(index=queries, columns=ORDER)
        ax.imshow(R.values, cmap=CMAP, vmin=1, vmax=2.1, aspect="auto")
        for i in range(len(queries)):
            for j in range(len(ORDER)):
                v = R.values[i, j]
                c = SURFACE if v < 1.45 else INK
                ax.text(j, i - 0.14, f"{v:.2f}×", ha="center", va="center", fontsize=9.5, color=c, fontweight="semibold")
                f = pretty(Fm.values[i, j])
                ax.text(j, i + 0.2, f if len(f) <= 15 else f[:14] + "…", ha="center", va="center", fontsize=7, color=c)
        ax.set_xticks(range(len(ORDER)), ORDER, fontsize=9, color=INK2, rotation=20, ha="right")
        ax.set_yticks(range(len(queries)), [label[x] for x in queries] if k == 0 else [], fontsize=8.5, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=11, color=INK)
    if not args.no_title:
        fig.suptitle("Best OMat24 match per subset for each MOF query · FGW distance relative to the query's best match anywhere (1.00× = best)",
                     fontsize=12.5, color=INK, y=0.99)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
