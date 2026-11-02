"""Do atoms that are neighbours in the supercell stay neighbours after matching?
Each panel draws the supercell's neighbour bonds (minimum-image distance < 1.3
sigma): grey when the two atoms' matched partners are also neighbours, red when
the matching pulls them apart. Columns: the true correspondence (the ceiling,
since rattling breaks some bonds), plain feature OT and FGW, for UMA and MH1."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mof_embeddings.config import DATA
from mof_embeddings.matching.plot_ot_mapping import ELEMENT_COLORS, INK, INK2, SURFACE, cell_edges, min_image

KEPT, BROKEN = "#898781", "#d03b3b"


def panel(ax, g, key, title):
    lattice = np.array(g["lattice"])
    lf = np.array(g["large_frac"]); large = lf @ lattice
    species = np.array(g["large_species"])
    cell_edges(ax, lattice)
    kept = np.array(g["bonds_kept"][key])
    for (a, b), ok in zip(g["bonds"], kept):
        if a < b:
            end = large[a] + min_image(lattice, lf[a], lf[b])
            ax.plot(*np.stack([large[a], end]).T, color=KEPT if ok else BROKEN,
                    lw=1.0 if ok else 1.7, alpha=0.85 if ok else 0.95, zorder=2)
    for e, c in zip(dict.fromkeys(g["small_species"]), ELEMENT_COLORS):
        m = species == e
        ax.scatter(*large[m].T, s=40, color=c, edgecolor=SURFACE, linewidth=1.1, depthshade=False, zorder=3)
        for p in large[m]:
            ax.text(*p, f" {e}", fontsize=6, color=INK2, zorder=4)
    ax.set_axis_off()
    ax.set_box_aspect(np.ptp(np.vstack([large, np.array([[0, 0, 0], [1, 1, 1]]) @ lattice]), axis=0))
    ax.view_init(elev=18, azim=-60)
    ax.set_title(f"{title}\n{kept.mean():.0%} of bonds kept", fontsize=8.5, color=INK, pad=-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=Path(f"{DATA}/similar_structures/fgw_mapping_periodic"))
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--formulas", nargs="*", help="only these materials, in this order")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    geom = json.loads((args.map_dir / "geometry.json").read_text())
    if args.formulas:
        by_formula = {g["formula"]: (m, g) for m, g in geom.items()}
        geom = dict(by_formula[f] for f in args.formulas)
    columns = [("truth", "True correspondence"), ("UMA_0.0", "UMA · plain OT"),
               (f"UMA_{args.alpha}", f"UMA · FGW α={args.alpha}"), ("MH1_0.0", "MH1 · plain OT"),
               (f"MH1_{args.alpha}", f"MH1 · FGW α={args.alpha}")]

    fig = plt.figure(figsize=(3.0 * len(columns), 3.1 * len(geom) + 0.6), facecolor=SURFACE)
    header = (0.45 if args.no_title else 0.9) / fig.get_figheight()
    for r, g in enumerate(geom.values()):
        for c, (key, title) in enumerate(columns):
            ax = fig.add_subplot(len(geom), len(columns), r * len(columns) + c + 1, projection="3d", facecolor=SURFACE)
            panel(ax, g, key, title)
            if c == 0:
                ax.text2D(-0.02, 0.5, f"{g['formula']}\n{len(g['small_species'])} → {len(g['large_species'])} atoms",
                          transform=ax.transAxes, fontsize=10.5, color=INK, ha="right", va="center")
    handles = [Line2D([], [], color=KEPT, lw=1.2, label="neighbours matched to neighbours"),
               Line2D([], [], color=BROKEN, lw=1.8, label="neighbours matched to atoms far apart")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK2,
               bbox_to_anchor=(0.55, 1.0 if args.no_title else 1 - 0.4 / fig.get_figheight()))
    if not args.no_title:
        fig.suptitle("Do atoms close in space stay close after matching? Supercell neighbour bonds, small cell → 2× supercell",
                 fontsize=13, color=INK, x=0.55, y=1 - 0.04 / len(geom))
    fig.subplots_adjust(left=0.1, right=0.99, top=1 - header, bottom=0.005, wspace=0.0, hspace=0.1)
    out = args.out or args.map_dir / "neighbour_preservation_3d.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
