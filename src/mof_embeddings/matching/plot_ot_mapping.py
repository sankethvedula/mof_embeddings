"""Draw the OT atom matches in real space. One 3D panel per material and model:
supercell atoms (filled, coloured and labelled by element), and for every atom a
line to the nearest copy of the small-cell atom OT matched it to, after the
small cell is aligned onto the supercell. Short grey line = the true partner
(length = rattling); dashed grey = a symmetry-equivalent copy of the true site
(only where the parent space group was recovered); red = a different site."""
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

ELEMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4
INK, INK2, MUTED, SURFACE, EDGE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#c3c2b7"
GOOD, BAD = "#52514e", "#d03b3b"                                   # true partner / wrong site


def min_image(lattice, frac_from, frac_to):
    d = frac_to - frac_from
    d -= np.round(d)
    return d @ lattice


def cell_edges(ax, lattice):
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ lattice
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                ax.plot(*corners[[a, b]].T, color=EDGE, lw=0.6, zorder=0)


def panel(ax, g, model, scores):
    lattice = np.array(g["lattice"])
    lf, af = np.array(g["large_frac"]), np.array(g["aligned_frac"])   # aligned small cell, supercell basis
    large = lf @ lattice
    truth, match = np.array(g["truth"]), np.array(g[f"ot_{model}"])
    orbit = np.array(g["orbit"]) if g["orbit"] is not None else None
    species = np.array(g["large_species"])
    elements = list(dict.fromkeys(g["small_species"]))
    cell_edges(ax, lattice)
    for j in range(len(large)):
        copies = np.flatnonzero(truth == match[j])
        vecs = np.stack([min_image(lattice, lf[j], af[c]) for c in copies])
        v = vecs[np.argmin(np.linalg.norm(vecs, axis=1))]
        ok = match[j] == truth[j]
        equivalent = not ok and orbit is not None and orbit[match[j]] == orbit[truth[j]]
        ax.plot(*np.stack([large[j], large[j] + v]).T, color=GOOD if ok or equivalent else BAD,
                lw=1.0 if ok or equivalent else 1.6, ls="--" if equivalent else "-", alpha=0.9, zorder=2)
    for e, c in zip(elements, ELEMENT_COLORS):
        m = species == e
        ax.scatter(*large[m].T, s=46, color=c, edgecolor=SURFACE, linewidth=1.2, depthshade=False, zorder=3)
        for p in large[m]:
            ax.text(*p, f" {e}", fontsize=6.5, color=INK2, zorder=4)
    ax.set_axis_off()
    ax.set_box_aspect(np.ptp(np.vstack([large, (np.array([[0, 0, 0], [1, 1, 1]]) @ lattice)]), axis=0))
    ax.view_init(elev=18, azim=-60)
    sym = f"spg {g['spg']}" if g["orbit"] is not None else "symmetry not recovered"
    ax.set_title(f"{g['formula']} ({len(g['small_species'])}→{len(g['large_species'])} atoms, {sym}) · {model}\n"
                 f"{scores['exact']:.0%} exact site · median line {scores['error']:.1f} Å",
                 fontsize=9, color=INK, pad=-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=Path(f"{DATA}/similar_structures/ot_mapping"))
    parser.add_argument("--formulas", nargs="*", help="only these materials, in this order")
    parser.add_argument("--per-row", type=int, default=2, help="materials per row (each takes 2 panels)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    geom = json.loads((args.map_dir / "geometry.json").read_text())
    if args.formulas:
        by_formula = {g["formula"]: (m, g) for m, g in geom.items()}
        geom = dict(by_formula[f] for f in args.formulas)
    df = pd.read_csv(args.map_dir / "atom_matches.csv")
    models = ["UMA", "MH1"]

    per = args.per_row
    nrows = (len(geom) + per - 1) // per
    fig = plt.figure(figsize=(7.5 * per, 3.7 * nrows + (0.3 if args.no_title else 0) + (0.85 if per == 1 else 0)), facecolor=SURFACE)
    for r, (mat, g) in enumerate(geom.items()):
        for c, model in enumerate(models):
            ax = fig.add_subplot(nrows, 2 * per, 2 * r + c + 1, projection="3d", facecolor=SURFACE)
            sub = df[(df.material == mat) & (df.model == model)]
            panel(ax, g, model, dict(exact=sub.exact.mean(), error=sub.error_A.median()))
    handles = [Line2D([], [], color=GOOD, lw=1.2, label="matched to its true partner (line = rattling)"),
               Line2D([], [], color=GOOD, lw=1.2, ls="--", label="matched to a symmetry-equivalent copy"),
               Line2D([], [], color=BAD, lw=1.8, label="matched to a different site (line = how far off)")]
    legend_cols = 3 if per >= 2 else 1                          # narrow figures stack the legend
    fig.legend(handles=handles, loc="upper center", ncol=legend_cols, frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.995 if args.no_title else 0.975))
    if not args.no_title:
        fig.suptitle("OT matching of atom embeddings, small cell → 2× supercell, drawn in the supercell",
                 fontsize=13, color=INK, y=0.995)
    header_in = (0.45 if args.no_title else 0.9) + (0.85 if legend_cols == 1 else 0)   # legend (+ title) band, inches
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - header_in / fig.get_figheight(), bottom=0.01, wspace=0.02, hspace=0.12)
    out = args.out or args.map_dir / "ot_mapping_3d.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
