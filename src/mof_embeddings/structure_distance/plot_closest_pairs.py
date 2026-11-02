#!/usr/bin/env python
"""Examples of the closest MOF-off <-> OMat24 pairs under the structure FGW distance
(alpha 0.5, sigma 3 A). Each row: the MOF, the OMat24 structure (unit cells, CPK
colours), and where the matching sends each MOF element: the share of each MOF
element's atoms transported to each OMat24 element. One pair is picked per theme
(closest overall, closest per OMat24 subset of interest, the organic crystal)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.io import read
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor

from mof_embeddings.config import DATA
from mof_embeddings.structure_distance.compute_pairwise_fgw import setup

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
FLOW = LinearSegmentedColormap.from_list("flow", [SURFACE, "#86b6ef", "#2a78d6", "#0d366b"])


def draw(ax, atoms, title):
    radii = [0.32 + 0.1 * (z > 10) for z in atoms.numbers]
    plot_atoms(atoms, ax, radii=radii, rotation="15x,25y,0z", show_unit_cell=2)
    ax.set_aspect("equal", adjustable="datalim")                 # keep the panel box; fit the drawing inside it
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5, color=INK, pad=4)
    els = list(dict.fromkeys(sorted(set(atoms.get_chemical_symbols()), key=lambda e: atomic_numbers[e])))
    ax.legend(handles=[Patch(facecolor=jmol_colors[atomic_numbers[e]], edgecolor=MUTED, linewidth=0.6, label=e)
                       for e in els], loc="lower center",
              bbox_to_anchor=(0.5, -0.16), ncol=min(len(els), 6), frameon=False, fontsize=7.5, handlelength=0.9,
              columnspacing=0.8, labelcolor=INK2)


def element_flow(plan, sym1, sym2):
    e1 = sorted(set(sym1), key=lambda e: atomic_numbers[e]); e2 = sorted(set(sym2), key=lambda e: atomic_numbers[e])
    F = np.zeros((len(e1), len(e2)))
    for a, s in enumerate(sym1):
        for b, t in enumerate(sym2):
            F[e1.index(s), e2.index(t)] += plan[a, b]
    return F / F.sum(1, keepdims=True), e1, e2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(f"{DATA}/structure_fgw"))
    parser.add_argument("--rows", type=int, nargs="*", help="only these example rows (0-based), in this order")
    parser.add_argument("--out", type=Path, help="figure path (default: analysis/closest_pairs.png)")
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    meta = pd.read_csv(args.root / "sample/structures.csv")
    from mof_embeddings.structure_distance.analyze_pairwise_fgw import FAMILY
    meta["group"] = meta.subdataset.map(FAMILY)
    atoms = read(args.root / "sample/structures.extxyz", ":")
    D = np.load(args.root / "pairwise/D_alpha0.5.npy")
    D0 = np.load(args.root / "pairwise/D_alpha0.0.npy")
    mof = np.flatnonzero(meta.dataset == "MOF-off")

    def best_pair(omat_mask):
        om = np.flatnonzero(omat_mask)
        sub = D[np.ix_(mof, om)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        return mof[i], om[j]

    isom = (meta.dataset == "OMat24").to_numpy()
    themes = [("Closest pair overall", isom),
              ("Closest from AIMD 1000 K", isom & (meta.group == "AIMD 1000 K").to_numpy() & (meta.formula != "C3H5N5O").to_numpy()),
              ("The organic crystal C₃H₅N₅O", (meta.formula == "C3H5N5O").to_numpy()),
              ("Closest from rattled 1000", isom & (meta.group == "rattled 1000").to_numpy()),
              ("Closest from AIMD 3000 K", isom & (meta.group == "AIMD 3000 K").to_numpy()),
              ("Closest carbonate (BaCO₃)", (meta.formula == "CBaO3").to_numpy())]
    pairs = [(t, *best_pair(m)) for t, m in themes]
    if args.rows:
        pairs = [pairs[i] for i in args.rows]

    feats, K, constants = setup(args.root / "sample", 3.0)
    from mof_embeddings.structure_distance.structure_fgw import fgw_distance
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 3.55 * len(pairs)), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.25, hspace=0.62))
    rows = []
    for r, (theme, i, j) in enumerate(pairs):
        value, plan = fgw_distance(feats[i], K[i], feats[j], K[j], 0.5, constants["m0"], return_plan=True)
        F, e1, e2 = element_flow(plan, atoms[i].get_chemical_symbols(), atoms[j].get_chemical_symbols())
        rank = int((D[i, isom] < D[i, j]).sum()) + 1
        draw(axes[r, 0], atoms[i], f"MOF-off · {meta.formula[i]} ({meta.natoms[i]} atoms)")
        draw(axes[r, 1], atoms[j], f"OMat24 {meta.group[j]} · {meta.formula[j]} ({meta.natoms[j]} atoms)")
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
        ax.set_title(f"where each MOF element's atoms go", fontsize=9.5, color=INK, pad=4)
        axes[r, 0].text(-0.02, 1.2, f"{theme}  ·  FGW distance {D[i, j]:.2f} (features only {D0[i, j]:.2f})  ·  "
                        f"#{rank} closest of 1,000 OMat24 structures to this MOF", transform=axes[r, 0].transAxes,
                        fontsize=10.5, color=INK, fontweight="semibold")
        rows.append(dict(theme=theme, mof=meta.formula[i], omat=meta.formula[j], subset=meta.group[j],
                         fgw=round(float(D[i, j]), 3), features_only=round(float(D0[i, j]), 3), rank_for_mof=rank,
                         recomputed=round(value, 3)))
    fig.subplots_adjust(top=0.94)
    if not args.no_title:
        fig.suptitle("Closest MOF-off ↔ OMat24 pairs under structure FGW (UMA+MH1 features, σ = 3 Å, α = 0.5)",
                 fontsize=13, color=INK, y=0.975)
    fig.savefig(args.out or args.root / "analysis/closest_pairs.png", dpi=130, facecolor=SURFACE, bbox_inches="tight")
    if not args.rows:
        pd.DataFrame(rows).to_csv(args.root / "analysis/closest_pairs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
