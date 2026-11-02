#!/usr/bin/env python
"""Why do MOF queries match OMat24 AIMD frames and not rattled frames?

1. Best candidates per subset: for every query and OMat24 subset family, the 50
   best prefilter candidates are rescored with exact FGW, split into its two
   halves at the optimal plan: feature cost <M, T> and geometry cost (GW term).
2. Same chemistry: scanned sr_10m structures containing both C and H (up to 400 per
   family) are scored against every query with the two prefilter halves (feature
   OT, geometry stand-in), and their C-H bond lengths (H to nearest C under 1.6 A)
   are measured against the equilibrium ~1.09 A.
Scores are divided by each query's median over all scored structures, so queries
of very different size can be pooled.
"""
import json

import numpy as np
import pandas as pd

from mof_embeddings.config import ARCHIVE, DATA, use_local_packages
use_local_packages()
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect
from scipy.spatial.distance import cdist

from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.mof_search.prefilter import eccentricity_quantiles
from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"
FAMILY = {"aimd-from-PBE-1000-npt": "AIMD 1000 K", "aimd-from-PBE-1000-nvt": "AIMD 1000 K",
          "aimd-from-PBE-3000-npt": "AIMD 3000 K", "aimd-from-PBE-3000-nvt": "AIMD 3000 K",
          "rattled-1000": "rattled 1000", "rattled-1000-subsampled": "rattled 1000",
          "rattled-500": "rattled 500", "rattled-500-subsampled": "rattled 500",
          "rattled-300": "rattled 300", "rattled-300-subsampled": "rattled 300", "rattled-relax": "rattled relax"}
ORDER = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax", "rattled 300", "rattled 500", "rattled 1000"]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
FAM_COLOR = {"AIMD 1000 K": "#2a78d6", "AIMD 3000 K": "#6250d6", "rattled relax": "#1baf7a",
             "rattled 300": "#eda100", "rattled 500": "#eb6834", "rattled 1000": "#e34948"}


def ch_bonds(at):
    d = at.get_all_distances(mic=True)
    H, C = np.flatnonzero(at.numbers == 1), np.flatnonzero(at.numbers == 6)
    if not len(H) or not len(C):
        return np.array([])
    near = d[np.ix_(H, C)].min(1)
    return near[near < 1.6]


def main():
    names = json.load(open(L + ".json"))["names"]
    family_of = np.array([FAMILY[n] for n in names])[np.load(L + ".npy")]
    const = json.loads((OUT / "constants.json").read_text())
    m0, k0, sigma, alpha = const["m0"], const["k0"], const["sigma"], const["alpha"]
    parts = [np.load(p) for p in sorted((OUT / "prefilter_parts").glob("part_*.npz"))]
    fr = np.concatenate([p["frame"] for p in parts]); sc = np.concatenate([p["score"] for p in parts])
    sc[np.concatenate([p["min_dist"] for p in parts]) < 0.8] = np.inf
    fam = family_of[fr]
    print(f"{len(parts)} prefilter parts, {len(fr):,} frames; scanned per family:",
          pd.Series(fam).value_counts().to_dict(), flush=True)

    mu, sd = uma_stats()
    q, qf, qa = query_features(mu, sd)
    qK, q_ecc = [], []
    for a in qa:
        C, _ = periodic_kernel_fast(np.array(a.cell), a.positions, sigma)
        qK.append(len(C) * C / k0); q_ecc.append(eccentricity_quantiles(qK[-1]))
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    cache = {}

    def load(f):
        if f not in cache:
            at = db.get_atoms(id=int(f) + 1)
            x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
            C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, sigma)
            cache[f] = (at, x, len(C) * C / k0)
        return cache[f]

    # 1. best candidates per family: exact FGW and its two halves
    rows = []
    for j in range(len(qf)):
        p, = [ot.unif(len(qf[j]))]
        for family in ORDER:
            idx = np.flatnonzero(fam == family)
            if not len(idx):
                continue
            for t in idx[np.argsort(sc[idx, j])[:50]]:
                at, x, K = load(int(fr[t]))
                M = cdist(qf[j], x, "sqeuclidean") / m0
                pq = ot.unif(len(x))
                best = None
                for G0 in [ot.emd(p, pq, M), None]:
                    T, log = ot.gromov.fused_gromov_wasserstein(M, qK[j], K, p, pq, loss_fun="square_loss",
                                                                alpha=alpha, G0=G0, log=True, max_iter=200)
                    if best is None or log["fgw_dist"] < best[0]:
                        best = (log["fgw_dist"], T)
                T = best[1]
                gw = ot.gromov.gwloss(*ot.gromov.init_matrix(qK[j], K, p, pq, "square_loss"), T)
                rows.append(dict(query=q["query"][j], family=family, frame=int(fr[t]), fgw=best[0],
                                 feature_term=float(np.sum(M * T)), geometry_term=float(gw)))
        print(f"  {q['query'][j]} best-per-family done", flush=True)
    best = pd.DataFrame(rows)
    for col in ["fgw", "feature_term", "geometry_term"]:
        best[col + "_rel"] = best[col] / best.groupby("query")[col].transform("median")
    best.to_csv(OUT / "subset_terms_best50.csv", index=False)

    # 2. same chemistry: C+H structures per family
    comp = pd.read_parquet(f"{DATA}/overlap/omat_sr10m_compositions.parquet", columns=["has_C", "has_H"])
    organic = np.flatnonzero((comp.has_C & comp.has_H).to_numpy())
    scanned = np.zeros(len(family_of), bool); scanned[fr] = True
    organic = organic[scanned[organic]]
    rng = np.random.default_rng(0)
    rows, bonds = [], []
    for family in ORDER:
        pick = organic[family_of[organic] == family]
        pick = rng.choice(pick, min(400, len(pick)), replace=False) if len(pick) else pick
        for f in pick:
            at, x, K = load(int(f))
            ecc = eccentricity_quantiles(K)
            b = ch_bonds(at)
            bonds += [dict(family=family, length=v) for v in b]
            for j in range(len(qf)):
                rows.append(dict(query=q["query"][j], family=family, frame=int(f),
                                 feature_ot=ot.emd2(ot.unif(len(qf[j])), ot.unif(len(x)), cdist(qf[j], x, "sqeuclidean") / m0),
                                 geometry=float(np.mean((q_ecc[j] - ecc) ** 2))))
    org = pd.DataFrame(rows)
    for col in ["feature_ot", "geometry"]:
        org[col + "_rel"] = org[col] / org.groupby("query")[col].transform("median")
    org.to_csv(OUT / "subset_terms_organic.csv", index=False)
    bonds = pd.DataFrame(bonds)
    bonds.to_csv(OUT / "subset_ch_bonds.csv", index=False)

    summary = dict(
        best50=best.groupby("family")[["fgw_rel", "feature_term_rel", "geometry_term_rel"]].median().reindex(ORDER).round(3).to_dict("index"),
        organic=org.groupby("family")[["feature_ot_rel", "geometry_rel"]].median().reindex(ORDER).round(3).to_dict("index"),
        organic_structures=org.groupby("family").frame.nunique().reindex(ORDER).to_dict(),
        ch_bond=bonds.groupby("family").length.agg(["count", "median", "std",
                                                   lambda s: float(np.mean(np.abs(s - 1.09) > 0.15))]).reindex(ORDER)
                    .rename(columns={"<lambda_0>": "share_off_by_0.15A"}).round(3).to_dict("index"))
    (OUT / "subset_terms_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))

    # Figure: three rows of small multiples
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), facecolor=SURFACE, gridspec_kw=dict(wspace=0.32))
    panels = [(best, "fgw_rel", "Best 50 per subset: FGW distance"),
              (best, "feature_term_rel", "…its feature (embedding) part"),
              (best, "geometry_term_rel", "…its geometry part")]
    for ax, (df, col, title) in zip(axes[:3], panels):
        fams = [f for f in ORDER if f in set(df.family)]
        data = [df[df.family == f][col].to_numpy() for f in fams]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for b, f in zip(bp["boxes"], fams):
            b.set(facecolor=FAM_COLOR[f], edgecolor=SURFACE, alpha=0.85)
        ax.set_yticks(range(1, len(fams) + 1), fams, fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.set_xlabel("relative to the query's median (lower = closer)", fontsize=8.5, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK)
        ax.axvline(1, color=GRID, lw=1, zorder=0)
    ax = axes[3]
    for f in ORDER:
        v = bonds[bonds.family == f].length
        if len(v) > 20:
            ax.hist(v, bins=np.linspace(0.6, 1.6, 51), density=True, histtype="step", lw=1.8, color=FAM_COLOR[f], label=f"{f} ({len(v)})")
    ax.axvline(1.09, color=MUTED, ls="--", lw=1)
    ax.text(1.06, ax.get_ylim()[1] * 0.55, "equilibrium\nC–H 1.09 Å", fontsize=8, color=MUTED, ha="right")
    ax.set_title("C–H bond lengths in organic OMat24 frames", fontsize=10, color=INK)
    ax.set_xlabel("H to nearest C (Å)", fontsize=8.5, color=MUTED); ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    for a in axes:
        a.set_facecolor(SURFACE); a.tick_params(colors=INK2, labelsize=8)
        for s in a.spines.values(): s.set_color(GRID)
    fig.suptitle("Why MOF queries match AIMD frames: distance and its parts by OMat24 subset, and C–H bond distortion",
                 fontsize=12, color=INK, y=1.02)
    fig.savefig(OUT / "subset_terms.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print("saved", OUT / "subset_terms.png")


if __name__ == "__main__":
    main()
