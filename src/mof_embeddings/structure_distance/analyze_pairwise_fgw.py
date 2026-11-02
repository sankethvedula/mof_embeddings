#!/usr/bin/env python
"""Patterns in the pairwise structure-FGW matrix of 100 MOF-off + 1000 OMat24 structures.

Reports block statistics (MOF-MOF / OMat-OMat / MOF-OMat), which OMat24 structures
come closest to MOFs, how well 10 nearest neighbours agree on labels (OMat24
simulation family, MOF metal, chemistry) against a shuffled baseline, how much the
distance tracks atom count, and what the geometry term (alpha 0.5) changes
relative to features only (alpha 0). Figures: a clustered heatmap, a t-SNE map as
small multiples, and per-MOF nearest-neighbour distances.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from mof_embeddings.config import DATA

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY = {"aimd-from-PBE-1000-npt": "AIMD 1000 K", "aimd-from-PBE-1000-nvt": "AIMD 1000 K",
          "aimd-from-PBE-3000-npt": "AIMD 3000 K", "aimd-from-PBE-3000-nvt": "AIMD 3000 K",
          "rattled-300": "rattled 300", "rattled-300-subsampled": "rattled 300",
          "rattled-500": "rattled 500", "rattled-500-subsampled": "rattled 500",
          "rattled-1000": "rattled 1000", "rattled-1000-subsampled": "rattled 1000",
          "rattled-relax": "rattled relax", "MOF-off R2SCAN": "MOF-off"}
GROUP_ORDER = ["MOF-off", "rattled relax", "rattled 300", "rattled 500", "rattled 1000", "AIMD 1000 K", "AIMD 3000 K"]
METALS_EXCLUDED = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53}   # non-metals / metalloids


def knn_agreement(D, labels, k=10, rng=None, shuffles=200):
    """Fraction of each item's k nearest neighbours sharing its label, and the same under label shuffles."""
    nn = np.argsort(D + np.diag(np.full(len(D), np.inf)), 1)[:, :k]
    labels = np.asarray(labels)
    obs = np.mean(labels[nn] == labels[:, None])
    null = [np.mean(p[nn] == p[:, None]) for p in (rng.permutation(labels) for _ in range(shuffles))]
    return obs, float(np.mean(null)), float(np.percentile(null, 99))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(f"{DATA}/structure_fgw"))
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    meta = pd.read_csv(args.root / "sample/structures.csv")
    emb = np.load(args.root / "sample/embeddings.npz")
    off, Z = emb["frame_offsets"], emb["atomic_numbers"]
    meta["group"] = meta.subdataset.map(FAMILY)
    zs = [Z[off[i]:off[i + 1]] for i in range(len(meta))]
    meta["has_CH"] = [bool(np.isin(6, z) and np.isin(1, z)) for z in zs]
    def main_metal(z):
        m = [e for e in z if e not in METALS_EXCLUDED]
        return int(pd.Series(m).mode()[0]) if m else 0
    meta["metal"] = [main_metal(z) for z in zs]
    meta["element_set"] = [" ".join(map(str, np.unique(z))) for z in zs]
    is_mof = (meta.dataset == "MOF-off").to_numpy()
    rng = np.random.default_rng(0)
    report = {}

    D = {a: np.load(args.root / f"pairwise/D_alpha{a}.npy") for a in [0.0, 0.5]}
    for a, M in D.items():
        assert np.allclose(M, M.T) and np.all(np.diag(M) == 0) and np.all(M[~np.eye(len(M), dtype=bool)] > 0)
    Dm = D[0.5]
    iu = np.triu_indices(len(Dm), 1)

    # 1. blocks
    blocks = {}
    for name, (ra, rb) in {"MOF-MOF": (is_mof, is_mof), "OMat-OMat": (~is_mof, ~is_mof), "MOF-OMat": (is_mof, ~is_mof)}.items():
        sub = Dm[np.ix_(ra, rb)]
        vals = sub[np.triu_indices(len(sub), 1)] if name != "MOF-OMat" else sub.ravel()
        blocks[name] = dict(median=float(np.median(vals)), p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)))
    report["blocks_alpha0.5"] = blocks

    # 2. MOFs' nearest neighbours
    near_mof = np.sort(Dm[np.ix_(is_mof, is_mof)] + np.diag(np.full(is_mof.sum(), np.inf)), 1)[:, 0]
    near_omat_idx = np.argmin(Dm[np.ix_(is_mof, ~is_mof)], 1)
    near_omat = Dm[np.ix_(is_mof, ~is_mof)][np.arange(is_mof.sum()), near_omat_idx]
    omat_rows = meta[~is_mof].reset_index(drop=True)
    closest = omat_rows.iloc[near_omat_idx].assign(mof_formula=meta[is_mof].formula.values, distance=near_omat)
    report["mofs_closer_to_an_omat_than_to_any_mof"] = int(np.sum(near_omat < near_mof))
    report["median_nearest_mof_distance"] = float(np.median(near_mof))
    report["median_nearest_omat_distance"] = float(np.median(near_omat))
    hub = closest.groupby(["formula", "group"]).size().sort_values(ascending=False)
    report["omat_structures_most_often_nearest_to_mofs"] = [dict(formula=f, group=g, n_mofs=int(n)) for (f, g), n in hub.head(8).items()]
    report["nearest_omat_contains_C_and_H"] = float(closest.has_CH.mean())
    report["omat_with_C_and_H_in_sample"] = int(omat_rows.has_CH.sum())

    # 3. what organises the matrix
    org = {}
    om = ~is_mof
    for a, M in D.items():
        Mo = M[np.ix_(om, om)]
        org[f"alpha{a}"] = dict(
            omat_family=knn_agreement(Mo, meta.group[om], rng=rng),
            omat_element_set=knn_agreement(Mo, meta.element_set[om], rng=rng),
            omat_main_metal=knn_agreement(Mo, meta.metal[om], rng=rng),
            mof_main_metal=knn_agreement(M[np.ix_(is_mof, is_mof)], meta.metal[is_mof], rng=rng),
            dataset=knn_agreement(M, meta.dataset, rng=rng),
            spearman_with_log_size_ratio=float(spearmanr(M[iu], np.abs(np.log(meta.natoms.values[iu[0]] / meta.natoms.values[iu[1]]))).statistic),
            spearman_with_log_size_ratio_omat_only=float(spearmanr(Mo[np.triu_indices(om.sum(), 1)],
                np.abs(np.log(meta.natoms.values[om][np.triu_indices(om.sum(), 1)[0]] / meta.natoms.values[om][np.triu_indices(om.sum(), 1)[1]]))).statistic))
    report["knn10_label_agreement_observed_null_null99"] = org
    report["spearman_alpha0_vs_alpha0.5"] = float(spearmanr(D[0.0][iu], D[0.5][iu]).statistic)
    nn0 = np.argsort(D[0.0] + np.diag(np.full(len(Dm), np.inf)), 1)[:, :10]
    nn5 = np.argsort(D[0.5] + np.diag(np.full(len(Dm), np.inf)), 1)[:, :10]
    report["shared_10nn_alpha0_vs_alpha0.5"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn0, nn5)]))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    closest.to_csv(out / "nearest_omat_to_each_mof.csv", index=False)
    print(json.dumps(report, indent=2, default=float))

    # Figures ---------------------------------------------------------------
    order = leaves_list(linkage(squareform(Dm, checks=False), "average"))
    fig = plt.figure(figsize=(10.5, 10.0), facecolor=SURFACE)
    ax = fig.add_axes([0.1, 0.04, 0.78, 0.8])
    v = np.percentile(Dm[iu], [1, 99])
    seq = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#2a78d6", "#9ec5f4", SURFACE])
    im = ax.imshow(Dm[np.ix_(order, order)], cmap=seq, vmin=v[0], vmax=v[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    gcol = {g: SLOTS[i] for i, g in enumerate(GROUP_ORDER)}
    strip = np.array([matplotlib.colors.to_rgb(gcol[g]) for g in meta.group.values[order]])
    for pos in ([0.1, 0.855, 0.78, 0.018], [0.075, 0.04, 0.018, 0.8]):
        sa = fig.add_axes(pos)
        sa.imshow(strip[None] if pos[3] < 0.1 else strip[:, None], aspect="auto", interpolation="nearest")
        sa.set_axis_off()
    cax = fig.add_axes([0.9, 0.04, 0.015, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8); cb.set_label("FGW distance (darker = more similar)", color=INK2, fontsize=8.5)
    handles = [matplotlib.patches.Patch(color=gcol[g], label=f"{g} ({(meta.group == g).sum()})") for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.49, 0.965))
    fig.suptitle("Pairwise structure FGW (UMA+MH1 features, σ = 3 Å, α = 0.5) · 1,100 structures, clustered",
                 color=INK, fontsize=12, y=0.99)
    fig.savefig(out / "heatmap_clustered.png", dpi=150, facecolor=SURFACE)

    xy = TSNE(metric="precomputed", init="random", perplexity=30, random_state=0).fit_transform(Dm)
    np.save(out / "tsne_alpha0.5.npy", xy)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), facecolor=SURFACE)
    for axp, g in zip(axes.flat, GROUP_ORDER + ["has C and H"]):
        m = meta.has_CH.values if g == "has C and H" else (meta.group == g).values
        axp.scatter(*xy[~m].T, s=5, color="#c9c8c2", linewidths=0)
        axp.scatter(*xy[m].T, s=11, color=SLOTS[0], linewidths=0)
        axp.set_title(f"{g} ({m.sum()})", color=INK, fontsize=10)
        axp.set_xticks([]); axp.set_yticks([]); axp.set_facecolor(SURFACE)
        for s in axp.spines.values(): s.set_color(GRID)
    fig.suptitle("t-SNE of the FGW distance matrix · each panel highlights one group in blue", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "tsne_groups.png", dpi=150, facecolor=SURFACE)

    fig, axp = plt.subplots(figsize=(5.6, 5.2), facecolor=SURFACE)
    lim = [0, max(near_mof.max(), near_omat.max()) * 1.05]
    axp.plot(lim, lim, color=MUTED, lw=1, ls="--")
    axp.scatter(near_mof, near_omat, s=18, color=SLOTS[0], edgecolor=SURFACE, linewidth=0.6)
    axp.set_xlim(lim); axp.set_ylim(lim)
    axp.set_xlabel("distance to nearest other MOF", color=INK2); axp.set_ylabel("distance to nearest OMat24 structure", color=INK2)
    axp.set_title("Each MOF's nearest neighbours (points above the line: a MOF is closer)", color=INK, fontsize=10)
    axp.tick_params(colors=INK2, labelsize=8); axp.set_facecolor(SURFACE)
    for s in axp.spines.values(): s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(out / "mof_nearest_neighbours.png", dpi=150, facecolor=SURFACE)
    print("figures saved to", out)


if __name__ == "__main__":
    main()
