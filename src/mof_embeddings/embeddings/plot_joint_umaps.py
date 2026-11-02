#!/usr/bin/env python
"""Figures for the joint UMA / MACE-MH1 atom UMAPs (joint_atom_umap.py) coloured
by attributes (atom_attributes.py):
  joint_by_dataset.png        all atoms, then each dataset highlighted
  joint_by_omat_subset.png    OMat24 atoms by generation recipe
  joint_by_attribute.png      per-pixel median percentile of |F|, atoms per structure and
                              volume per atom (percentiles over all atoms, so OMat24's 90%
                              share does not wash the colours out; colour bars give values)
  joint_by_element.png        element families (all datasets)
  coverage.csv                per dataset and map: share of its atoms in pixels where it is
                              >= 10x enriched relative to OMat24 (or OMat24 is absent)
Rows are the two models.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy.stats import binned_statistic_2d

from mof_embeddings.config import RUNS, SCRATCH
from mof_embeddings.embeddings.joint_atom_umap import BANK, DATASETS
from mof_embeddings.embeddings.plot_umap_families import DENS, FAMILIES, FOLD, GREY, HIGHLIGHT, INK, INK2, SURFACE

SUBSET = {"aimd-from-PBE-1000-npt": "AIMD 1000 K", "aimd-from-PBE-1000-nvt": "AIMD 1000 K", "aimd-from-PBE-3000-npt": "AIMD 3000 K",
          "aimd-from-PBE-3000-nvt": "AIMD 3000 K", "rattled-relax": "rattled relax", "rattled-300": "rattled 300",
          "rattled-300-subsampled": "rattled 300", "rattled-500": "rattled 500", "rattled-500-subsampled": "rattled 500",
          "rattled-1000": "rattled 1000", "rattled-1000-subsampled": "rattled 1000"}
SUBSETS = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax", "rattled 300", "rattled 500", "rattled 1000"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#e7f0fb", "#86b6ef", "#2a78d6", "#104281", "#0d1f3a"])
BINS = 300


def labels(rows, key):
    """Per joint row: element family ('family') or OMat24 subset ('subset', None outside OMat24)."""
    out = np.empty(len(rows["dataset"]), dtype=object)
    for k, (_, d) in enumerate(DATASETS):
        m = rows["dataset"] == k
        lab_dir = BANK["UMA"](d) / ("labels_element_group" if key == "family" else "labels_subdataset")
        if key == "subset" and k:
            continue
        names = json.loads((lab_dir / "row_labels.json").read_text())["names"]
        lab = np.array(names)[np.load(lab_dir / "row_labels.npy")[rows["source_row"][m]]]
        out[m] = [FOLD[x] for x in lab] if key == "family" else [SUBSET[x] for x in lab]
    return out


def attributes(rows, att_dir):
    out = {k: np.empty(len(rows["dataset"]), np.float32) for k in ["force", "natoms", "volume_per_atom"]}
    for k, (name, _) in enumerate(DATASETS):
        z = np.load(att_dir / f"attributes_{name}.npz")
        m = rows["dataset"] == k
        for key in out:
            out[key][m] = z[key][rows["source_row"][m]]
    return out


def frame(xy):
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0); pad = 0.04 * (hi - lo)
    return [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]


def style(ax, title=None, ylabel=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
    for s in ax.spines.values(): s.set_color("#e3e2de")
    if title: ax.set_title(title, fontsize=9, color=INK, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=11, color=INK, labelpad=6)


def highlight_grid(maps, masks, titles, out, suptitle):
    fig, axes = plt.subplots(len(maps), len(titles) + 1, figsize=(2.05 * (len(titles) + 1), 2.3 * len(maps)), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.03, hspace=0.12), squeeze=False)
    for r, (model, xy) in enumerate(maps):
        ex = frame(xy)
        bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=ex)
        ax = axes[r, 0]
        ax.imshow(np.ma.masked_where(bg == 0, np.log1p(bg)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=DENS, aspect="auto", interpolation="nearest")
        style(ax, "all atoms" if r == 0 else None, model)
        for c, (m, t) in enumerate(zip(masks, titles), 1):
            h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
            ax = axes[r, c]
            ax.imshow(np.log1p(bg).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=GREY, aspect="auto", interpolation="nearest")
            ax.imshow(np.ma.masked_where(h == 0, np.log1p(h)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=HIGHLIGHT, aspect="auto", interpolation="nearest")
            style(ax, t if r == 0 else None)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11.5, color=INK, y=1.02)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(f"{SCRATCH}/joint_umap"))
    parser.add_argument("--out", type=Path, default=Path(f"{RUNS}/joint_umap"))
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    title = (lambda t: None) if args.no_title else (lambda t: t)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = dict(np.load(args.dir / "rows.npz", allow_pickle=True))
    maps = [(m, np.load(args.dir / f"umap_{m}.npy")) for m in ["UMA", "MH1"]]
    maps = [("MACE-MH1" if m == "MH1" else m, xy) for m, xy in maps]
    ds = rows["dataset"]; names = list(rows["names"])
    n = [int((ds == k).sum()) for k in range(len(names))]

    highlight_grid(maps, [ds == k for k in range(len(names))], [f"{nm}\n{c / 1e6:.2f}M atoms" for nm, c in zip(names, n)],
                   args.out / "joint_by_dataset.png", title("Joint atom UMAPs, OMat24 sr_1m + MOF-off, MAD, MP-ALOE, MatPES samples · each dataset highlighted"))
    sub = labels(rows, "subset")
    highlight_grid(maps, [sub == s for s in SUBSETS], [f"OMat24 · {s}" for s in SUBSETS],
                   args.out / "joint_by_omat_subset.png", title("OMat24 atoms by how the frame was generated"))
    fam = labels(rows, "family")
    highlight_grid(maps, [fam == f for f in FAMILIES], FAMILIES, args.out / "joint_by_element.png", title("Element families, all datasets"))

    att = attributes(rows, args.dir)
    specs = [("force", "|F|", "eV/Å"), ("natoms", "atoms per structure", ""), ("volume_per_atom", "volume per atom", "Å³")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), facecolor=SURFACE, gridspec_kw=dict(wspace=0.12, hspace=0.12))
    for c, (key, name, unit) in enumerate(specs):
        v = att[key]; ok = np.isfinite(v)
        pct = np.full(len(v), np.nan, np.float32)
        order = np.argsort(v[ok], kind="stable")
        pct[np.flatnonzero(ok)[order]] = np.linspace(0, 100, ok.sum(), dtype=np.float32)
        for r, (model, xy) in enumerate(maps):
            ex = frame(xy)
            stat, _, _, _ = binned_statistic_2d(xy[ok, 0], xy[ok, 1], pct[ok], statistic="median", bins=BINS, range=ex)
            ax = axes[r, c]
            im = ax.imshow(np.ma.masked_invalid(stat).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=SEQ, vmin=0, vmax=100,
                           aspect="auto", interpolation="nearest")
            style(ax, f"median percentile of {name}" if r == 0 else None, model if c == 0 else None)
        cb = fig.colorbar(im, ax=axes[:, c], orientation="horizontal", fraction=0.04, pad=0.03)
        q = [5, 25, 50, 75, 95]; vals = np.percentile(v[ok], q)
        cb.set_ticks(q); cb.set_ticklabels([f"{x:.3g}" for x in vals])
        cb.set_label(f"{name}{f' ({unit})' if unit else ''} at that percentile", fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2); cb.outline.set_visible(False)
    if not args.no_title:
        fig.suptitle("Joint atom UMAPs coloured by physical attributes (same maps as the dataset figure)", fontsize=11.5, color=INK, y=0.95)
    fig.savefig(args.out / "joint_by_attribute.png", dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print("saved", args.out / "joint_by_attribute.png")

    rows_cov = []
    for model, xy in maps:
        ex = frame(xy)
        h0, xe, ye = np.histogram2d(xy[ds == 0, 0], xy[ds == 0, 1], bins=BINS, range=ex)
        for k in range(1, len(names)):
            m = ds == k
            h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
            enrich = (h / h.sum()) / np.maximum(h0 / h0.sum(), 1e-12)
            rows_cov.append(dict(map=model, dataset=names[k], atoms=int(m.sum()),
                                 share_where_omat_absent=float(h[h0 == 0].sum() / h.sum()),
                                 share_enriched_10x=float(h[enrich >= 10].sum() / h.sum())))
    import pandas as pd
    cov = pd.DataFrame(rows_cov); cov.to_csv(args.out / "coverage.csv", index=False)
    print(cov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
