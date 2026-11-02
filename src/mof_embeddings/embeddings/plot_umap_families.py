"""Full atom UMAPs of OMat24 sr_1m (18.7M atoms) for UMA and MACE-MH1, fitted
with identical settings on OMat24 only, shown as small multiples: the first
column is every atom (log density), then one column per element family
highlighted over the grey cloud. Families fold the archive's 11 element groups
into 8 (H and noble gases join the non-metals, lanthanides and actinides form f-block)."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mof_embeddings.config import ARCHIVE, RUNS

LABELS = Path(f"{ARCHIVE}/embeddings/uma/atom/omat24_sr1m/labels_element_group")
RUN = Path(f"{RUNS}/atom_umap_sr1m")
FOLD = {"hydrogen": "non-metals (H, C, N, O, P, S…)", "reactive nonmetals": "non-metals (H, C, N, O, P, S…)",
        "noble gases": "non-metals (H, C, N, O, P, S…)", "alkali metals": "alkali metals",
        "alkaline earth metals": "alkaline-earth metals", "transition metals": "transition metals",
        "post-transition metals": "post-transition metals", "metalloids": "metalloids", "halogens": "halogens",
        "lanthanides": "f-block (Ln, An)", "actinides": "f-block (Ln, An)"}
FAMILIES = ["non-metals (H, C, N, O, P, S…)", "halogens", "alkali metals", "alkaline-earth metals",
            "transition metals", "post-transition metals", "metalloids", "f-block (Ln, An)"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GREY = LinearSegmentedColormap.from_list("grey", [SURFACE, "#c9c8c2"])
DENS = LinearSegmentedColormap.from_list("dens", [SURFACE, "#9ec5f4", "#2a78d6", "#0d366b"])
HIGHLIGHT = LinearSegmentedColormap.from_list("hl", ["#f6c8a8", "#b8642a", "#6e3412"])
BINS = 320


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN / "umap_families.png")
    args = parser.parse_args()
    names = json.loads((LABELS / "row_labels.json").read_text())["names"]
    fam = np.array([FOLD[n] for n in names])[np.load(LABELS / "row_labels.npy")]
    rows = [("UMA", np.load(RUN / "umap_uma_raw128.npy")), ("MACE-MH1", np.load(RUN / "umap_mh1_readout96.npy"))]
    cols = ["all atoms"] + FAMILIES
    fig, axes = plt.subplots(2, len(cols), figsize=(2.05 * len(cols), 4.6), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.03, hspace=0.18))
    for r, (model, xy) in enumerate(rows):
        lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
        pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
        ex = [[lo[0], hi[0]], [lo[1], hi[1]]]
        bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=ex)
        for c, name in enumerate(cols):
            ax = axes[r, c]; ax.set_facecolor(SURFACE)
            if c == 0:
                ax.imshow(np.ma.masked_where(bg == 0, np.log1p(bg)).T, origin="lower", extent=[*ex[0], *ex[1]],
                          cmap=DENS, aspect="auto", interpolation="nearest")
            else:
                m = fam == name
                h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
                ax.imshow(np.log1p(bg).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=GREY, aspect="auto", interpolation="nearest")
                ax.imshow(np.ma.masked_where(h == 0, np.log1p(h)).T, origin="lower", extent=[*ex[0], *ex[1]],
                          cmap=HIGHLIGHT, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color("#e3e2de")
            if r == 0:
                label = name if c == 0 else f"{name}\n{(fam == name).mean():.0%} of atoms"
                ax.set_title(label, fontsize=9, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(model, fontsize=11, color=INK, labelpad=6)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
