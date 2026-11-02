"""Element maps across atom UMAPs of the OMat24 sr_1m atoms: Aryan's joint UMA
fit, our OMat-only UMA refit, and our OMat-only MACE-MH1 fit (rows that are
missing from --run-dir are skipped). Grey = all atoms, blue = highlighted
element. Also writes 10-NN element agreement per layout to element_knn.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mof_embeddings.config import ARCHIVE

A = f"{ARCHIVE}/"
U = A + "embeddings/uma/atom/omat24_sr1m/"
UMA_JOINT = A + "derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/umap_raw128.npy"
ELEMENTS = [("H", 1), ("Li", 3), ("O", 8), ("F", 9), ("Si", 14), ("Fe", 26), ("Cu", 29), ("La", 57)]
BINS = 300
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#104281"])
GREY = LinearSegmentedColormap.from_list("grey", [SURFACE, "#c9c8c2"])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, help="default: <run-dir>/umap_element_comparison.png")
parser.add_argument("--no-title", action="store_true")
args = parser.parse_args()

candidates = [("uma_joint", "UMA · UMAP\njoint fit, 36.6M atoms\n(Aryan)", Path(UMA_JOINT)),
              ("uma_omat", "UMA · UMAP\nOMat24 only", args.run_dir / "umap_uma_raw128.npy"),
              ("mh1_omat", "MACE-MH1 · UMAP\nOMat24 only", args.run_dir / "umap_mh1_readout96.npy")]
rows = [(k, t, np.load(p, mmap_mode="r")) for k, t, p in candidates if p.exists()]
n = min(len(xy) if k != "uma_joint" else 18_682_025 for k, _, xy in rows)
rows = [(k, t, np.asarray(xy[:n], dtype=np.float32)) for k, t, xy in rows]
Z = np.asarray(np.load(U + "atom_atomic_numbers.npy", mmap_mode="r")[:n])


def knn_element_agreement(xy, n_frames=5000, k=10, seed=0):
    """Same 5,000-structure sample and metric as atom_element_clustering.py."""
    import torch
    off = np.load(U + "frame_atom_offsets.npy")
    off = off[off <= n]
    frames = np.sort(np.random.default_rng(seed).choice(len(off) - 1, n_frames, replace=False))
    idx = np.concatenate([np.arange(off[f], off[f + 1]) for f in frames])
    x, z = torch.from_numpy(xy[idx]), Z[idx].astype(int)
    nb = []
    for i in range(0, len(x), 4096):
        d = torch.cdist(x[i:i + 4096], x)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        nb.append(d.topk(k, largest=False).indices)
    nb = torch.cat(nb).numpy()
    vote = np.array([np.bincount(r).argmax() for r in z[nb]])
    return dict(knn_element_acc=float((vote == z).mean()), nbr_same_element=float((z[nb] == z[:, None]).mean()))


scores = {k: knn_element_agreement(xy) for k, _, xy in rows}
(args.run_dir / "element_knn.json").write_text(json.dumps(scores, indent=2) + "\n")
print(json.dumps(scores, indent=2))

fig, axes = plt.subplots(len(rows), len(ELEMENTS), figsize=(2.1 * len(ELEMENTS), 2.35 * len(rows)),
                         facecolor=SURFACE, gridspec_kw=dict(wspace=0.04, hspace=0.12), squeeze=False)
for r, (key, title, xy) in enumerate(rows):
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    ex = [[lo[0], hi[0]], [lo[1], hi[1]]]
    bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=ex)
    for c, (sym, z) in enumerate(ELEMENTS):
        ax = axes[r, c]; ax.set_facecolor(SURFACE)
        m = Z == z
        h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
        ax.imshow(np.log1p(bg).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=GREY, aspect="auto", interpolation="nearest")
        ax.imshow(np.ma.masked_where(h == 0, np.log1p(h)).T, origin="lower", extent=[*ex[0], *ex[1]],
                  cmap=BLUE, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#e3e2de")
        if r == 0: ax.set_title(f"{sym}  ·  {m.sum()/1e3:,.0f}k atoms", fontsize=10, color=INK, pad=5)
        if c == 0:
            ax.set_ylabel(f"{title}\n10-NN element match {100*scores[key]['knn_element_acc']:.0f}%",
                          fontsize=9, color=INK2, labelpad=6)
if not args.no_title:
    fig.suptitle(f"Atom UMAPs of {n:,} OMat24 sr_1m atoms · same UMAP settings in every row "
                 "(grey = all atoms, blue = element, log density)", fontsize=11.5, color=INK, x=0.5, y=1.0)
out = args.out or args.run_dir / "umap_element_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
print("saved", out)
