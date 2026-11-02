"""Small multiples: where do individual elements sit in 2D projections of
all 18.7M OMat24 sr_1m atom embeddings? Rows: UMA UMAP, UMA PCA, MH1-96 PCA."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mof_embeddings.config import ARCHIVE, RUNS

A = f"{ARCHIVE}/"
U = A + "embeddings/uma/atom/omat24_sr1m/"
M = A + "embeddings/mace_mh1/atom/omat24_sr1m/readout96/atom_latents.npy"
N = 18_682_025
CHUNK = 2_000_000
ELEMENTS = [("H", 1), ("Li", 3), ("O", 8), ("F", 9), ("Si", 14), ("Fe", 26), ("Cu", 29), ("La", 57)]
BINS = 300
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

Z = np.load(U + "atom_atomic_numbers.npy")

def mh1_pca2():
    x = np.load(M, mmap_mode="r")
    s, ss = np.zeros(96), np.zeros((96, 96))
    for i in range(0, N, CHUNK):
        c = np.asarray(x[i:i + CHUNK], dtype=np.float64); s += c.sum(0); ss += c.T @ c
    mu = s / N; cov = ss / N - np.outer(mu, mu)
    w, v = np.linalg.eigh(cov); v = v[:, ::-1][:, :2]; evr = w[::-1][:2] / w.sum()
    out = np.empty((N, 2), np.float32)
    for i in range(0, N, CHUNK):
        out[i:i + CHUNK] = (np.asarray(x[i:i + CHUNK], dtype=np.float64) - mu) @ v
    return out, evr

rows = [
    ("UMA\nUMAP", np.load(A + "derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/umap_raw128.npy", mmap_mode="r")[:N]),
    ("UMA\nPCA 1–2 (44% var)", np.load(A + "derived/uma_atom_pca_sr_1m/pc25_all.npy", mmap_mode="r")[:, :2]),
]
mh, evr = mh1_pca2()
rows.append((f"MACE-MH1\nPCA 1–2 ({100*evr.sum():.0f}% var)", mh))

blue = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#104281"])
grey = LinearSegmentedColormap.from_list("grey", [SURFACE, "#c9c8c2"])
fig, axes = plt.subplots(len(rows), len(ELEMENTS), figsize=(2.1 * len(ELEMENTS), 2.35 * len(rows)),
                         facecolor=SURFACE, gridspec_kw=dict(wspace=0.04, hspace=0.12))
for r, (title, xy) in enumerate(rows):
    xy = np.asarray(xy, dtype=np.float32)
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)          # clip extreme outliers for display
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    ex = [[lo[0], hi[0]], [lo[1], hi[1]]]
    bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=ex)
    for c, (sym, z) in enumerate(ELEMENTS):
        ax = axes[r, c]; ax.set_facecolor(SURFACE)
        m = Z == z
        h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
        ax.imshow(np.log1p(bg).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=grey, aspect="auto", interpolation="nearest")
        hm = np.ma.masked_where(h == 0, np.log1p(h))
        ax.imshow(hm.T, origin="lower", extent=[*ex[0], *ex[1]], cmap=blue, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#e3e2de")
        if r == 0: ax.set_title(f"{sym}  ·  {m.sum()/1e3:,.0f}k atoms", fontsize=10, color=INK, pad=5)
        if c == 0: ax.set_ylabel(title, fontsize=9.5, color=INK2, labelpad=6)
fig.suptitle("Where single elements sit among all 18.7M OMat24 atoms (grey = all atoms, blue = highlighted element, log density)",
             fontsize=11.5, color=INK, x=0.5, y=0.995)
out = RUNS / "atom_element_clustering" / "atom_element_maps.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
print("saved", out, "MH1 PCA evr", evr)
