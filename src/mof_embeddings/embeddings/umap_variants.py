#!/usr/bin/env python
"""cuML UMAP of every per-atom feature variant (evaluate_mace_variants.VARIANTS)
on the 50k-structure OMat24 sr_1m sample (~0.93M atoms), with the same settings
as the full UMA / MH1 atom UMAPs, then a small-multiples figure (one row per
variant; columns: all atoms, then each element family) and the 10-NN element
vote in each 2D layout. --plot-only redraws from saved coordinates. Also writes
umap_variants_compact.png (one panel per variant, transition metals and non-metals
highlighted) for slides.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mof_embeddings.config import ARCHIVE
from mof_embeddings.embeddings.evaluate_mace_variants import OUT, U, VARIANTS, Features, stats

UMA_FIT = Path(f"{ARCHIVE}/derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/fit.json")


def sample_labels(F, frames):
    names = json.loads((U / "labels_element_group/row_labels.json").read_text())["names"]
    zb = np.load(U / "atom_atomic_numbers.npy", mmap_mode="r"); gb = np.load(U / "labels_element_group/row_labels.npy", mmap_mode="r")
    Z = np.concatenate([np.asarray(zb[F.bank_off[f]:F.bank_off[f + 1]]) for f in frames])
    G = np.array(names)[np.concatenate([np.asarray(gb[F.bank_off[f]:F.bank_off[f + 1]]) for f in frames])]
    return Z, G


def knn_vote_2d(xy, Z, n=60_000, seed=0):
    import torch
    idx = np.random.default_rng(seed).choice(len(xy), n, replace=False)
    x = torch.from_numpy(xy[idx].astype(np.float32)); z = Z[idx].astype(int)
    nn = []
    for i in range(0, n, 4096):
        d = torch.cdist(x[i:i + 4096], x)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        nn.append(d.topk(10, largest=False).indices)
    nn = torch.cat(nn).numpy()
    return float(np.mean([np.bincount(r).argmax() == zz for r, zz in zip(z[nn], z)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = OUT / "umap"; out.mkdir(parents=True, exist_ok=True)
    F = Features()
    frames = F.frames[:50000]
    Z, G = sample_labels(F, frames)
    if not args.plot_only:
        from cuml.manifold import UMAP
        params = json.loads(UMA_FIT.read_text())["params"]
        st = stats(F, frames[::10])
        for v in VARIANTS:
            mu, sd = st[v]
            X = np.concatenate([(F.get(v, f) - mu) / sd for f in frames]).astype(np.float32)
            xy = np.asarray(UMAP(**params).fit_transform(X), dtype=np.float32)
            np.save(out / f"{v.replace(' ', '_')}.npy", xy)
            print(f"{v}: {X.shape} done", flush=True)
    votes = {v: knn_vote_2d(np.load(out / f"{v.replace(' ', '_')}.npy"), Z) for v in VARIANTS}
    (out / "knn_vote_2d.json").write_text(json.dumps(votes, indent=2))
    print(json.dumps(votes, indent=2))
    plot(out, Z, G, votes)
    plot_compact(out, G, votes)


def plot_compact(out, G, votes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from mof_embeddings.embeddings.plot_umap_families import FAMILIES, FOLD, GREY, INK, SURFACE
    fam = np.array([FOLD[g] for g in G])
    hi = [("transition metals", "#b8642a"), (FAMILIES[0], "#1f8a83")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.9), facecolor=SURFACE, gridspec_kw=dict(wspace=0.04, hspace=0.22))
    for ax, v in zip(axes.flat, VARIANTS):
        xy = np.load(out / f"{v.replace(' ', '_')}.npy")
        lo, hi_ = np.percentile(xy, [0.5, 99.5], axis=0); pad = 0.04 * (hi_ - lo); lo, hi_ = lo - pad, hi_ + pad
        ex = [[lo[0], hi_[0]], [lo[1], hi_[1]]]
        bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=220, range=ex)
        ax.imshow(np.ma.masked_where(bg == 0, np.log1p(bg)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=GREY, aspect="auto", interpolation="nearest")
        for name, col in hi:
            h, _, _ = np.histogram2d(xy[fam == name, 0], xy[fam == name, 1], bins=[xe, ye])
            cm = LinearSegmentedColormap.from_list(name, ["#ffffff", col])
            ax.imshow(np.ma.masked_where(h == 0, np.log1p(h)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=cm, vmin=-1,
                      aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        for sp in ax.spines.values(): sp.set_color("#e3e2de")
        ax.set_title(f"{v}\n2D element vote {votes[v]:.0%}", fontsize=10.5, color=INK, pad=5)
    fig.legend(handles=[Patch(color=c, label=n) for n, c in hi] + [Patch(color="#c9c9c4", label="all other atoms")],
               loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "umap_variants_compact.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out / "umap_variants_compact.png")


def plot(out, Z, G, votes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mof_embeddings.embeddings.plot_umap_families import DENS, FAMILIES, FOLD, GREY, HIGHLIGHT, INK, SURFACE
    fam = np.array([FOLD[g] for g in G])
    cols = ["all atoms"] + FAMILIES
    fig, axes = plt.subplots(len(VARIANTS), len(cols), figsize=(2.0 * len(cols), 2.05 * len(VARIANTS)), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.03, hspace=0.1))
    for r, v in enumerate(VARIANTS):
        xy = np.load(out / f"{v.replace(' ', '_')}.npy")
        lo, hi = np.percentile(xy, [0.5, 99.5], axis=0); pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
        ex = [[lo[0], hi[0]], [lo[1], hi[1]]]
        bg, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=220, range=ex)
        for c, name in enumerate(cols):
            ax = axes[r, c]; ax.set_facecolor(SURFACE)
            if c == 0:
                ax.imshow(np.ma.masked_where(bg == 0, np.log1p(bg)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=DENS, aspect="auto", interpolation="nearest")
            else:
                m = fam == name
                h, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=[xe, ye])
                ax.imshow(np.log1p(bg).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=GREY, aspect="auto", interpolation="nearest")
                ax.imshow(np.ma.masked_where(h == 0, np.log1p(h)).T, origin="lower", extent=[*ex[0], *ex[1]], cmap=HIGHLIGHT, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color("#e3e2de")
            if r == 0:
                ax.set_title(name, fontsize=8.5, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(f"{v}\n2D element vote {votes[v]:.0%}", fontsize=8.5, color=INK, labelpad=5)
    fig.savefig(out / "umap_variants.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out / "umap_variants.png")


if __name__ == "__main__":
    main()
