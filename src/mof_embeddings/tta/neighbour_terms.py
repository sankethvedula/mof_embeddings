#!/usr/bin/env python
"""How close are the fine-tuning structures to the target, term by term, and does it predict the gain?

terms:  for every target and selection method (fgw, global, random), the 105 structures of the
        100-structure set (training ranks 1-100, validation 101-105; the 20-structure set is ranks
        1-20) are scored against the target's query frame with structure FGW (UMA features,
        intensive periodic kernel, the constants of the 10M search) and split at the optimal plan T:
          feature_term    <M, T>, UMA feature cost of the matched atoms (chemical environment)
          geometry_term   GW loss of T on the two kernels (local geometry / packing)
          fgw             (1 - alpha) feature_term + alpha geometry_term
          same_element    share of the target's atoms sent to atoms of the same element
          metal_to_metal  share of the target's metal atoms sent to metal atoms
        -> data/tta/neighbour_terms/<target>.csv (one row per structure)
relate: averages the terms over each training set (20 or 100 structures), joins the change of the
        target force MAE after 60 epochs (tta.compare final.csv) and reports Spearman correlations
        and figures.
        -> data/tta/neighbour_terms/sets.csv, correlations.csv, terms_vs_gain.png

  $PY -m mof_embeddings.tta.neighbour_terms terms [--targets Q6 Q3 ...] [--workers 10]
  $PY -m mof_embeddings.tta.neighbour_terms relate
"""
import argparse
import json

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA, use_local_packages
from mof_embeddings.tta.common import LR, METHODS, N_TRAIN, SEED, TARGETS, run_dir, selection_name, target_query

OUT = DATA / "tta" / "neighbour_terms"
NONMETAL = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te",
            "I", "Xe"}
TERMS = [("fgw", "FGW distance"), ("feature_term", "feature term ⟨M,T⟩"),
         ("geometry_term", "geometry term (GW)"), ("same_element", "same-element share")]


def score_target(target):
    use_local_packages()
    import ot
    from ase.data import chemical_symbols
    from ase.db import connect
    from scipy.spatial.distance import cdist
    from mof_embeddings.mof_search.common import OUT as SEARCH, SR10M, SR10M_LMDB, uma_stats
    from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

    const = json.loads((SEARCH / "constants.json").read_text())
    m0, k0, sigma, alpha = const["m0"], const["k0"], const["sigma"], const["alpha"]
    mu, sd = uma_stats()
    f1, _, a1 = target_query(target, mu, sd)
    C, _ = periodic_kernel_fast(np.array(a1.cell), a1.positions, sigma)
    K1 = len(C) * C / k0
    metal1 = np.array([chemical_symbols[z] not in NONMETAL for z in a1.numbers])
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    p = ot.unif(len(f1))
    rows = []
    for method in METHODS:
        d = run_dir(target, selection_name(method, 100))
        if not d.exists():
            d = run_dir(target, method)                # 20-structure set only
        if not d.exists():
            continue
        sets = pd.read_csv(d / "sets.csv")
        for r in sets.itertuples():
            f = int(r.sr10m_frame)
            at = db.get_atoms(id=f + 1)
            f2 = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
            C2, _ = periodic_kernel_fast(np.array(at.cell), at.positions, sigma)
            K2 = len(C2) * C2 / k0
            M = cdist(f1, f2, "sqeuclidean") / m0
            pq = ot.unif(len(f2))
            best = (np.inf, None)
            for G0 in [ot.emd(p, pq, M), None]:
                T, log = ot.gromov.fused_gromov_wasserstein(M, K1, K2, p, pq, loss_fun="square_loss", alpha=alpha,
                                                            G0=G0, log=True, max_iter=200)
                if log["fgw_dist"] < best[0]:
                    best = (float(log["fgw_dist"]), T)
            T = best[1]
            gw = float(ot.gromov.gwloss(*ot.gromov.init_matrix(K1, K2, p, pq, "square_loss"), T))
            same = a1.numbers[:, None] == at.numbers[None, :]
            metal2 = np.array([chemical_symbols[z] not in NONMETAL for z in at.numbers])
            mt = T[metal1].sum()
            rows.append(dict(target=target, method=method, rank=int(r.rank), split=r.split, sr10m_frame=f,
                             formula=r.formula, natoms=int(r.natoms), fgw=best[0],
                             feature_term=float(np.sum(M * T)), geometry_term=gw,
                             same_element=float(T[same].sum() / T.sum()),
                             metal_to_metal=float(T[np.ix_(metal1, metal2)].sum() / mt) if mt > 0 else np.nan))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{target}.csv", index=False)
    return f"{target}: {len(df)} structures scored"


def set_means():
    """Mean terms over the training structures of each (target, method, size)."""
    rows = []
    for t in TARGETS:
        path = OUT / f"{t}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for method, n in [(m, n) for m in METHODS for n in [N_TRAIN, 100]]:
            s = d[(d.method == method) & (d["rank"] <= n)]        # ranks 1-100 are the training structures
            if s["rank"].max() < n:                                  # no set of this size
                continue
            rows.append(dict(target=t, method=method, n_train=n, **{c: s[c].mean() for c in
                             ["fgw", "feature_term", "geometry_term", "same_element", "metal_to_metal"]},
                             distinct_formulas=s.formula.nunique()))
    return pd.DataFrame(rows)


def relate():
    from scipy.stats import spearmanr
    sets = set_means()
    fin = pd.read_csv(DATA / "tta" / "final.csv")
    fin = fin[fin.seed == SEED] if "seed" in fin else fin
    fin = fin[fin.variant.fillna("") == ""] if "variant" in fin else fin       # default training protocol
    df = fin.merge(sets, on=["target", "method", "n_train"])
    sets.to_csv(OUT / "sets.csv", index=False)
    rows = []
    for (n, lr), g in df.groupby(["n_train", "lr"]):
        for subset, h in [("all methods", g), ("neighbour sets (fgw, global)", g[g.method != "random"])]:
            for col, _ in TERMS + [("metal_to_metal", "")]:
                rho, pv = spearmanr(h[col], h.force_change_pct, nan_policy="omit")
                rows.append(dict(n_train=n, lr=lr, subset=subset, term=col, n=len(h), spearman=rho, p=pv))
    cor = pd.DataFrame(rows)
    cor.to_csv(OUT / "correlations.csv", index=False)
    plot(df)
    return df, cor


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    from mof_embeddings.tta.compare import COL, LAB
    panels = [(N_TRAIN, LR), (N_TRAIN, 1e-5), (100, LR)]
    fig, axes = plt.subplots(len(panels), len(TERMS), figsize=(3.3 * len(TERMS), 2.9 * len(panels) + 0.4),
                             facecolor=SURFACE, squeeze=False)
    for r, (n, lr) in enumerate(panels):
        g = df[(df.n_train == n) & np.isclose(df.lr, lr)]
        for c, (col, lab) in enumerate(TERMS):
            ax = axes[r, c]
            for m in METHODS:
                h = g[g.method == m]
                ax.scatter(h[col], h.force_change_pct, s=26, color=COL[m], label=LAB[m], zorder=3,
                           edgecolor="white", lw=0.5)
                for x, y, t in zip(h[col], h.force_change_pct, h.target):
                    ax.annotate(t, (x, y), fontsize=5.5, color=INK2, xytext=(2, 2), textcoords="offset points")
            ax.axhline(0, color=INK, lw=0.8)
            rho = spearmanr(g[col], g.force_change_pct).statistic
            nb = g[g.method != "random"]
            rho_nb = spearmanr(nb[col], nb.force_change_pct).statistic
            ax.set_title(f"ρ = {rho:+.2f} (all), {rho_nb:+.2f} (neighbours)", fontsize=8, color=INK)
            if r == len(panels) - 1:
                ax.set_xlabel(f"{lab}, mean over training set", fontsize=8.5, color=INK2)
            if c == 0:
                ax.set_ylabel(f"{n} structures, lr {lr:g}\nchange in target force MAE (%)", fontsize=8.5,
                              color=INK2)
            ax.tick_params(labelsize=7.5, colors=INK2); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
            for s_ in ["top", "right"]:
                ax.spines[s_].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(OUT / "terms_vs_gain.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["terms", "relate"])
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "terms":
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                for msg in pool.imap_unordered(score_target, args.targets):
                    print(msg, flush=True)
        else:
            for t in args.targets:
                print(score_target(t), flush=True)
    else:
        df, cor = relate()
        pd.set_option("display.width", 200)
        print(cor[cor.subset == "all methods"].pivot_table(index=["n_train", "lr"], columns="term",
                                                           values="spearman").round(2))
        print(cor[cor.subset != "all methods"].pivot_table(index=["n_train", "lr"], columns="term",
                                                           values="spearman").round(2))


if __name__ == "__main__":
    main()
