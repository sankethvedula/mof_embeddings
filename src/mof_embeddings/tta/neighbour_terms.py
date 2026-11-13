#!/usr/bin/env python
"""How close are the fine-tuning structures to the target, term by term, and does it predict the gain?

terms:  for every target and selection method (tta.common.METHODS), the structures of the largest
        set (100-structure set: training ranks 1-100, validation 101-105, whose ranks 1-20 are the
        20-structure set; otherwise the 25 structures of the 20-structure set) are scored against
        the target's query frame with structure FGW at one fixed yardstick (UMA features, intensive
        periodic kernel, the constants of the 10M search including alpha 0.5, whatever alpha picked
        the set) and split at the optimal plan T:
          feature_term    <M, T>, UMA feature cost of the matched atoms (chemical environment)
          geometry_term   GW loss of T on the two kernels (local geometry / packing)
          fgw             (1 - alpha) feature_term + alpha geometry_term
          same_element    share of the target's atoms sent to atoms of the same element
          metal_to_metal  share of the target's metal atoms sent to metal atoms
        -> data/tta/neighbour_terms/<target>.csv (one row per structure); methods already in the
           file are kept, so new selections are scored incrementally (--overwrite rescores all)
relate: averages the terms over each training set (20 or 100 structures) and relates them to the
        change of each target metric after 60 epochs versus MACE-OMAT-0: force MAE, offset-free
        energy MAE, stress MAE, raw energy MAE (20-structure sets, seeds averaged, tta.effect), and
        force MAE for the 100-structure sets (tta.compare final.csv). Spearman correlations
          across sets     every (target, set) pair; mixes target difficulty with set quality
          within target   term and change minus the target's mean over its sets, so only the
                          choice of training set for a given target is compared
        over all sets, neighbour sets only (no random), and neighbour sets of targets without a +U
        metal; the +U share of the set (tta.effect) is related the same way for comparison.
        -> data/tta/neighbour_terms/sets.csv, correlations.csv, terms_vs_gain.png (lr 3e-5, three
           metrics), terms_vs_gain_force.png (force at the other learning rates and set sizes)

  $PY -m mof_embeddings.tta.neighbour_terms terms [--targets Q6 S0 ...] [--workers 10] [--overwrite]
  $PY -m mof_embeddings.tta.neighbour_terms relate
"""
import argparse
import json

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA, use_local_packages
from mof_embeddings.tta.common import METHODS, N_TRAIN, SEED, TARGETS, run_dir, selection_name, target_query

OUT = DATA / "tta" / "neighbour_terms"
NONMETAL = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te",
            "I", "Xe"}
TERMS = [("fgw", "FGW distance"), ("feature_term", "feature term ⟨M,T⟩"),
         ("geometry_term", "geometry term (GW)"), ("same_element", "same-element share")]


def score_target(target, overwrite=False):
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
    path = OUT / f"{target}.csv"
    old = pd.read_csv(path) if path.exists() and not overwrite else pd.DataFrame(columns=["method"])
    rows = []
    for method in METHODS:
        if method in set(old.method):
            continue
        d = run_dir(target, selection_name(method, 100))
        if not d.exists():
            d = run_dir(target, method)                # 20-structure set only
        if not d.exists():
            continue
        sets = pd.read_csv(d / "sets.csv")
        if "sr10m_frame" not in sets:                  # MOF-off neighbours (moffgw): not OMat24 frames
            continue
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
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return f"{target}: {len(rows)} structures scored, {len(df)} in total"


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
            if s.empty or s["rank"].max() < n:                       # no set of this size
                continue
            rows.append(dict(target=t, method=method, n_train=n, **{c: s[c].mean() for c in
                             ["fgw", "feature_term", "geometry_term", "same_element", "metal_to_metal"]},
                             distinct_formulas=s.formula.nunique()))
    return pd.DataFrame(rows)


TERM_COLS = ["fgw", "feature_term", "geometry_term", "same_element", "metal_to_metal", "frac_plus_u"]
GAINS = [("target_force_mae", "force MAE"), ("target_energy_mae_offset_free_meV", "energy MAE (offset-free)"),
         ("target_stress_mae_meV_A3", "stress MAE"), ("target_energy_mae_meV", "energy MAE (raw)")]


def gains():
    """Per (target, method, n_train, lr): % change of each metric after 60 epochs, seeds averaged."""
    from mof_embeddings.tta.effect import changes, per_target, set_features
    pt = per_target(changes().merge(set_features(), on=["target", "method"], how="left"))
    pt["n_train"] = N_TRAIN
    fin = pd.read_csv(DATA / "tta" / "final.csv")
    fin = fin[(fin.n_train != N_TRAIN) & (fin.seed == SEED) & (fin.variant.fillna("") == "")]
    big = fin[["target", "method", "n_train", "lr"]].assign(target_force_mae=fin.force_change_pct)
    return pd.concat([pt, big], ignore_index=True)


def within(df, cols):
    """Subtract each target's mean over its sets (same set size and learning rate)."""
    out = df.copy()
    out[cols] = df[cols] - df.groupby(["target", "n_train", "lr"])[cols].transform("mean")
    return out


def subsets(g):
    nb = g[g.method != "random"]
    return [("all sets", g), ("neighbour sets", nb), ("neighbour sets, targets without +U metal", nb[~nb.target_metal_u.fillna(False).astype(bool)])]


def relate():
    from scipy.stats import spearmanr
    sets = set_means()
    sets.to_csv(OUT / "sets.csv", index=False)
    df = gains().merge(sets, on=["target", "method", "n_train"])
    rows = []
    for (n, lr), g in df.groupby(["n_train", "lr"]):
        for subset, h in subsets(g):
            for how, x in [("across sets", h), ("within target", within(h, TERM_COLS + [m for m, _ in GAINS]))]:
                for gain, _ in GAINS:
                    for col in TERM_COLS:
                        ok = x[[col, gain]].dropna()
                        if len(ok) < 6 or ok[col].nunique() < 3:
                            continue
                        rho, pv = spearmanr(ok[col], ok[gain])
                        rows.append(dict(n_train=n, lr=lr, subset=subset, comparison=how, metric=gain, term=col,
                                         n=len(ok), targets=h.target.nunique(), spearman=rho, p=pv))
    cor = pd.DataFrame(rows)
    cor.to_csv(OUT / "correlations.csv", index=False)
    plot(df, [(N_TRAIN, 3e-5, m, lab) for m, lab in GAINS[:3]], "terms_vs_gain.png")
    plot(df, [(N_TRAIN, 1e-4, GAINS[0][0], GAINS[0][1]), (N_TRAIN, 1e-5, GAINS[0][0], GAINS[0][1]),
              (100, 1e-4, GAINS[0][0], GAINS[0][1])], "terms_vs_gain_force.png")
    return df, cor


def plot(df, panels, fname):
    """Rows = (set size, lr, metric); columns = terms. Points are within-target deviations, so each
    target's own level is removed; titles give Spearman rho within target and across sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    from mof_embeddings.tta.compare import COL, LAB
    fig, axes = plt.subplots(len(panels), len(TERMS), figsize=(3.3 * len(TERMS), 2.9 * len(panels) + 0.4),
                             facecolor=SURFACE, squeeze=False)
    for r, (n, lr, gain, glab) in enumerate(panels):
        g = df[(df.n_train == n) & np.isclose(df.lr, lr)].dropna(subset=[gain])
        w = within(g, [c for c, _ in TERMS] + [gain])
        for c, (col, lab) in enumerate(TERMS):
            ax = axes[r, c]
            for m in METHODS:
                h = w[w.method == m]
                ax.scatter(h[col], h[gain], s=20, color=COL[m], label=LAB[m], zorder=3, edgecolor="white", lw=0.5)
            ax.axhline(0, color=INK, lw=0.8)
            nb = w[w.method != "random"]
            rho_w = spearmanr(w[col], w[gain]).statistic
            rho_nb = spearmanr(nb[col], nb[gain]).statistic
            rho_a = spearmanr(g[col], g[gain]).statistic
            ax.set_title(f"within target ρ {rho_w:+.2f}, neighbours {rho_nb:+.2f}\nacross sets ρ {rho_a:+.2f}",
                         fontsize=7.5, color=INK)
            if r == len(panels) - 1:
                ax.set_xlabel(f"{lab}\n(set mean minus target mean)", fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel(f"{n} structures, lr {lr:g}, {g.target.nunique()} targets\n"
                              f"{glab} change (points vs target mean)", fontsize=7.5, color=INK2)
            ax.tick_params(labelsize=7, colors=INK2); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
            for s_ in ["top", "right"]:
                ax.spines[s_].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(METHODS), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(OUT / fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    from functools import partial
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["terms", "relate"])
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="rescore methods already in the file")
    args = parser.parse_args()
    if args.stage == "terms":
        score = partial(score_target, overwrite=args.overwrite)
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                for msg in pool.imap_unordered(score, args.targets):
                    print(msg, flush=True)
        else:
            for t in args.targets:
                print(score(t), flush=True)
    else:
        df, cor = relate()
        pd.set_option("display.width", 220)
        for how in ["within target", "across sets"]:
            c = cor[(cor.comparison == how) & (cor.subset == "neighbour sets") & (cor.n_train == N_TRAIN)]
            print(f"\n== {how}, neighbour sets, {N_TRAIN} structures: Spearman rho with % change")
            print(c.pivot_table(index=["lr", "metric"], columns="term", values="spearman").round(2))


if __name__ == "__main__":
    main()
