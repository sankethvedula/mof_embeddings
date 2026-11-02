#!/usr/bin/env python
"""Fused Gromov-Wasserstein (FGW) matching between the atoms of a small cell and
its 2x supercell, compared with plain feature OT on the same 10 pairs.

FGW minimises (1 - alpha) * <M, T> + alpha * sum (C1[a,b] - C2[x,y])^2 T[a,x] T[b,y]
  M   squared Euclidean distance between z-scored atom embeddings (scaled to [0, 1])
  C   neighbour kernel inside each structure, sigma = median nearest-neighbour distance
      of the small cell (same for both cells), with --kernel
        periodic   sum over all lattice images: C[a,b] = sum_L exp(-|r_b + L - r_a|^2 / 2 sigma^2)
        minimage   nearest image only: exp(-d_ab^2 / 2 sigma^2), d = minimum-image distance
      Both C1 and C2 are divided by the same constant (their joint maximum), so the
      geometry term stays on a [0, 1] scale and alpha is comparable across kernels.

The small cell is first tiled to the supercell's size with the integer matrix S that
best maps its lattice onto the supercell lattice (found from the cell vectors alone,
no atom information). Comparing a cell with its own supercell directly is
inconsistent: each neighbour appears once in the small cell but split over k copies
in the supercell, so even the true matching has a high geometry cost.
alpha = 0 is plain feature OT; alpha = 1 uses geometry only. FGW is non-convex, so
each fit is restarted from the uniform plan, the plain-OT plan and --restarts random
element-preserving permutations, keeping the lowest objective. The soft plan is
rounded to a balanced assignment (each small atom gets exactly k = 2 supercell atoms)
and scored as in ot_atom_mapping.py. Results go to fgw_mapping_<kernel>/.
"""
import argparse
import json
from pathlib import Path

from mof_embeddings.config import DATA, use_local_packages
use_local_packages()
import numpy as np
import ot
import pandas as pd
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mof_embeddings.matching.ot_atom_mapping import BANKS, EMB_KEY, bank_stats, orbits

ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
FEATURE_SETS = ["UMA", "MH1", "UMA+MH1"]


def atom_features(feature_set, emb, stats, start, stop):
    """z-scored atom embeddings; "UMA+MH1" concatenates both, each block scaled by
    1/sqrt(width) so the two models contribute equally to squared distances."""
    blocks = []
    for model in feature_set.split("+"):
        mu, sd = stats[model]
        x = (emb[EMB_KEY[model]][start:stop] - mu) / sd
        blocks.append(x / np.sqrt(x.shape[1]) if "+" in feature_set else x)
    return np.hstack(blocks)


_INT = np.array(np.meshgrid(*[np.arange(-2, 3)] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
_DET = np.rint(np.linalg.det(_INT)).astype(int)                       # computed once, reused per pair


def supercell_matrix(small_lattice, large_lattice):
    """Integer S with |det S| = k whose metric S G S^T best matches the large cell's,
    after scaling the small cell to the same volume per formula (tolerates strain)."""
    k = round(large_lattice.volume / small_lattice.volume)
    cand = _INT[_DET == k]
    scale = (large_lattice.volume / (k * small_lattice.volume)) ** (2 / 3)
    G_s, G_l = small_lattice.matrix @ small_lattice.matrix.T * scale, large_lattice.matrix @ large_lattice.matrix.T
    resid = np.linalg.norm(cand @ G_s @ cand.transpose(0, 2, 1) - G_l, axis=(1, 2)) / np.linalg.norm(G_l)
    return cand[np.argmin(resid)], resid.min()


def neighbour_kernel(structure, sigma, kernel="periodic"):
    if kernel == "minimage":
        return np.exp(-structure.distance_matrix ** 2 / (2 * sigma ** 2))
    C = np.eye(len(structure))                                          # the atom itself
    for a, neighbours in enumerate(structure.get_all_neighbors(4 * sigma)):   # exp(-8) beyond this
        for n in neighbours:                                            # every image, own images included
            C[a, n.index] += np.exp(-n.nn_distance ** 2 / (2 * sigma ** 2))
    return (C + C.T) / 2


def round_plan(plan, k):
    """Balanced assignment from a soft plan: each row (small atom) used exactly k times."""
    r, c = linear_sum_assignment(-np.repeat(plan, k, 0))
    match = np.empty(plan.shape[1], int)
    match[c] = r // k
    return match


def random_permutation_plans(z1, z2, n, rng):
    """Random permutations that only pair atoms of the same element (as plans)."""
    for _ in range(n):
        T = np.zeros((len(z1), len(z2)))
        for e in np.unique(z1):
            a, b = np.flatnonzero(z1 == e), rng.permutation(np.flatnonzero(z2 == e))
            T[a, b] = 1 / len(z2)
        yield T


def fgw(M, C1, C2, alpha, z1, z2, restarts, rng):
    p, q = ot.unif(M.shape[0]), ot.unif(M.shape[1])
    if alpha == 0:
        return ot.emd(p, q, M)
    best = None
    for G0 in [None, ot.emd(p, q, M), *random_permutation_plans(z1, z2, restarts, rng)]:
        T, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=alpha,
                                                    G0=G0, log=True, max_iter=1000)
        if best is None or log["fgw_dist"] < best[1]:
            best = (T, log["fgw_dist"])
    return best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(f"{DATA}/similar_structures"))
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--kernel", choices=["periodic", "minimage"], default="periodic")
    args = parser.parse_args()
    out = args.data_dir / f"fgw_mapping_{args.kernel}"
    out.mkdir(exist_ok=True)

    frames = pd.read_csv(args.data_dir / "frames.csv")
    structures = read(args.data_dir / "structures.extxyz", ":")
    emb = np.load(args.data_dir / "embeddings.npz")
    off = emb["frame_offsets"]
    stats = {m: bank_stats(p) for m, p in BANKS.items()}
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8, attempt_supercell=True, primitive_cell=False)
    geometry = json.loads((args.data_dir / "ot_mapping/geometry.json").read_text())

    rows, pair_rows = [], []
    for k in range(0, len(frames), 2):
        mat = frames.material[k]
        small = AseAtomsAdaptor.get_structure(structures[k])
        large = AseAtomsAdaptor.get_structure(structures[k + 1])
        small.add_site_property("orig", list(range(len(small))))
        aligned = matcher.get_s2_like_s1(large, small)
        truth = np.array([s.properties["orig"] for s in aligned])
        eq, _, _ = orbits(small, int(frames.parent_id[k].split("spg")[-1]))
        orbit_known = eq is not None
        eq = eq if orbit_known else np.arange(len(small))
        zs = np.array([s.specie.Z for s in small])
        dist = large.lattice.get_all_distances(large.frac_coords, aligned.frac_coords)
        to_small = np.stack([dist[:, truth == i].min(1) for i in range(len(small))], 1)

        nn = np.where(np.eye(len(small), dtype=bool), np.inf, small.distance_matrix).min(1)
        sigma = np.median(nn)
        S, lattice_resid = supercell_matrix(small.lattice, large.lattice)
        tiled = small.copy()
        tiled.make_supercell(S)
        orig = np.array([s.properties["orig"] for s in tiled])          # tiled atom -> small-cell atom
        z_tiled, z_large = np.array([s.specie.Z for s in tiled]), np.array([s.specie.Z for s in large])
        C1, C2 = neighbour_kernel(tiled, sigma, args.kernel), neighbour_kernel(large, sigma, args.kernel)
        scale = max(C1.max(), C2.max())
        C1, C2 = C1 / scale, C2 / scale
        gw_cost = lambda T: ot.gromov.gwloss(*ot.gromov.init_matrix(C1, C2, T.sum(1), T.sum(0), "square_loss"), T)
        # True correspondence from the alignment: aligned site j is the copy of small atom truth[j] that
        # lies on supercell atom j, so the identity plan between `aligned` and `large` is the truth.
        C_true = neighbour_kernel(aligned, sigma, args.kernel) / scale
        identity = np.eye(len(large)) / len(large)
        gw_true = ot.gromov.gwloss(*ot.gromov.init_matrix(C_true, C2, identity.sum(1), identity.sum(0),
                                                          "square_loss"), identity)

        D1, D2, cutoff = tiled.distance_matrix, large.distance_matrix, 1.3 * sigma
        neighbour_pairs = (D2 < cutoff) & ~np.eye(len(large), dtype=bool)

        def neighbour_scores(tiled_of, key=None, distances=None):
            """tiled_of[j] = tiled atom matched to supercell atom j; key stores per-bond flags for plotting.
            `distances` gives the matched atoms' distance matrix directly (used for the true match)."""
            d1 = (D1[tiled_of[:, None], tiled_of[None, :]] if distances is None else distances)[neighbour_pairs]
            if key:
                geometry[mat].setdefault("bonds_kept", {})[key] = (d1 < cutoff).tolist()
            return (d1 < cutoff).mean(), np.median(np.abs(d1 - D2[neighbour_pairs]))

        geometry[mat]["bonds"] = np.argwhere(neighbour_pairs).tolist()

        rng = np.random.default_rng(1)
        rand = np.mean([neighbour_scores(round_plan(T, 1)) for T in
                        random_permutation_plans(z_tiled, z_large, 200, rng)], axis=0)
        reference = dict(truth=neighbour_scores(None, "truth", aligned.distance_matrix), random_same_element=rand)
        for name, (kept, distortion) in reference.items():
            pair_rows.append(dict(material=mat, formula=frames.formula[k], model=name, alpha=np.nan,
                                  neighbours_kept=kept, distortion_A=distortion))

        for model in FEATURE_SETS:
            a = atom_features(model, emb, stats, off[k], off[k + 1])
            b = atom_features(model, emb, stats, off[k + 1], off[k + 2])
            M = cdist(a[orig], b, "sqeuclidean")
            M /= M.max()
            for alpha in ALPHAS:
                plan = fgw(M, C1, C2, alpha, z_tiled, z_large, args.restarts, np.random.default_rng(0))
                tiled_of = round_plan(plan, 1)
                match = orig[tiled_of]
                kept, distortion = neighbour_scores(tiled_of, f"{model}_{alpha}")
                pair_rows.append(dict(material=mat, formula=frames.formula[k], model=model, alpha=alpha,
                                      neighbours_kept=kept, distortion_A=distortion))
                geometry[mat][f"fgw_{model}_{alpha}"] = match.tolist()
                for j in range(len(large)):
                    i, t = match[j], truth[j]
                    rows.append(dict(material=mat, formula=frames.formula[k], model=model, alpha=alpha, atom=j,
                                     element=str(large[j].specie), true_site=t, ot_site=i,
                                     same_element=zs[i] == zs[t], exact=i == t,
                                     same_orbit=float(eq[i] == eq[t]) if orbit_known else np.nan,
                                     error_A=to_small[j, i], sigma_A=sigma,
                                     gw_cost=gw_cost(plan), gw_cost_true=gw_true))
        print(f"  {frames.formula[k]:12s} sigma {sigma:.2f} A, S={S.tolist()} (lattice residual {lattice_resid:.3f})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "atom_matches.csv", index=False)
    (out / "geometry.json").write_text(json.dumps(geometry))
    summary = df.groupby(["model", "alpha"]).agg(
        same_element=("same_element", "mean"), exact=("exact", "mean"), same_orbit=("same_orbit", "mean"),
        median_error_A=("error_A", "median"), within_1A=("error_A", lambda e: (e < 1.0).mean()),
        geometry_prefers_truth=("gw_cost", lambda c: np.nan))
    pref = df.groupby(["model", "alpha", "material"]).first()
    summary["geometry_prefers_truth"] = (pref.gw_cost_true <= pref.gw_cost + 1e-9).groupby(["model", "alpha"]).mean()
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "neighbour_scores.csv", index=False)
    nb = pairs.groupby(["model", "alpha"], dropna=False)[["neighbours_kept", "distortion_A"]].mean()
    summary = summary.join(nb[nb.index.get_level_values("alpha").notna()])
    summary.to_csv(out / "summary.csv")
    per = df.groupby(["formula", "model", "alpha"], sort=False).exact.mean().unstack(["model", "alpha"])
    per.to_csv(out / "per_material_exact.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string()); print(); print(per.round(2).to_string())
        print("\nreferences:\n" + nb[nb.index.get_level_values("alpha").isna()].round(3).to_string())


if __name__ == "__main__":
    main()
