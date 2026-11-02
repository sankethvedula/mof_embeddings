#!/usr/bin/env python
"""Pure Gromov-Wasserstein matching where the pairwise term comes from the atom
embeddings: no pointwise term and no geometry.

  C1[a,b] = |f_a - f_b|  Euclidean embedding distance between atoms a, b of the
                         small cell tiled to supercell size (copies are identical)
  C2[x,y] = |g_x - g_y|  the same inside the supercell
  GW minimises sum (C1[a,b] - C2[x,y])^2 T[a,x] T[b,y]

So atoms are matched only by how they relate to the other atoms of their own
structure in embedding space. Nothing compares an atom with an atom of the other
structure directly, which means:
  - element identity is not enforced: random restarts may pair any elements;
  - the two sides can use different models ("UMA>MH1": small cell in UMA,
    supercell in MACE-MH1), since only within-structure distances are compared.
Same-space matrices share one normalising constant; cross-model ones are each
scaled by their own maximum. Each fit is restarted from the uniform plan and
--restarts random permutations; the lowest GW objective is kept. Scoring matches
fgw_atom_mapping.py (exact / symmetry site, distance, neighbour bonds kept).
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
from scipy.spatial.distance import cdist

from mof_embeddings.matching.fgw_atom_mapping import atom_features, round_plan, supercell_matrix
from mof_embeddings.matching.ot_atom_mapping import BANKS, bank_stats, orbits

# (name, feature set of the small cell, feature set of the supercell)
VARIANTS = [("UMA", "UMA", "UMA"), ("MH1", "MH1", "MH1"), ("UMA+MH1", "UMA+MH1", "UMA+MH1"),
            ("UMA>MH1", "UMA", "MH1"), ("MH1>UMA", "MH1", "UMA")]


def random_plans(n, count, rng):
    """Random permutations of n atoms, with no element constraint."""
    for _ in range(count):
        T = np.zeros((n, n))
        T[np.arange(n), rng.permutation(n)] = 1 / n
        yield T


def gw(C1, C2, restarts, rng):
    p = q = ot.unif(len(C1))
    best = None
    for G0 in [None, *random_plans(len(C1), restarts, rng)]:
        T, log = ot.gromov.gromov_wasserstein(C1, C2, p, q, loss_fun="square_loss", G0=G0, log=True, max_iter=1000)
        if best is None or log["gw_dist"] < best[1]:
            best = (T, log["gw_dist"])
    return best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(f"{DATA}/similar_structures"))
    parser.add_argument("--restarts", type=int, default=100)
    args = parser.parse_args()
    out = args.data_dir / "gw_embedding"
    out.mkdir(exist_ok=True)

    frames = pd.read_csv(args.data_dir / "frames.csv")
    structures = read(args.data_dir / "structures.extxyz", ":")
    emb = np.load(args.data_dir / "embeddings.npz")
    off = emb["frame_offsets"]
    stats = {m: bank_stats(p) for m, p in BANKS.items()}
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8, attempt_supercell=True, primitive_cell=False)

    rows, pair_rows, geometry = [], [], {}
    for k in range(0, len(frames), 2):
        mat, formula = frames.material[k], frames.formula[k]
        small = AseAtomsAdaptor.get_structure(structures[k])
        large = AseAtomsAdaptor.get_structure(structures[k + 1])
        small.add_site_property("orig", list(range(len(small))))
        truth = np.array([s.properties["orig"] for s in matcher.get_s2_like_s1(large, small)])
        aligned = matcher.get_s2_like_s1(large, small)
        eq, _, _ = orbits(small, int(frames.parent_id[k].split("spg")[-1]))
        orbit_known = eq is not None
        eq = eq if orbit_known else np.arange(len(small))
        zs = np.array([s.specie.Z for s in small])
        dist = large.lattice.get_all_distances(large.frac_coords, aligned.frac_coords)
        to_small = np.stack([dist[:, truth == i].min(1) for i in range(len(small))], 1)

        S, _ = supercell_matrix(small.lattice, large.lattice)
        tiled = small.copy()
        tiled.make_supercell(S)
        orig = np.array([s.properties["orig"] for s in tiled])
        nn = np.where(np.eye(len(small), dtype=bool), np.inf, small.distance_matrix).min(1)
        D1, D2, cutoff = tiled.distance_matrix, large.distance_matrix, 1.3 * np.median(nn)
        neighbour_pairs = (D2 < cutoff) & ~np.eye(len(large), dtype=bool)
        geometry[mat] = dict(bonds=np.argwhere(neighbour_pairs).tolist(), bonds_kept={})

        copies = [np.flatnonzero(orig == i) for i in range(len(small))]
        targets = [np.flatnonzero(truth == i) for i in range(len(small))]

        for name, fs_small, fs_large in VARIANTS:
            a = atom_features(fs_small, emb, stats, off[k], off[k + 1])[orig]
            b = atom_features(fs_large, emb, stats, off[k + 1], off[k + 2])
            C1, C2 = cdist(a, a), cdist(b, b)
            if fs_small == fs_large:
                scale = max(C1.max(), C2.max())
                C1, C2 = C1 / scale, C2 / scale
            else:
                C1, C2 = C1 / C1.max(), C2 / C2.max()
            cost = lambda T: ot.gromov.gwloss(*ot.gromov.init_matrix(C1, C2, T.sum(1), T.sum(0), "square_loss"), T)
            gw_true = cost(_copy_plan(copies, targets, len(large)))   # copies share embeddings: any assignment

            plan = gw(C1, C2, args.restarts, np.random.default_rng(0))
            tiled_of = round_plan(plan, 1)
            match = orig[tiled_of]
            d1 = D1[tiled_of[:, None], tiled_of[None, :]][neighbour_pairs]
            geometry[mat]["bonds_kept"][f"GW_{name}"] = (d1 < cutoff).tolist()
            geometry[mat][f"gw_{name}"] = match.tolist()
            pair_rows.append(dict(material=mat, formula=formula, model=name, neighbours_kept=(d1 < cutoff).mean(),
                                  distortion_A=np.median(np.abs(d1 - D2[neighbour_pairs])),
                                  gw_cost=cost(plan), gw_cost_true=gw_true))
            for j in range(len(large)):
                i, t = match[j], truth[j]
                rows.append(dict(material=mat, formula=formula, model=name, atom=j, element=str(large[j].specie),
                                 true_site=t, ot_site=i, same_element=zs[i] == zs[t], exact=i == t,
                                 same_orbit=float(eq[i] == eq[t]) if orbit_known else np.nan, error_A=to_small[j, i]))
        print(f"  {formula}", flush=True)

    df, pairs = pd.DataFrame(rows), pd.DataFrame(pair_rows)
    df.to_csv(out / "atom_matches.csv", index=False)
    pairs.to_csv(out / "neighbour_scores.csv", index=False)
    (out / "geometry.json").write_text(json.dumps(geometry))
    summary = df.groupby("model", sort=False).agg(
        same_element=("same_element", "mean"), exact=("exact", "mean"), same_orbit=("same_orbit", "mean"),
        median_error_A=("error_A", "median"), within_1A=("error_A", lambda e: (e < 1.0).mean()))
    summary = summary.join(pairs.groupby("model", sort=False)[["neighbours_kept", "distortion_A"]].mean())
    summary["embedding_GW_prefers_truth"] = (pairs.gw_cost_true <= pairs.gw_cost + 1e-9).groupby(pairs.model, sort=False).mean()
    summary.to_csv(out / "summary.csv")
    per = df.groupby(["formula", "model"], sort=False)[["same_element", "exact"]].mean().unstack("model")
    per.to_csv(out / "per_material.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string()); print(); print(per.round(2).to_string())


def _copy_plan(copies, targets, n):
    T = np.zeros((n, n))
    for c, t in zip(copies, targets):
        T[list(c), t] = 1 / n
    return T


if __name__ == "__main__":
    main()
