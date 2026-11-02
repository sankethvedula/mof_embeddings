#!/usr/bin/env python
"""Optimal-transport matching between the atom-wise embeddings of a small cell
(N atoms) and its 2x supercell (2N atoms), scored against the true
crystallographic correspondence.

Ground truth: pymatgen StructureMatcher aligns the small cell onto the supercell
(get_s2_like_s1). Tagging each small-cell atom with its index gives, for every
supercell atom j, the small-cell atom it is a periodic copy of.

OT: uniform weights (1/N and 1/2N) and squared Euclidean cost on z-scored
features. With integer ratio k = 2N/N, the exact plan is an assignment of k
copies of each small atom to the supercell atoms (each small atom receives k).

Each OT match (small atom i -> supercell atom j) is scored as:
  same_element  i and j are the same element
  same_orbit    i and j's true partner are symmetry-equivalent in the small cell
                (NaN when the parent space group cannot be recovered from the rattled cell)
  exact         i is j's true partner
  error_A       distance from j to the nearest aligned copy of i (0 ~ correct site,
                the rattling displacement included); compared with a baseline
                that matches each atom to a random same-element partner.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mof_embeddings.config import ARCHIVE, DATA

A = Path(str(ARCHIVE))
BANKS = {"UMA": A / "embeddings/uma/atom/omat24_sr1m/atom_latents.npy",
         "MH1": A / "embeddings/mace_mh1/atom/omat24_sr1m/readout96/atom_latents.npy"}
EMB_KEY = {"UMA": "atom_uma128", "MH1": "atom_mh1_96"}


def bank_stats(path, n=200_000, seed=0):
    """Per-channel mean/std from a random sample of the full sr_1m atom bank."""
    bank = np.load(path, mmap_mode="r")
    rows = np.sort(np.random.default_rng(seed).choice(len(bank), n, replace=False))
    x = np.asarray(bank[rows], dtype=np.float64)
    return x.mean(0), x.std(0) + 1e-8


def orbits(structure, spg_target):
    """Symmetry-equivalent atoms of the (slightly rattled) small cell, accepted only
    at a tolerance that recovers the parent's space group; otherwise None."""
    for symprec in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        try:
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            eq, spg = np.asarray(sga.get_symmetry_dataset().equivalent_atoms), sga.get_space_group_number()
        except Exception:
            continue
        if spg == spg_target:
            return eq, spg, symprec
    return None, None, None


def ot_plan(a, b):
    """Exact uniform OT for len(b) = k * len(a): returns, for every row of b, its matched row of a."""
    k = len(b) // len(a)
    cost = cdist(np.repeat(a, k, 0), b, "sqeuclidean")
    r, c = linear_sum_assignment(cost)
    match = np.empty(len(b), int)
    match[c] = r // k
    return match, cost[r, c].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(f"{DATA}/similar_structures"))
    args = parser.parse_args()
    out = args.data_dir / "ot_mapping"
    out.mkdir(exist_ok=True)

    frames = pd.read_csv(args.data_dir / "frames.csv")
    structures = read(args.data_dir / "structures.extxyz", ":")
    emb = np.load(args.data_dir / "embeddings.npz")
    off = emb["frame_offsets"]
    stats = {m: bank_stats(p) for m, p in BANKS.items()}
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8, attempt_supercell=True, primitive_cell=False)
    rng = np.random.default_rng(0)

    rows, geometry = [], {}
    for k in range(0, len(frames), 2):
        small_f, large_f = frames.iloc[k], frames.iloc[k + 1]
        assert small_f.material == large_f.material and small_f.family == "relax" and large_f.family == "r300"
        small = AseAtomsAdaptor.get_structure(structures[k])
        large = AseAtomsAdaptor.get_structure(structures[k + 1])
        small.add_site_property("orig", list(range(len(small))))
        aligned = matcher.get_s2_like_s1(large, small)          # small cell, tiled and aligned onto `large`
        truth = np.array([s.properties["orig"] for s in aligned])
        assert all(a.specie == b.specie for a, b in zip(aligned, large))
        eq, spg, symprec = orbits(small, int(small_f.parent_id.split("spg")[-1]))
        zs = np.array([s.specie.Z for s in small])
        orbit_known = eq is not None
        if not orbit_known:
            eq = np.arange(len(small))

        # distance from each supercell atom j to the nearest aligned copy of each small atom i
        dist = large.lattice.get_all_distances(large.frac_coords, aligned.frac_coords)   # (2N, 2N)
        to_small = np.stack([dist[:, truth == i].min(1) for i in range(len(small))], 1)  # (2N, N)

        geometry[small_f.material] = dict(
            formula=small_f.formula, spg=spg, symprec=symprec, truth=truth.tolist(),
            orbit=eq.tolist() if orbit_known else None,
            large_frac=large.frac_coords.tolist(), aligned_frac=aligned.frac_coords.tolist(),  # both in the supercell basis
            lattice=large.lattice.matrix.tolist(), large_species=[str(s.specie) for s in large],
            small_species=[str(s.specie) for s in small])

        for model in BANKS:
            mu, sd = stats[model]
            a = (emb[EMB_KEY[model]][off[k]:off[k + 1]] - mu) / sd
            b = (emb[EMB_KEY[model]][off[k + 1]:off[k + 2]] - mu) / sd
            match, cost = ot_plan(a, b)
            geometry[small_f.material][f"ot_{model}"] = match.tolist()
            for j in range(len(large)):
                i, t = match[j], truth[j]
                same_el = np.flatnonzero(zs == zs[t])
                orbit_hit = float(eq[i] == eq[t]) if orbit_known else np.nan
                random_orbit = float(np.mean(eq[same_el] == eq[t])) if orbit_known else np.nan
                rows.append(dict(material=small_f.material, formula=small_f.formula, model=model, atom=j,
                                 element=str(large[j].specie), true_site=t, ot_site=i,
                                 same_element=zs[i] == zs[t], same_orbit=orbit_hit, exact=i == t,
                                 random_same_orbit=random_orbit, random_exact=1 / len(same_el),
                                 error_A=to_small[j, i], rattle_A=to_small[j, t],
                                 random_same_element_error_A=to_small[j, rng.choice(same_el)],
                                 n_orbits_of_element=len(np.unique(eq[same_el])) if orbit_known else np.nan,
                                 ot_cost=cost))

    df = pd.DataFrame(rows)
    df.to_csv(out / "atom_matches.csv", index=False)
    (out / "geometry.json").write_text(json.dumps(geometry))
    per = df.groupby(["formula", "model"], sort=False).agg(
        same_element=("same_element", "mean"), same_orbit=("same_orbit", "mean"), exact=("exact", "mean"),
        median_error_A=("error_A", "median"), median_rattle_A=("rattle_A", "median")).unstack("model")
    tot = df.groupby("model").agg(same_element=("same_element", "mean"), same_orbit=("same_orbit", "mean"),
                                  random_same_orbit=("random_same_orbit", "mean"),
                                  exact=("exact", "mean"), random_exact=("random_exact", "mean"),
                                  median_error_A=("error_A", "median"),
                                  median_rattle_A=("rattle_A", "median"),
                                  random_same_element_error_A=("random_same_element_error_A", "median"))
    per.to_csv(out / "per_material.csv")
    tot.to_csv(out / "summary.csv")
    with pd.option_context("display.width", 160):
        print(per.round(2).to_string()); print(); print(tot.round(3).to_string())


if __name__ == "__main__":
    main()
