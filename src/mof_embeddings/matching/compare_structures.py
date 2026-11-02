#!/usr/bin/env python
"""Starter: compare structures with different atom counts using UMA / MACE-MH1
atom embeddings, and check which comparison recognises the same material in a
bigger cell.

Methods (each gives a distance between two structures A and B):
  sum      the archive's stored structure vector H = sum_i Z_i h_i  (grows with cell size)
  z-mean   H / sum_i Z_i: the same vector divided by total atomic number
  mean     plain mean of the atom vectors, (1/N) sum_i h_i
  ot       optimal-transport (Wasserstein-2) distance between the two sets of atom
           vectors, each atom weighted 1/N; a supercell of identical atoms gives 0

Test: for each small relaxed cell, rank its own 2x rattled-300 supercell among all
supercell candidates (rank 1 = correct match is the nearest). Features are
z-scored per channel over the frames being compared.
"""
import argparse
from math import lcm
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mof_embeddings.config import ARCHIVE, DATA

A = Path(str(ARCHIVE))
ATOM_BANKS = {"UMA": A / "embeddings/uma/atom/omat24_sr1m/atom_latents.npy",
              "MH1": A / "embeddings/mace_mh1/atom/omat24_sr1m/readout96/atom_latents.npy"}
STRUCT_BANKS = {"UMA": A / "embeddings/uma/structure/omat24_full/all_uma_embeddings.npy",
                "MH1": A / "embeddings/mace_mh1/structure/readout96/omat24_full/all_mace_mh1_readout96.npy"}
Z_BANK = A / "embeddings/uma/atom/omat24_sr1m/atom_atomic_numbers.npy"
OFFSETS = A / "embeddings/uma/atom/omat24_sr1m/frame_atom_offsets.npy"
GLOBAL_ROW = A / "derived/atomic_rbf_sr1m/identity/frame_global_row.npy"


def ot_distance(a, b):
    """Exact W2 between uniform distributions on the rows of a and b (replicate to a common size)."""
    n = lcm(len(a), len(b))
    cost = cdist(np.repeat(a, n // len(a), 0), np.repeat(b, n // len(b), 0), "sqeuclidean")
    r, c = linear_sum_assignment(cost)
    return np.sqrt(cost[r, c].mean())


def ranks(dist):
    """dist[i, j] = distance from small cell i to supercell j; true partner is j == i."""
    return (dist < np.diag(dist)[:, None]).sum(1) + 1


def evaluate(pairs, model, with_ot):
    off, gr = np.load(OFFSETS), np.load(GLOBAL_ROW)
    atoms = np.load(ATOM_BANKS[model], mmap_mode="r")
    zbank, struct = np.load(Z_BANK, mmap_mode="r"), np.load(STRUCT_BANKS[model], mmap_mode="r")
    frames = np.concatenate([pairs.frame_relax.to_numpy(), pairs.frame_r300.to_numpy()])
    sets = [np.asarray(atoms[off[f]:off[f + 1]], dtype=np.float64) for f in frames]
    zsum = np.array([np.asarray(zbank[off[f]:off[f + 1]]).sum() for f in frames], dtype=np.float64)
    mu, sd = np.concatenate(sets).mean(0), np.concatenate(sets).std(0) + 1e-8
    sets = [(s - mu) / sd for s in sets]
    H = np.stack([np.asarray(struct[gr[f]], dtype=np.float64) for f in frames])
    vecs = {"sum": H, "z-mean": H / zsum[:, None], "mean": np.stack([s.mean(0) for s in sets])}
    n = len(pairs)
    out = {}
    for name, v in vecs.items():
        v = (v - v.mean(0)) / (v.std(0) + 1e-8)
        out[name] = cdist(v[:n], v[n:])
    if with_ot:
        out["ot"] = np.array([[ot_distance(sets[i], sets[n + j]) for j in range(n)] for i in range(n)])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(f"{DATA}/similar_structures"))
    parser.add_argument("--ot-max-materials", type=int, default=300,
                        help="OT on the full set is O(n^2) assignments; cap the number of materials")
    args = parser.parse_args()

    curated = pd.read_csv(args.data_dir / "frames.csv")
    curated = curated[curated.family.isin(["relax", "r300"])].pivot(index="material", columns="family", values="frame")
    curated = curated.rename(columns={"relax": "frame_relax", "r300": "frame_r300"})
    full = pd.read_csv(args.data_dir / "pairs_relax_r300.csv", index_col=0)
    sub = full.sample(n=min(args.ot_max_materials, len(full)), random_state=0)

    rows = []
    for label, pairs, with_ot in [(f"curated ({len(curated)} materials, dV<5%)", curated, True),
                                  (f"random {len(sub)} of all pairs", sub, True),
                                  (f"all {len(full)} pairs", full, False)]:
        for model in ATOM_BANKS:
            for method, d in evaluate(pairs, model, with_ot).items():
                r = ranks(d)
                rows.append(dict(set=label, model=model, method=method, top1=np.mean(r == 1),
                                 median_rank=np.median(r), candidates=len(pairs)))
    res = pd.DataFrame(rows)
    res.to_csv(args.data_dir / "retrieval_results.csv", index=False)
    with pd.option_context("display.width", 140):
        print(res.pivot_table(index=["set", "method"], columns="model", values=["top1", "median_rank"],
                              sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
