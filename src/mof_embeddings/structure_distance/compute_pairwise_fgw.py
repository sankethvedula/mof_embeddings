#!/usr/bin/env python
"""Pairwise structure-FGW distances (structure_fgw.py) over a sample built by
build_fgw_sample.py. Run as a Slurm array: task t handles pairs p with
p % n_tasks == t, so every task gets a similar mix of large and small structures.
`--merge` assembles the task outputs into full symmetric matrices, one per alpha.

Global constants (shared by all tasks, computed deterministically from the sample):
  m0  median squared feature distance between random atom pairs of the sample
  k0  median per-atom periodic-kernel sum over the sample
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from mof_embeddings.config import DATA
from mof_embeddings.matching.fgw_atom_mapping import atom_features
from mof_embeddings.matching.ot_atom_mapping import BANKS, bank_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel


def setup(sample, sigma, features="UMA+MH1", constants=None):
    """constants: optional {"m0", "k0"} to use instead of values derived from this sample."""
    stats = {m: bank_stats(p) for m, p in BANKS.items() if m in features.split("+")}
    emb = np.load(sample / "embeddings.npz")
    off = emb["frame_offsets"]
    feats = [atom_features(features, emb, stats, off[i], off[i + 1]) for i in range(len(off) - 1)]
    structures = [AseAtomsAdaptor.get_structure(a) for a in read(sample / "structures.extxyz", ":")]
    kernels = [periodic_kernel(s, sigma) for s in structures]
    pooled = np.concatenate(feats)
    i, j = np.random.default_rng(0).integers(0, len(pooled), (2, 100_000))
    m0 = float(np.median(np.sum((pooled[i] - pooled[j]) ** 2, 1)))
    k0 = float(np.median(np.concatenate([C.sum(1) for C in kernels])))
    if constants:
        m0, k0 = constants["m0"], constants["k0"]
    K = [len(C) * C / k0 for C in kernels]                     # intensive kernel
    return feats, K, dict(m0=m0, k0=k0, sigma=sigma, features=features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path(f"{DATA}/structure_fgw/sample"))
    parser.add_argument("--out", type=Path, default=Path(f"{DATA}/structure_fgw/pairwise"))
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--features", default="UMA+MH1", choices=["UMA", "MH1", "UMA+MH1"])
    parser.add_argument("--constants", type=Path, help="JSON with m0, k0 to use instead of sample-derived values")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = len(np.load(args.sample / "embeddings.npz")["frame_offsets"]) - 1
    iu, ju = np.triu_indices(n, 1)

    if args.merge:
        parts = sorted(args.out.glob("part_*.npz"))
        assert len(parts) == args.n_tasks, f"found {len(parts)} of {args.n_tasks} parts"
        for alpha in args.alphas:
            D = np.zeros((n, n))
            for p in parts:
                z = np.load(p)
                D[iu[z["pairs"]], ju[z["pairs"]]] = z[f"alpha_{alpha}"]
            np.save(args.out / f"D_alpha{alpha}.npy", D + D.T)
        print("merged", [f"D_alpha{a}.npy" for a in args.alphas])
        return

    fixed = json.loads(args.constants.read_text()) if args.constants else None
    feats, K, constants = setup(args.sample, args.sigma, args.features, fixed)
    if args.task == 0:
        (args.out / "constants.json").write_text(json.dumps(dict(constants, alphas=args.alphas), indent=2))
    pairs = np.arange(args.task, len(iu), args.n_tasks)
    out = {f"alpha_{a}": np.empty(len(pairs)) for a in args.alphas}
    for k, p in enumerate(pairs):
        a, b = iu[p], ju[p]
        for alpha in args.alphas:
            out[f"alpha_{alpha}"][k] = fgw_distance(feats[a], K[a], feats[b], K[b], alpha, constants["m0"])
        if k % 5000 == 0:
            print(f"task {args.task}: {k}/{len(pairs)}", flush=True)
    np.savez(args.out / f"part_{args.task:03d}.npz", pairs=pairs, **out)


if __name__ == "__main__":
    main()
