#!/usr/bin/env python
"""Check that the structure-level FGW distance (structure_fgw.py) is not fooled by
cell size: for each of the 94 materials (10 curated + 84 held-out), the distance
from its small cell to its own 2x/4x supercell should be the smallest among all
94 supercells. Compares kernel widths, alpha, and the intensive (N-scaled) kernel
against the plain periodic kernel. Writes the chosen global constants for reuse.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from mof_embeddings.config import DATA
from mof_embeddings.matching.fgw_atom_mapping import atom_features
from mof_embeddings.matching.ot_atom_mapping import BANKS, bank_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel

DATA = Path(str(DATA))
SETS = ["similar_structures", "similar_structures_heldout"]


def load_pairs(stats):
    small, large = [], []
    for s in SETS:
        frames = pd.read_csv(DATA / s / "frames.csv")
        structures = read(DATA / s / "structures.extxyz", ":")
        emb = np.load(DATA / s / "embeddings.npz")
        off = emb["frame_offsets"]
        for k in range(0, len(frames), 2):
            for dest, j in [(small, k), (large, k + 1)]:
                dest.append(dict(structure=AseAtomsAdaptor.get_structure(structures[j]),
                                 features=atom_features("UMA+MH1", emb, stats, off[j], off[j + 1]),
                                 formula=frames.formula[k]))
    return small, large


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA / "structure_fgw/validation")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {m: bank_stats(p) for m, p in BANKS.items()}
    small, large = load_pairs(stats)
    rng = np.random.default_rng(0)
    pooled = np.concatenate([d["features"] for d in small + large])
    i, j = rng.integers(0, len(pooled), (2, 50_000))
    m0 = float(np.median(np.sum((pooled[i] - pooled[j]) ** 2, 1)))

    rows, constants = [], dict(m0=m0, k0={})
    for sigma in [1.0, 2.0, 3.0]:
        kern = {id(d): periodic_kernel(d["structure"], sigma) for d in small + large}
        k0 = float(np.median(np.concatenate([kern[id(d)].sum(1) for d in small + large])))
        constants["k0"][str(sigma)] = k0
        for scaling in ["intensive", "plain"]:
            def K(d):
                C = kern[id(d)]
                return (len(C) * C if scaling == "intensive" else C) / (k0 if scaling == "intensive" else C.max())
            Ks, Kl = [K(d) for d in small], [K(d) for d in large]
            for alpha in [0.0, 0.25, 0.5, 0.75]:
                if alpha == 0 and (sigma != 1.0 or scaling != "intensive"):
                    continue                                          # plain OT does not use the kernel
                D = np.array([[fgw_distance(a["features"], Ka, b["features"], Kb, alpha, m0)
                               for b, Kb in zip(large, Kl)] for a, Ka in zip(small, Ks)])
                rank = (D < np.diag(D)[:, None]).sum(1) + 1
                rows.append(dict(sigma=sigma if alpha else np.nan, kernel=scaling if alpha else "-", alpha=alpha,
                                 top1=np.mean(rank == 1), top5=np.mean(rank <= 5), median_rank=np.median(rank),
                                 true_over_median=np.median(np.diag(D) / np.median(D, 1))))
                np.save(args.out / f"D_sigma{sigma}_{scaling}_alpha{alpha}.npy", D)
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "retrieval.csv", index=False)
    (args.out / "constants.json").write_text(json.dumps(constants, indent=2))
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
