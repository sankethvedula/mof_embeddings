#!/usr/bin/env python
"""Balanced samples for grouped FGW distance matrices: MOF-off plus an equal number
of structures from each OMat24 subset family of sr_10m (UMA atom embeddings).

  random   --n random structures per family
  organic  --n structures per family that contain both C and H (chemistry held equal)
Both use the same MOF-off structures (taken from data/structure_fgw/sample). Frames
with atoms closer than 0.8 A are skipped. Output layout matches build_fgw_sample.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from ase.io import read, write

from mof_embeddings.config import ARCHIVE, DATA
from mof_embeddings.mof_search.compare_subset_terms import FAMILY, ORDER
from mof_embeddings.mof_search.common import SAMPLE, SR10M, SR10M_LMDB
from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"
OUT = Path(f"{DATA}/grouped_fgw")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    names = json.load(open(L + ".json"))["names"]
    sub = np.array(names)[np.load(L + ".npy")]
    family = np.array([FAMILY[s] for s in sub])
    comp = pd.read_parquet(f"{DATA}/overlap/omat_sr10m_compositions.parquet", columns=["has_C", "has_H"])
    organic = (comp.has_C & comp.has_H).to_numpy()
    off = np.load(SR10M / "frame_atom_offsets.npy")
    uma = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    z = np.load(SR10M / "atom_atomic_numbers.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)

    mof_meta = pd.read_csv(SAMPLE / "structures.csv")
    mof_emb = np.load(SAMPLE / "embeddings.npz")
    mof_atoms = read(SAMPLE / "structures.extxyz", ":")
    mof_idx = np.sort(rng.choice(np.flatnonzero(mof_meta.dataset == "MOF-off"), args.n, replace=False))

    for kind in ["random", "organic"]:
        rows, atoms, feats, zs = [], [], [], []
        o = mof_emb["frame_offsets"]
        for i in mof_idx:
            rows.append(dict(dataset="MOF-off", subdataset="MOF-off R2SCAN", frame=int(mof_meta.source_frame[i]),
                             natoms=int(mof_meta.natoms[i]), formula=mof_meta.formula[i]))
            atoms.append(mof_atoms[i]); feats.append(mof_emb["atom_uma128"][o[i]:o[i + 1]])
            zs.append(mof_emb["atomic_numbers"][o[i]:o[i + 1]])
        for fam in ORDER:
            pool = np.flatnonzero((family == fam) & (organic if kind == "organic" else True))
            taken = 0
            for f in rng.permutation(pool):
                at = db.get_atoms(id=int(f) + 1)
                _, min_d = periodic_kernel_fast(np.array(at.cell), at.positions, 1.0)
                if min_d < 0.8 or len(at) < 2:
                    continue
                zz = np.asarray(z[off[f]:off[f + 1]])
                assert np.array_equal(zz, at.numbers)
                rows.append(dict(dataset="OMat24", subdataset=sub[f], frame=int(f), natoms=len(at),
                                 formula=at.get_chemical_formula(mode="hill", empirical=True)))
                atoms.append(at); feats.append(np.asarray(uma[off[f]:off[f + 1]])); zs.append(zz)
                taken += 1
                if taken == args.n:
                    break
        d = OUT / kind / "sample"
        d.mkdir(parents=True, exist_ok=True)
        meta = pd.DataFrame(rows)
        meta.to_csv(d / "structures.csv", index=False)
        write(d / "structures.extxyz", atoms)
        np.savez(d / "embeddings.npz", frame_offsets=np.concatenate([[0], np.cumsum(meta.natoms)]),
                 atom_uma128=np.concatenate(feats), atomic_numbers=np.concatenate(zs))
        print(kind, meta.subdataset.map(dict(FAMILY, **{"MOF-off R2SCAN": "MOF-off"})).value_counts().to_dict())


if __name__ == "__main__":
    main()
