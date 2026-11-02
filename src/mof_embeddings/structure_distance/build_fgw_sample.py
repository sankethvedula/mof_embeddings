#!/usr/bin/env python
"""Sample MOF-off and OMat24 structures with atom-level UMA + MH1 embeddings for a
pairwise structure-FGW distance matrix.

  MOF-off   R2SCAN-D4 10% atom sample (only MOF-off set with atom embeddings);
            frames MACE could not handle are skipped; geometry from R2SCAN_train.traj
  OMat24    sr_1m atom banks; structures with at least 2 atoms; geometry from sr_1m train.lmdb

Writes structures.extxyz, embeddings.npz (frame_offsets, atom_uma128, atom_mh1_96,
atomic_numbers; same layout as data/similar_structures) and structures.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from ase.io import Trajectory, write

from mof_embeddings.config import ARCHIVE, DATA, PRETRAIN

A = Path(str(ARCHIVE))
P = Path(f"{PRETRAIN}/data/processed")
SOURCES = {
    "MOF-off": dict(uma=A / "embeddings/uma/atom/external_subsets/mof_off_r2scan",
                    mh1=A / "embeddings/mace_mh1/atom/external_subsets/mof_off_r2scan/readout96"),
    "OMat24": dict(uma=A / "embeddings/uma/atom/omat24_sr1m",
                   mh1=A / "embeddings/mace_mh1/atom/omat24_sr1m/readout96"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mof", type=int, default=100)
    parser.add_argument("--n-omat", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(f"{DATA}/structure_fgw/sample"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mof_traj = Trajectory(P / "mof_off/R2SCAN/R2SCAN_train.traj")
    omat_db = connect(P / "omat24/stratified_random_nested/sr_1m/train.lmdb", type="aselmdb", readonly=True)
    omat_sub = pd.read_parquet(A / "derived/atomic_rbf_sr1m/identity/frame_provenance.parquet", columns=["subdataset"])

    rows, atoms, uma, mh1, z = [], [], [], [], []
    for name, n in [("MOF-off", args.n_mof), ("OMat24", args.n_omat)]:
        src = SOURCES[name]
        off = np.load(src["uma"] / "frame_atom_offsets.npy")
        assert np.array_equal(off, np.load(src["mh1"] / "frame_atom_offsets.npy"))
        ok = np.diff(off) >= 2
        unsupported = src["mh1"] / "unsupported_frames.npy"
        if unsupported.exists():
            ok[np.load(unsupported)] = False
        frames = np.sort(rng.choice(np.flatnonzero(ok), n, replace=False))
        # bank frame -> frame in the source file (sr_1m bank frames are the lmdb records themselves)
        source_index = np.load(src["uma"] / "frame_source_index.npy") if name == "MOF-off" else np.arange(len(off) - 1)
        bu, bm = np.load(src["uma"] / "atom_latents.npy", mmap_mode="r"), np.load(src["mh1"] / "atom_latents.npy", mmap_mode="r")
        bz = np.load(src["uma"] / "atom_atomic_numbers.npy", mmap_mode="r")
        for f in frames:
            s = slice(off[f], off[f + 1])
            src_frame = int(source_index[f])
            at = mof_traj[src_frame] if name == "MOF-off" else omat_db.get_atoms(id=src_frame + 1)
            zz = np.asarray(bz[s])
            assert np.array_equal(zz, at.numbers), f"atom order differs for {name} frame {f}"
            atoms.append(at); uma.append(np.asarray(bu[s])); mh1.append(np.asarray(bm[s])); z.append(zz)
            rows.append(dict(dataset=name, bank_frame=int(f), source_frame=src_frame, natoms=len(at),
                             formula=at.get_chemical_formula(mode="hill", empirical=True),
                             subdataset="MOF-off R2SCAN" if name == "MOF-off" else omat_sub.subdataset[src_frame],
                             volume_per_atom=at.get_volume() / len(at)))
        print(f"{name}: {n} structures", flush=True)

    meta = pd.DataFrame(rows)
    meta.to_csv(args.out_dir / "structures.csv", index=False)
    write(args.out_dir / "structures.extxyz", atoms)
    np.savez(args.out_dir / "embeddings.npz", frame_offsets=np.concatenate([[0], np.cumsum(meta.natoms)]),
             atom_uma128=np.concatenate(uma), atom_mh1_96=np.concatenate(mh1), atomic_numbers=np.concatenate(z))
    print(meta.groupby("dataset").natoms.describe().round(1).to_string())


if __name__ == "__main__":
    main()
