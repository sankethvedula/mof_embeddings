#!/usr/bin/env python
"""Per-atom attributes for the joint UMAPs, aligned with each dataset's atom bank
(the same rows joint_atom_umap.py samples): DFT force magnitude |F| (eV/A),
atoms in the structure, and the structure's volume per atom (A^3; NaN for
non-periodic frames such as MAD's molecules and clusters). Every frame's
atomic numbers are checked against the bank's.

Writes attributes_<dataset>.npz to --out-dir.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from mof_embeddings.config import PRETRAIN, SCRATCH
from mof_embeddings.embeddings.joint_atom_umap import BANK, DATASETS

P = Path(f"{PRETRAIN}/data/processed")
MAD_SPLITS = [("train", 76483), ("val", 9566), ("test", 9546)]
_src = {}


def open_source(name):
    from ase.db import connect
    from ase.io import Trajectory
    if name == "OMat24":
        db = connect(str(P / "omat24/stratified_random_nested/sr_1m/train.lmdb"), type="aselmdb", readonly=True)
        return lambda f: db.get_atoms(id=int(f) + 1)
    if name == "MatPES":
        db = connect(str(P / "matpes/all.lmdb"), type="aselmdb", readonly=True)
        return lambda f: db.get_atoms(id=int(f) + 1)
    if name == "MOF-off":
        tr = Trajectory(str(P / "mof_off/R2SCAN/R2SCAN_train.traj"))
        return lambda f: tr[int(f)]
    if name == "MP-ALOE":
        tr = Trajectory(str(P / "mpaloe/mpaloe_all.traj"))
        return lambda f: tr[int(f)]
    if name == "MAD":
        trs = [Trajectory(str(P / f"mad/{s}/mad_{s}.traj")) for s, _ in MAD_SPLITS]
        starts = np.cumsum([0] + [n for _, n in MAD_SPLITS])
        def get(f):
            k = int(np.searchsorted(starts, f, side="right") - 1)
            return trs[k][int(f - starts[k])]
        return get


def work(job):
    name, frames, offsets, zbank = job
    if name not in _src:
        _src[name] = open_source(name)
    get = _src[name]
    n = offsets[-1] - offsets[0]
    force = np.full(n, np.nan, np.float32); natoms = np.zeros(n, np.int32); vpa = np.zeros(n, np.float32)
    for f, a, b in zip(frames, offsets[:-1], offsets[1:]):
        if a == b:                                             # frame has no rows in the bank (e.g. MAD's isolated atoms)
            continue
        at = get(f)
        s = slice(a - offsets[0], b - offsets[0])
        assert np.array_equal(at.numbers, zbank[s]), f"atom order differs: {name} frame {f}"
        force[s] = np.linalg.norm(at.get_forces(), axis=1)
        natoms[s] = len(at); vpa[s] = at.get_volume() / len(at) if at.cell.rank == 3 else np.nan
    return force, natoms, vpa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(f"{SCRATCH}/joint_umap"))
    parser.add_argument("--procs", type=int, default=16)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, (name, d) in enumerate(DATASETS):
        bank = BANK["UMA"](d)
        off = np.load(bank / "frame_atom_offsets.npy")
        frames = np.arange(len(off) - 1) if k == 0 else np.load(bank / "frame_source_index.npy")
        z = np.load(bank / "atom_atomic_numbers.npy")
        chunks = np.array_split(np.arange(len(frames)), max(1, len(frames) // 2000))
        jobs = [(name, frames[c], off[c[0]:c[-1] + 2], z[off[c[0]]:off[c[-1] + 1]]) for c in chunks if len(c)]
        with Pool(args.procs) as pool:
            res = pool.map(work, jobs, chunksize=1)
        force, natoms, vpa = (np.concatenate(x) for x in zip(*res))
        np.savez(args.out_dir / f"attributes_{name}.npz", force=force, natoms=natoms, volume_per_atom=vpa)
        print(f"{name}: {len(force):,} atoms; median |F| {np.nanmedian(force):.3f} eV/A, median atoms {np.median(natoms):.0f}, "
              f"median volume/atom {np.nanmedian(vpa):.1f} A^3 ({np.isnan(vpa).mean():.1%} non-periodic)", flush=True)


if __name__ == "__main__":
    main()
