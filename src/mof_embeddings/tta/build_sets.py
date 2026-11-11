#!/usr/bin/env python
"""Training, validation and target databases for test-time adaptation.

For each target MOF (a query of the 10M search):
  target.lmdb   every MOF-off PBE frame of the target framework (the query frame and the
                other AIMD frames, 300-1000 K), PBE labels; target.csv marks the query frame
  fgw/          OMat24 sr_10m frames ranked 1-20 by structure FGW to the query (train) and
                21-25 (valid), from the search's rerank results
  random/       25 random sr_10m frames (same split), a control
  global/       frames ranked 1-25 by the Z-weighted-mean global vector (global_vs_fgw), a control
Frames flagged as unphysical (atoms < 0.8 A apart) are never used. Labels are copied as
stored: OMat24 PBE(+U) total energy, forces and stress; MOF-off PBE (D3BJ dataset) total
energy, forces and stress. Also writes the foundation model's E0s to $SCRATCH/tta.

Run in the MACE environment (uses mace's LMDB writer):
  $PY_MACE -m mof_embeddings.tta.build_sets [--targets Q6 Q3 Q4 Q1] [--overwrite]
"""
import argparse
import glob
import json
import zlib

import numpy as np
import pandas as pd

from mof_embeddings.config import SCRATCH
from mof_embeddings.tta.common import (E0S, FOUNDATION, MOF_OFF_PBE, MOF_OFF_R2SCAN, N_TRAIN, N_VALID, SAMPLE,
                                       SEARCH, SELECTIONS, SR10M_LMDB, TARGETS, run_dir, target_dir)


def write_lmdb(path, atoms_list, overwrite):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.stress import full_3x3_to_voigt_6_stress
    from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to rebuild")
        path.unlink()
        lock = path.with_name(path.name + "-lock")
        if lock.exists():
            lock.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with LMDBDatabase(path) as db:
        for a in atoms_list:
            r = a.calc.results
            stress = np.asarray(r["stress"], dtype=float)
            stress = full_3x3_to_voigt_6_stress(stress) if stress.shape == (3, 3) else stress
            forces = np.asarray(r["forces"], dtype=float)
            assert forces.shape == (len(a), 3) and stress.shape == (6,) and np.isfinite(r["energy"])
            b = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
            b.calc = SinglePointCalculator(b, energy=float(r["energy"]), forces=forces, stress=stress)
            db.write(b)


def foundation_e0s():
    import torch
    m = torch.load(FOUNDATION, map_location="cpu", weights_only=False)
    z = m.atomic_numbers.cpu().numpy().tolist()
    e = m.atomic_energies_fn.atomic_energies.detach().cpu().numpy().ravel().tolist()
    E0S.parent.mkdir(parents=True, exist_ok=True)
    E0S.write_text(json.dumps({"atomic_numbers": z, "atomic_energies": {str(k): v for k, v in zip(z, e)},
                               "source": str(FOUNDATION)}, indent=1))


def flagged():
    parts = [np.load(p) for p in sorted(glob.glob(str(SEARCH / "prefilter_parts" / "part_*.npz")))]
    fr = np.concatenate([p["frame"] for p in parts]); md = np.concatenate([p["min_dist"] for p in parts])
    return set(fr[md < 0.8].tolist())


def mof_off_ids():
    tabs = []
    for split in ["train", "val", "test"]:
        t = pd.read_csv(MOF_OFF_PBE / f"PBE_{split}_sample_ids.tsv", sep="\t"); t["split"] = split; tabs.append(t)
    t = pd.concat(tabs, ignore_index=True)
    t["framework"] = t.cif_name.str.extract(r"^(.*_FSR)")[0]
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from ase.db import connect
    from ase.io import Trajectory

    foundation_e0s()
    queries = pd.read_csv(SEARCH / "queries.csv").set_index("query")
    sample = pd.read_csv(SAMPLE / "structures.csv")
    r2 = pd.read_csv(MOF_OFF_R2SCAN / "R2SCAN_train_sample_ids.tsv", sep="\t")
    pbe = mof_off_ids()
    trajs = {s: Trajectory(str(MOF_OFF_PBE / f"PBE_{s}.traj")) for s in ["train", "val", "test"]}
    qnames = list(pd.read_csv(SEARCH / "queries.csv")["query"])
    parts = [np.load(p) for p in sorted(glob.glob(str(SEARCH / "rerank_parts" / "part_*.npz")))]
    rerank = pd.DataFrame(dict(query=np.array(qnames)[np.concatenate([p["query"] for p in parts])],
                               frame=np.concatenate([p["frame"] for p in parts]),
                               fgw=np.concatenate([p["fgw"] for p in parts])))
    gdist = np.load(f"{SCRATCH}/global_pool/dist_zmean.npy", mmap_mode="r")
    bad = flagged()
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    n_frames = len(gdist)

    for target in args.targets:
        q = queries.loc[target]
        rec = r2.iloc[int(sample.source_frame[q.idx])]
        framework = rec.cif_name.split("_FSR")[0] + "_FSR"
        rows = pbe[pbe.framework == framework].copy()
        rows["is_query"] = rows.record_id == rec.record_id
        assert rows.is_query.sum() == 1, f"{target}: query frame not found among PBE frames"
        atoms = [trajs[s][int(i)] for s, i in zip(rows.split, rows.split_row)]
        assert all(a.info["mof_off_record_id"] == r for a, r in zip(atoms, rows.record_id))
        rows["natoms"] = [len(a) for a in atoms]
        write_lmdb(target_dir(target) / "target.lmdb", atoms, args.overwrite)
        rows.to_csv(target_dir(target) / "target.csv", index=False)

        fgw = rerank[rerank["query"] == target].sort_values("fgw")
        fgw = fgw[~fgw.frame.isin(bad)].frame.tolist()
        rng = np.random.default_rng(zlib.crc32(target.encode()))
        rand = [int(f) for f in rng.choice(n_frames, 200, replace=False) if int(f) not in bad]
        d = np.array(gdist[:, qnames.index(target)], dtype=np.float64)
        d[list(bad)] = np.inf
        glob_ = np.argsort(d)[:N_TRAIN + N_VALID].tolist()
        picks = {"fgw": fgw, "random": rand, "global": glob_}
        for sel in SELECTIONS:
            frames = picks[sel][:N_TRAIN + N_VALID]
            at = [db.get_atoms(id=int(f) + 1) for f in frames]
            write_lmdb(run_dir(target, sel) / "train.lmdb", at[:N_TRAIN], args.overwrite)
            write_lmdb(run_dir(target, sel) / "valid.lmdb", at[N_TRAIN:], args.overwrite)
            pd.DataFrame(dict(split=["train"] * N_TRAIN + ["valid"] * N_VALID, rank=range(1, len(frames) + 1),
                              sr10m_frame=frames, formula=[a.get_chemical_formula(mode="hill", empirical=True) for a in at],
                              natoms=[len(a) for a in at])).to_csv(run_dir(target, sel) / "sets.csv", index=False)
        print(f"{target} {q.formula}: framework {framework}, {len(rows)} PBE frames "
              f"({rows.natoms.iloc[0]} atoms); sets written for {', '.join(SELECTIONS)}", flush=True)


if __name__ == "__main__":
    main()
