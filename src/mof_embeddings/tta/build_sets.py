#!/usr/bin/env python
"""Training, validation and target databases for test-time adaptation.

For each target MOF (a query of the 10M search):
  target.lmdb   every MOF-off PBE frame of the target framework (the query frame and the
                other AIMD frames, 300-1000 K), PBE labels; target.csv marks the query frame
  fgw/          OMat24 sr_10m frames ranked 1-20 by structure FGW to the query (train) and
                21-25 (valid), from the search's rerank results
  random/       25 random sr_10m frames (same split), a control
  global/       frames ranked 1-25 by the Z-weighted-mean global vector (global_vs_fgw), a control
  fgwmid/, fgwlow/   ranks 1-25 by FGW at alpha 0.25 / 0.1 within a candidate pool (tta.select_alpha);
                for S targets (not search queries) fgw/ and global/ also come from that pool
  fgwmidnou/    as fgwmid, without structures OMat24 labels with PBE+U (oxides/fluorides of U_METALS),
                from alpha_rank/<target>_nou.csv where the pool has fewer than 25 such structures;
                where this gives the same 25 structures as fgwmid only same_as.txt ("fgwmid") is written
  moffgw/       MOF-off PBE frames of other frameworks ranked by FGW at alpha 0.25 (tta.select_mof), the
                best frame of each framework: 20 frameworks for training, the next 5 for validation;
                same level of theory as the target (PBE-D3(BJ)). sets.csv lists cif_name and framework
  moffgwnf/     as moffgw, without frameworks of the target's empirical formula (other CSD entries or
                polymorphs of the same MOF); same_as.txt ("moffgw") where nothing is removed
  mofglobal/    MOF-off frames of other frameworks ranked by the Z-weighted-mean global vector (the same
                pool), best frame per framework: a control for moffgw
  mofrandom/    25 random MOF-off frames (mof_off_r2scan bank, <= 200 atoms) of 25 other frameworks:
                a control for moffgw
  fgw100/ ...   the same with N training structures (ranks 1-N, valid N+1..N+5) for --sizes 20 100;
                larger sets contain the smaller ones (same ranking, same random draw)
Frames flagged as unphysical (atoms < 0.8 A apart) are never used. Labels are copied as
stored: OMat24 PBE(+U) total energy, forces and stress; MOF-off PBE (D3BJ dataset) total
energy, forces and stress. Also writes the foundation model's E0s to $SCRATCH/tta.

Run in the MACE environment (uses mace's LMDB writer):
  $PY_MACE -m mof_embeddings.tta.build_sets [--targets Q6 Q3 ...] [--sizes 20 100] [--overwrite]
Existing databases are kept unless --overwrite is given.
"""
import argparse
import glob
import json
import zlib

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA, SCRATCH
from mof_embeddings.tta.common import (E0S, FOUNDATION, METHOD_ALPHA, METHODS, MOF_OFF_PBE, MOF_OFF_R2SCAN, N_TRAIN,
                                       N_VALID, SAMPLE, SEARCH, SR10M_LMDB, TARGETS, plus_u, run_dir, selection_name,
                                       target_dir, targets_table)


ALPHA_RANK = DATA / "tta" / "alpha_rank"     # tta.select_alpha
MOF_RANK = DATA / "tta" / "mof_rank"         # tta.select_mof


def write_lmdb(path, atoms_list, overwrite):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.stress import full_3x3_to_voigt_6_stress
    from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase
    if path.exists():
        if not overwrite:
            return False
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
    return True


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
    parser.add_argument("--sizes", nargs="+", type=int, default=[N_TRAIN], help="training-set sizes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from ase.db import connect
    from ase.io import Trajectory

    foundation_e0s()
    queries = targets_table().set_index("query")
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
    from mof_embeddings.tta.select_mof import bank
    mof_bank = bank()[0]

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

        rng = np.random.default_rng(zlib.crc32(target.encode()))
        rand = [int(f) for f in rng.choice(n_frames, 200, replace=False) if int(f) not in bad]
        picks = {"random": rand}
        if target in qnames:                       # search query: exact FGW rerank and global distances
            fgw = rerank[rerank["query"] == target].sort_values("fgw")
            picks["fgw"] = fgw[~fgw.frame.isin(bad)].frame.tolist()
            d = np.array(gdist[:, qnames.index(target)], dtype=np.float64)
            d[list(bad)] = np.inf
            picks["global"] = np.argsort(d)[:max(args.sizes) + N_VALID].tolist()
        rank_path = ALPHA_RANK / f"{target}.csv"
        if rank_path.exists():                     # candidate pool scored by tta.select_alpha
            pool = pd.read_csv(rank_path)
            picks.setdefault("global", pool.sort_values("global_dist").sr10m_frame.astype(int).tolist())
            for method, a in METHOD_ALPHA.items():
                col = "fgw_a" + f"{a:g}".replace(".", "_")
                if method not in picks and pool[col].notna().all():
                    picks[method] = pool.sort_values(col).sr10m_frame.astype(int).tolist()
            if "fgw_a0_25" in pool and pool.fgw_a0_25.notna().all():
                plain = pool[~pool.formula.map(plus_u)]
                picks["fgwmidnou"] = plain.sort_values("fgw_a0_25").sr10m_frame.astype(int).tolist()
                nou = ALPHA_RANK / f"{target}_nou.csv"          # pool all +U: second pool without +U
                if len(picks["fgwmidnou"]) < max(args.sizes) + N_VALID and nou.exists():
                    picks["fgwmidnou"] = pd.read_csv(nou).sort_values("fgw_a0_25").sr10m_frame.astype(int).tolist()
        mof_path = MOF_RANK / f"{target}.csv"
        done = []
        if mof_path.exists():
            mr = pd.read_csv(mof_path)
            own = atoms[int(np.flatnonzero(rows.is_query)[0])].get_chemical_formula(mode="hill", empirical=True)
            mof_picks = {"moffgw": mr.sort_values("fgw"), "moffgwnf": mr[mr.formula != own].sort_values("fgw"),
                         "mofglobal": mr.sort_values("global_dist")}
            b = mof_bank[(mof_bank.framework != framework) & (mof_bank.natoms <= 200)]
            b = b.sample(frac=1, random_state=zlib.crc32(target.encode()) % 2**32)
            mof_picks["mofrandom"] = b.drop_duplicates("framework")
            for sel, m in mof_picks.items():
                if ((run_dir(target, sel) / "sets.csv").exists() or (run_dir(target, sel) / "same_as.txt").exists()) \
                        and not args.overwrite:
                    continue
                m = m.drop_duplicates("framework").head(N_TRAIN + N_VALID).reindex(
                    columns=["cif_name", "framework", "fgw", "feature", "geometry", "global_dist"])
                assert len(m) == N_TRAIN + N_VALID and framework not in set(m.framework)
                if sel == "moffgwnf" and m.cif_name.tolist() == mof_picks["moffgw"].drop_duplicates(
                        "framework").head(N_TRAIN + N_VALID).cif_name.tolist():
                    run_dir(target, sel).mkdir(parents=True, exist_ok=True)
                    (run_dir(target, sel) / "same_as.txt").write_text("moffgw\n")
                    continue
                src = pbe.set_index("cif_name").loc[m.cif_name]
                at = [trajs[s_][int(i)] for s_, i in zip(src.split, src.split_row)]
                assert all(a.info["mof_off_record_id"] == r for a, r in zip(at, src.record_id))
                write_lmdb(run_dir(target, sel) / "train.lmdb", at[:N_TRAIN], args.overwrite)
                write_lmdb(run_dir(target, sel) / "valid.lmdb", at[N_TRAIN:], args.overwrite)
                m.insert(0, "rank", range(1, len(at) + 1))
                m.insert(0, "split", ["train"] * N_TRAIN + ["valid"] * N_VALID)
                m["formula"] = [a.get_chemical_formula(mode="hill", empirical=True) for a in at]
                m["natoms"] = [len(a) for a in at]
                m.to_csv(run_dir(target, sel) / "sets.csv", index=False)
                done.append(sel)
        for n, method in [(n, m) for n in args.sizes for m in METHODS if m in picks and not m.startswith("mof")]:
            sel = selection_name(method, n)
            if ((run_dir(target, sel) / "sets.csv").exists() or (run_dir(target, sel) / "same_as.txt").exists()) \
                    and not args.overwrite:
                continue
            frames = picks[method][:n + N_VALID]
            if len(frames) < n + N_VALID:              # e.g. a pool-ranked method asked for 100 structures
                continue
            if method == "fgwmidnou" and frames == picks.get("fgwmid", [])[:n + N_VALID]:
                run_dir(target, sel).mkdir(parents=True, exist_ok=True)
                (run_dir(target, sel) / "same_as.txt").write_text(selection_name("fgwmid", n) + "\n")
                continue
            at = [db.get_atoms(id=int(f) + 1) for f in frames]
            write_lmdb(run_dir(target, sel) / "train.lmdb", at[:n], args.overwrite)
            write_lmdb(run_dir(target, sel) / "valid.lmdb", at[n:], args.overwrite)
            done.append(sel)
            pd.DataFrame(dict(split=["train"] * n + ["valid"] * N_VALID, rank=range(1, len(frames) + 1),
                              sr10m_frame=frames, formula=[a.get_chemical_formula(mode="hill", empirical=True) for a in at],
                              natoms=[len(a) for a in at])).to_csv(run_dir(target, sel) / "sets.csv", index=False)
        print(f"{target} {q.formula}: framework {framework}, {len(rows)} PBE frames "
              f"({rows.natoms.iloc[0]} atoms); sets written: {', '.join(done) or 'none (all exist)'}", flush=True)


if __name__ == "__main__":
    main()
