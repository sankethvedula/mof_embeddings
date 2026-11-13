#!/usr/bin/env python
"""Rank MOF-off frames of other frameworks as fine-tuning neighbours of a target, by structure FGW.

OMat24 neighbours carry PBE(+U) labels without dispersion; the targets are MOF-off PBE-D3(BJ). This
ranks MOF-off frames instead, whose labels match the target's level of theory, so that the effect of
FGW selection can be separated from the level-of-theory mismatch.

Candidates: the MOF-off frames with UMA atom features (the mof_off_r2scan bank, 10% of the R2SCAN
training frames: 8064 frames of 2305 frameworks); every one has a PBE-labelled twin with the same
cif_name. Frames of the target's own framework and frames with more than --max-atoms atoms are
excluded. Pool: the --pool frames nearest by the Z-weighted mean UMA vector (the global vector of
tta.select_alpha), at most --per-framework frames per framework. Each pool frame is scored by FGW at
alpha 0.25 (UMA features z-scored with the search statistics, intensive periodic kernel, the
constants of the 10M search) and split into feature and geometry terms at the optimal plan.
tta.build_sets keeps the best frame of each framework and takes 20 frameworks for training and the
next 5 for validation (selection moffgw).

  $PY -m mof_embeddings.tta.select_mof [--targets Q6 S12 ...] [--workers 24] [--skip-done]
  -> data/tta/mof_rank/<target>.csv (one row per pool frame: cif_name, framework, natoms, formula,
     global_dist, fgw, feature, geometry)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA, use_local_packages
from mof_embeddings.tta.common import MOF_OFF_R2SCAN, TARGETS, target_query, targets_table

OUT = DATA / "tta" / "mof_rank"
BANK = "uma/atom/external_subsets/mof_off_r2scan"
ALPHA = 0.25


def bank():
    """Frame table of the MOF-off UMA bank: source row, cif_name, framework, natoms, atom offsets."""
    from mof_embeddings.config import ARCHIVE      # = mof_search.common.A, without its imports (MACE env)
    d = Path(f"{ARCHIVE}/embeddings") / BANK
    src = np.load(d / "frame_source_index.npy")
    off = np.load(d / "frame_atom_offsets.npy")
    r2 = pd.read_csv(MOF_OFF_R2SCAN / "R2SCAN_train_sample_ids.tsv", sep="\t")
    t = r2.iloc[src][["cif_name"]].reset_index(drop=True)
    t["framework"] = t.cif_name.str.extract(r"^(.*_FSR)")[0]
    t["source"] = src
    t["start"], t["stop"] = off[:-1], off[1:]
    t["natoms"] = t.stop - t.start
    return t, d


def target_framework(target):
    from mof_embeddings.tta.common import SAMPLE
    idx = int(targets_table().set_index("query").idx[target])
    s = pd.read_csv(SAMPLE / "structures.csv")
    r2 = pd.read_csv(MOF_OFF_R2SCAN / "R2SCAN_train_sample_ids.tsv", sep="\t")
    return r2.cif_name.iloc[int(s.source_frame[idx])].split("_FSR")[0] + "_FSR"


def rank_target(job):
    target, n_pool, per_framework, max_atoms = job
    use_local_packages()
    import ot
    from ase.io import Trajectory
    from scipy.spatial.distance import cdist
    from mof_embeddings.mof_search.common import OUT as SEARCH, uma_stats
    from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

    const = json.loads((SEARCH / "constants.json").read_text())
    m0, k0, sigma = const["m0"], const["k0"], const["sigma"]
    mu, sd = uma_stats()
    t, d = bank()
    feats = np.load(d / "atom_latents.npy", mmap_mode="r")
    zs = np.load(d / "atom_atomic_numbers.npy", mmap_mode="r")
    x1, z1, a1 = target_query(target, mu, sd)
    qv = (x1 * z1[:, None]).sum(0) / z1.sum()

    ok = t[(t.framework != target_framework(target)) & (t.natoms <= max_atoms)].copy()
    gd = []
    for r in ok.itertuples():                       # global vector of every candidate frame
        x = (np.asarray(feats[r.start:r.stop], dtype=np.float64) - mu) / sd
        z = np.asarray(zs[r.start:r.stop], dtype=np.float64)
        gd.append(np.linalg.norm((x * z[:, None]).sum(0) / z.sum() - qv))
    ok["global_dist"] = gd
    ok = ok.sort_values("global_dist")
    ok = ok[ok.groupby("framework").cumcount() < per_framework].head(n_pool)

    traj = Trajectory(str(MOF_OFF_R2SCAN / "R2SCAN_train.traj"))
    C, _ = periodic_kernel_fast(np.array(a1.cell), a1.positions, sigma)
    K1 = len(C) * C / k0
    p = ot.unif(len(x1))
    rows = []
    for r in ok.itertuples():
        at = traj[int(r.source)]
        assert len(at) == r.natoms and (at.numbers == np.asarray(zs[r.start:r.stop])).all()
        f2 = (np.asarray(feats[r.start:r.stop], dtype=np.float64) - mu) / sd
        C2, _ = periodic_kernel_fast(np.array(at.cell), at.positions, sigma)
        K2 = len(C2) * C2 / k0
        M = cdist(x1, f2, "sqeuclidean") / m0
        pq = ot.unif(len(f2))
        best = (np.inf, None)
        for G0 in [ot.emd(p, pq, M), None]:
            T, log = ot.gromov.fused_gromov_wasserstein(M, K1, K2, p, pq, loss_fun="square_loss", alpha=ALPHA,
                                                        G0=G0, log=True, max_iter=200)
            if log["fgw_dist"] < best[0]:
                best = (float(log["fgw_dist"]), T)
        cK = ot.gromov.init_matrix(K1, K2, p, pq, "square_loss")
        rows.append(dict(cif_name=r.cif_name, framework=r.framework, natoms=int(r.natoms),
                         formula=at.get_chemical_formula(mode="hill", empirical=True), global_dist=r.global_dist,
                         fgw=best[0], feature=float(np.sum(M * best[1])),
                         geometry=float(ot.gromov.gwloss(*cK, best[1]))))
    df = pd.DataFrame(rows).sort_values("fgw")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{target}.csv", index=False)
    return f"{target}: {len(df)} MOF-off frames scored ({df.framework.nunique()} frameworks), " \
           f"nearest {df.framework.iloc[0]} fgw {df.fgw.iloc[0]:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pool", type=int, default=300, help="frames scored by FGW per target")
    parser.add_argument("--per-framework", type=int, default=2, help="pool frames per framework")
    parser.add_argument("--max-atoms", type=int, default=200)
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()
    targets = [t for t in args.targets if not (args.skip_done and (OUT / f"{t}.csv").exists())]
    jobs = [(t, args.pool, args.per_framework, args.max_atoms) for t in targets]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(rank_target, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(rank_target(j), flush=True)


if __name__ == "__main__":
    main()
