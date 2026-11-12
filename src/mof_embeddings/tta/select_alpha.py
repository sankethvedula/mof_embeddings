#!/usr/bin/env python
"""Rank fine-tuning candidates by structure FGW at several alphas within a candidate pool.

The 10M search ranked structures by FGW at alpha = 0.5 (features and geometry weighted equally).
Training-set analysis (tta.neighbour_terms) showed that the feature term tracks the fine-tuning gain
better than the geometry term, so candidates are re-ranked at the lower alphas of
tta.common.METHOD_ALPHA. Pools:
  search queries (Q targets)  top 1000 by FGW at alpha 0.5 (search rerank) U top 500 by the
                              Z-weighted global vector; scored at the alphas below 0.5
  further targets (S)         top 1000 by the Z-weighted global vector (no 10M FGW search was run for
                              them); scored at every alpha including 0.5
Structures flagged as unphysical (atoms < 0.8 A apart) are excluded. For every pool structure and
alpha the FGW value and its feature and geometry terms at that alpha's optimal plan are recorded;
tta.build_sets takes ranks 1-20 (train) and 21-25 (valid). Rankings are exact only within the pool.

  $PY -m mof_embeddings.tta.select_alpha [--targets Q6 S12 ...] [--workers 32] [--chunk 500]
  -> data/tta/alpha_rank/<target>.csv
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

from mof_embeddings.config import DATA, SCRATCH, use_local_packages
from mof_embeddings.tta.common import METHOD_ALPHA, SEARCH_TARGETS, TARGETS, target_query

OUT = DATA / "tta" / "alpha_rank"
N_FGW, N_GLOBAL, N_GLOBAL_ONLY = 1000, 500, 1000
SEARCH_ALPHA = 0.5


def global_distances(targets, mu, sd, chunk=1_000_000):
    """Euclidean distance between each target's Z-weighted mean UMA vector and every sr_10m frame's."""
    pooled = np.load(f"{SCRATCH}/global_pool/zmean.npy", mmap_mode="r")
    qv = []
    for t in targets:
        x, z, _ = target_query(t, mu, sd)
        qv.append((x * z[:, None]).sum(0) / z.sum())
    qv = np.stack(qv)
    d = np.empty((len(pooled), len(targets)), dtype=np.float32)
    for i in range(0, len(pooled), chunk):
        p = np.asarray(pooled[i:i + chunk], dtype=np.float64)
        d[i:i + chunk] = np.sqrt(np.maximum((p ** 2).sum(1)[:, None] - 2 * p @ qv.T + (qv ** 2).sum(1)[None], 0))
    return d


def search_pool(target, qnames, rerank, gdist, bad):
    r = rerank[rerank["query"] == qnames.index(target)].sort_values("fgw")
    r = r[~r.frame.isin(bad)]
    top = r.frame.tolist()[:N_FGW]
    d = np.array(gdist[:, qnames.index(target)], dtype=np.float64)
    d[list(bad)] = np.inf
    frames = list(dict.fromkeys(top + np.argsort(d)[:N_GLOBAL].tolist()))
    return pd.DataFrame(dict(sr10m_frame=frames, in_fgw_top=np.isin(frames, top),
                             fgw_a0_5=r.set_index("frame").fgw.reindex(frames).values, global_dist=d[frames]))


def global_pool(d, bad):
    d = np.array(d, dtype=np.float64)
    d[list(bad)] = np.inf
    frames = np.argsort(d)[:N_GLOBAL_ONLY].tolist()
    return pd.DataFrame(dict(sr10m_frame=frames, in_fgw_top=False, global_dist=d[frames]))


def score(job):
    """FGW at each alpha for one chunk of a target's pool; returns (target, scored rows)."""
    target, cand, alphas = job
    use_local_packages()
    import ot
    from ase.db import connect
    from scipy.spatial.distance import cdist
    from mof_embeddings.mof_search.common import OUT as SEARCH, SR10M, SR10M_LMDB, uma_stats
    from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

    const = json.loads((SEARCH / "constants.json").read_text())
    m0, k0, sigma = const["m0"], const["k0"], const["sigma"]
    mu, sd = uma_stats()
    f1, _, a1 = target_query(target, mu, sd)
    C, _ = periodic_kernel_fast(np.array(a1.cell), a1.positions, sigma)
    K1 = len(C) * C / k0
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    p = ot.unif(len(f1))
    rows = []
    for f in cand.sr10m_frame:
        at = db.get_atoms(id=int(f) + 1)
        f2 = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
        C2, _ = periodic_kernel_fast(np.array(at.cell), at.positions, sigma)
        K2 = len(C2) * C2 / k0
        M = cdist(f1, f2, "sqeuclidean") / m0
        pq = ot.unif(len(f2))
        T0 = ot.emd(p, pq, M)
        cK = ot.gromov.init_matrix(K1, K2, p, pq, "square_loss")
        row = dict(sr10m_frame=int(f), natoms=len(at), formula=at.get_chemical_formula(mode="hill", empirical=True))
        for a in alphas:
            best = (np.inf, None)
            for G0 in [T0, None]:
                T, log = ot.gromov.fused_gromov_wasserstein(M, K1, K2, p, pq, loss_fun="square_loss", alpha=a,
                                                            G0=G0, log=True, max_iter=200)
                if log["fgw_dist"] < best[0]:
                    best = (float(log["fgw_dist"]), T)
            tag = f"a{a:g}".replace(".", "_")
            row[f"fgw_{tag}"] = best[0]
            row[f"feature_{tag}"] = float(np.sum(M * best[1]))
            row[f"geometry_{tag}"] = float(ot.gromov.gwloss(*cK, best[1]))
        rows.append(row)
    return target, cand.merge(pd.DataFrame(rows), on="sr10m_frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=500, help="pool structures per task")
    parser.add_argument("--skip-done", action="store_true", help="skip targets that already have a ranking")
    args = parser.parse_args()
    if args.skip_done:
        args.targets = [t for t in args.targets if not (OUT / f"{t}.csv").exists()]
        if not args.targets:
            return
    from mof_embeddings.mof_search.common import OUT as SEARCH, uma_stats
    from mof_embeddings.tta.build_sets import flagged
    bad = flagged()
    pools, alphas = {}, {}
    search = [t for t in args.targets if t in SEARCH_TARGETS]
    extra = [t for t in args.targets if t not in SEARCH_TARGETS]
    if search:
        qnames = list(pd.read_csv(SEARCH / "queries.csv")["query"])
        parts = [np.load(p) for p in sorted(glob.glob(str(SEARCH / "rerank_parts" / "part_*.npz")))]
        rerank = pd.DataFrame(dict(query=np.concatenate([p["query"] for p in parts]),
                                   frame=np.concatenate([p["frame"] for p in parts]),
                                   fgw=np.concatenate([p["fgw"] for p in parts])))
        gdist = np.load(f"{SCRATCH}/global_pool/dist_zmean.npy", mmap_mode="r")
        for t in search:
            pools[t] = search_pool(t, qnames, rerank, gdist, bad)
            alphas[t] = sorted({a for a in METHOD_ALPHA.values() if a != SEARCH_ALPHA})
    if extra:
        mu, sd = uma_stats()
        d = global_distances(extra, mu, sd)
        for j, t in enumerate(extra):
            pools[t] = global_pool(d[:, j], bad)
            alphas[t] = sorted(set(METHOD_ALPHA.values()))
    print("pool sizes:", {t: len(c) for t, c in pools.items()}, flush=True)
    jobs = [(t, c.iloc[i:i + args.chunk], alphas[t]) for t, c in pools.items() for i in range(0, len(c), args.chunk)]
    jobs.sort(key=lambda j: -len(j[2]))                 # three-alpha chunks first
    done = {t: [] for t in pools}
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(max(args.workers, 1)) as pl:
        for t, df in pl.imap_unordered(score, jobs):
            done[t].append(df)
            if sum(map(len, done[t])) == len(pools[t]):
                pd.concat(done[t]).to_csv(OUT / f"{t}.csv", index=False)
                print(f"{t}: {len(pools[t])} pool structures scored at alpha {alphas[t]}", flush=True)


if __name__ == "__main__":
    main()
