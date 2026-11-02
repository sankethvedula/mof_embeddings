#!/usr/bin/env python
"""Stage 1 of the MOF -> OMat24 sr_10m search: a cheap FGW stand-in for every
structure in sr_10m (10M) against the 10 MOF queries. Run as a Slurm array over
contiguous frame blocks; `--merge` keeps the top --keep candidates per query.

Prefilter score (mirrors FGW at alpha 0.5):
  0.5 * OT(features)  +  0.5 * W2^2 between 1D distributions of per-atom kernel
  eccentricity sqrt(mean_b K[a,b]^2)
with UMA features, M scaled by m0, and the intensive periodic kernel K (sigma 3 A,
scaled by k0), exactly as in the FGW rerank. On a 1,000-structure check it has
Spearman 0.90 with full FGW and puts 99% of FGW's top 10 in its own top 10%.
Frames with any interatomic distance < 0.8 A are flagged (unphysical rattles).
"""
import argparse
import json
import time

import numpy as np

from mof_embeddings.config import use_local_packages
use_local_packages()
import ot
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import cdist

from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

SIGMA, QUANTILES = 3.0, (np.arange(64) + 0.5) / 64


def eccentricity_quantiles(K):
    return np.quantile(np.sqrt((K ** 2).mean(1)), QUANTILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=250)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--keep", type=int, default=50_000)
    args = parser.parse_args()
    const = json.loads((OUT / "constants.json").read_text())
    m0, k0 = const["m0"], const["k0"]
    part_dir = OUT / "prefilter_parts"
    part_dir.mkdir(exist_ok=True)

    if args.merge:
        parts = [np.load(p) for p in sorted(part_dir.glob("part_*.npz"))]
        assert len(parts) == args.n_tasks, f"{len(parts)} of {args.n_tasks} parts"
        frame = np.concatenate([p["frame"] for p in parts]); score = np.concatenate([p["score"] for p in parts])
        bad = np.concatenate([p["min_dist"] for p in parts]) < 0.8
        score[bad] = np.inf
        top = np.argsort(score, axis=0)[:args.keep]                  # (keep, n_queries)
        np.savez(OUT / "candidates.npz", frame=frame[top], score=np.take_along_axis(score, top, 0),
                 n_scanned=len(frame), n_flagged=int(bad.sum()))
        print(f"scanned {len(frame):,} frames, flagged {bad.sum():,} with atoms < 0.8 A apart; kept {args.keep} per query")
        return

    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    q_ecc, q_w = [], []
    for a in qa:
        C, _ = periodic_kernel_fast(np.array(a.cell), a.positions, SIGMA)
        q_ecc.append(eccentricity_quantiles(len(C) * C / k0)); q_w.append(np.full(len(a), 1 / len(a)))

    off = np.load(SR10M / "frame_atom_offsets.npy")
    frames = np.array_split(np.arange(len(off) - 1), args.n_tasks)[args.task]
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    block = (np.asarray(feats[off[frames[0]]:off[frames[-1] + 1]], dtype=np.float64) - mu) / sd
    base = off[frames[0]]
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    score = np.empty((len(frames), len(qf)), np.float32)
    min_dist = np.empty(len(frames), np.float32)
    t0 = time.time()
    for k, f in enumerate(frames):
        at = db.get_atoms(id=int(f) + 1)
        x = block[off[f] - base:off[f + 1] - base]
        C, min_dist[k] = periodic_kernel_fast(np.array(at.cell), at.positions, SIGMA)
        ecc = eccentricity_quantiles(len(C) * C / k0)
        w = np.full(len(x), 1 / len(x))
        for j, (qx, qe, qw) in enumerate(zip(qf, q_ecc, q_w)):
            score[k, j] = 0.5 * ot.emd2(qw, w, cdist(qx, x, "sqeuclidean") / m0) + 0.5 * np.mean((qe - ecc) ** 2)
        if k % 10_000 == 0:
            print(f"task {args.task}: {k}/{len(frames)} in {time.time() - t0:.0f}s", flush=True)
    np.savez(part_dir / f"part_{args.task:03d}.npz", frame=frames, score=score, min_dist=min_dist)
    print(f"task {args.task}: done {len(frames)} frames in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
