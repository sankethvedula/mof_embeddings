#!/usr/bin/env python
"""Stage 2 of the MOF -> OMat24 sr_10m search: exact structure FGW (UMA features,
sigma 3 A, alpha 0.5, structure_fgw.py) between each query and its prefilter
candidates. Run as a Slurm array over the unique candidate frames (each frame's
kernel is built once and reused for every query that shortlisted it); `--merge`
writes the ranked results and a per-query top-20 table.
"""
import argparse
import json

import numpy as np
import pandas as pd
from ase.db import connect

from mof_embeddings.config import ARCHIVE
from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel_fast

LABELS = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    const = json.loads((OUT / "constants.json").read_text())
    m0, k0, sigma, alpha = const["m0"], const["k0"], const["sigma"], const["alpha"]
    cand = np.load(OUT / "candidates.npz")
    frames, prescore = cand["frame"], cand["score"]                     # (keep, n_queries)
    part_dir = OUT / "rerank_parts"
    part_dir.mkdir(exist_ok=True)

    if args.merge:
        parts = [np.load(p) for p in sorted(part_dir.glob("part_*.npz"))]
        assert len(parts) == args.n_tasks, f"{len(parts)} of {args.n_tasks} parts"
        q_idx = np.concatenate([p["query"] for p in parts]); fr = np.concatenate([p["frame"] for p in parts])
        d = np.concatenate([p["fgw"] for p in parts])
        queries = pd.read_csv(OUT / "queries.csv")
        names = json.loads(open(LABELS + ".json").read())["names"]
        sub = np.load(LABELS + ".npy", mmap_mode="r")
        db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
        res = pd.DataFrame(dict(query=queries["query"].values[q_idx], frame=fr, fgw=d))
        rank_of = [dict(zip(frames[:, q].tolist(), range(1, len(frames) + 1))) for q in range(frames.shape[1])]
        res["prefilter_rank"] = [rank_of[q][f] for q, f in zip(q_idx, fr)]
        res = res.sort_values(["query", "fgw"])
        res["fgw_rank"] = res.groupby("query").cumcount() + 1
        res.to_parquet(OUT / "rerank_results.parquet", index=False)
        top = res[res.fgw_rank <= 20].copy()
        top["subset"] = [names[sub[f]] for f in top.frame]
        atoms = {f: db.get_atoms(id=int(f) + 1) for f in top.frame.unique()}
        top["formula"] = [atoms[f].get_chemical_formula(mode="hill", empirical=True) for f in top.frame]
        top["natoms"] = [len(atoms[f]) for f in top.frame]
        top = top.merge(queries[["query", "formula", "metal", "natoms"]], on="query", suffixes=("", "_query"))
        top.to_csv(OUT / "top20_per_query.csv", index=False)
        print(top[top.fgw_rank <= 3][["query", "formula_query", "fgw_rank", "formula", "subset", "natoms", "fgw",
                                      "prefilter_rank"]].round(3).to_string(index=False))
        return

    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    qK = []
    for a in qa:
        C, _ = periodic_kernel_fast(np.array(a.cell), a.positions, sigma)
        qK.append(len(C) * C / k0)
    unique = np.unique(frames)
    mine = np.array_split(unique, args.n_tasks)[args.task]
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    out_q, out_f, out_d = [], [], []
    for k, f in enumerate(mine):
        at = db.get_atoms(id=int(f) + 1)
        x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
        C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, sigma)
        K = len(C) * C / k0
        for q in np.flatnonzero((frames == f).any(0)):
            out_q.append(q); out_f.append(f); out_d.append(fgw_distance(qf[q], qK[q], x, K, alpha, m0))
        if k % 2000 == 0:
            print(f"task {args.task}: {k}/{len(mine)} frames", flush=True)
    np.savez(part_dir / f"part_{args.task:03d}.npz", query=np.array(out_q), frame=np.array(out_f), fgw=np.array(out_d))


if __name__ == "__main__":
    main()
