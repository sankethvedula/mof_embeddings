#!/usr/bin/env python
"""Best OMat24 match for each MOF query within each OMat24 subset family.

For every query and family, the --top best prefilter candidates of that family
(from the full 10M scan) are scored with exact structure FGW (UMA, sigma 3 A,
alpha 0.5); values already computed in the main rerank are reused. Writes all
scores and the best per (query, family) with formula, size and C-H bond check.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ase.db import connect

from mof_embeddings.config import ARCHIVE
from mof_embeddings.mof_search.compare_subset_terms import FAMILY, ORDER, ch_bonds
from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel_fast

L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"
_state = {}


def _init():
    const = json.loads((OUT / "constants.json").read_text())
    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    qK = []
    for a in qa:
        C, _ = periodic_kernel_fast(np.array(a.cell), a.positions, const["sigma"])
        qK.append(len(C) * C / const["k0"])
    _state.update(const=const, mu=mu, sd=sd, qf=qf, qK=qK, off=np.load(SR10M / "frame_atom_offsets.npy"),
                  feats=np.load(SR10M / "atom_latents.npy", mmap_mode="r"),
                  db=connect(SR10M_LMDB, type="aselmdb", readonly=True))


def _score(job):
    f, queries = job
    s = _state
    at = s["db"].get_atoms(id=int(f) + 1)
    x = (np.asarray(s["feats"][s["off"][f]:s["off"][f + 1]], dtype=np.float64) - s["mu"]) / s["sd"]
    C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, s["const"]["sigma"])
    K = len(C) * C / s["const"]["k0"]
    return [(q, f, fgw_distance(s["qf"][q], s["qK"][q], x, K, s["const"]["alpha"], s["const"]["m0"])) for q in queries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args()
    names = json.load(open(L + ".json"))["names"]
    family_of = np.array([FAMILY[n] for n in names])[np.load(L + ".npy")]
    parts = [np.load(p) for p in sorted((OUT / "prefilter_parts").glob("part_*.npz"))]
    fr = np.concatenate([p["frame"] for p in parts]); sc = np.concatenate([p["score"] for p in parts])
    sc[np.concatenate([p["min_dist"] for p in parts]) < 0.8] = np.inf
    fam = family_of[fr]
    queries = pd.read_csv(OUT / "queries.csv")

    want = {}                                                    # frame -> queries needing a score
    for j in range(sc.shape[1]):
        for family in ORDER:
            idx = np.flatnonzero(fam == family)
            for t in idx[np.argsort(sc[idx, j])[:args.top]]:
                want.setdefault(int(fr[t]), set()).add(j)
    done = pd.read_parquet(OUT / "rerank_results.parquet")
    qnum = {q: i for i, q in enumerate(queries["query"])}
    known = {(qnum[q], int(f)): d for q, f, d in zip(done["query"], done.frame, done.fgw)}
    jobs = [(f, sorted(j for j in qs if (j, f) not in known)) for f, qs in want.items()]
    jobs = [jb for jb in jobs if jb[1]]
    print(f"{sum(len(q) for q in want.values()):,} (query, frame) pairs, {len(jobs):,} frames need new FGW", flush=True)
    with Pool(args.procs, initializer=_init) as pool:
        new = [r for batch in pool.imap_unordered(_score, jobs, chunksize=20) for r in batch]
    scores = {**known, **{(q, f): d for q, f, d in new}}
    rows = [dict(query=queries["query"][q], family=family_of[f], frame=f, fgw=scores[(q, f)])
            for f, qs in want.items() for q in qs]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_subset_scores.csv", index=False)

    best = res.loc[res.groupby(["query", "family"]).fgw.idxmin()].copy()
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    info = {}
    for f in best.frame.unique():
        at = db.get_atoms(id=int(f) + 1)
        b = ch_bonds(at)
        info[f] = dict(formula=at.get_chemical_formula(mode="hill", empirical=True), natoms=len(at),
                       ch_median=float(np.median(b)) if len(b) else np.nan)
    for k in ["formula", "natoms", "ch_median"]:
        best[k] = best.frame.map(lambda f: info[f][k])
    best["vs_overall_best"] = best.fgw / best.groupby("query").fgw.transform("min")
    best = best.merge(queries[["query", "formula", "metal"]].rename(columns={"formula": "mof"}), on="query")
    best.to_csv(OUT / "best_per_subset.csv", index=False)
    piv = best.pivot(index="query", columns="family", values="vs_overall_best")[ORDER]
    piv = piv.loc[sorted(piv.index, key=lambda s: int(s[1:]))]
    print(piv.round(2).to_string())
    print(best.sort_values(["query", "fgw"])[["query", "mof", "family", "formula", "natoms", "fgw", "vs_overall_best", "ch_median"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
