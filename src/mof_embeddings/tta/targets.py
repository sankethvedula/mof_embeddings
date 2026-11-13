#!/usr/bin/env python
"""Choose further MOF targets for test-time adaptation from the structure-FGW sample.

Candidates are the MOF-off structures of data/structure_fgw/sample (which carry UMA atom features)
that are not queries of the 10M search. A candidate qualifies when its framework has at least
--min-frames MOF-off PBE frames (the target set), at most --max-atoms atoms (fine-tuning cost) and no
two atoms closer than 0.8 A. At most --per-metal targets share a metal (first metal in the formula);
within these limits targets are drawn at random (seed 0) up to --n. With --extend the existing table is
kept and further qualifying targets (other frameworks) are appended up to --n in total.

  $PY -m mof_embeddings.tta.targets [--n 20] [--min-frames 20] [--max-atoms 200] [--per-metal 2] [--extend]
  -> data/tta/targets_extra.csv (query = S<idx>, idx, formula, natoms, metal, min_dist, pbe_frames,
     round: 3 for the first 20 targets, one more for each --extend)
"""
import argparse

import numpy as np
import pandas as pd

from mof_embeddings.tta.common import EXTRA_TARGETS, MOF_OFF_PBE, MOF_OFF_R2SCAN, SAMPLE, SEARCH

NONMETAL = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te",
            "I", "Xe"}


def candidates():
    from ase.io import read
    from mof_embeddings.tta.build_sets import mof_off_ids
    s = pd.read_csv(SAMPLE / "structures.csv")
    r2 = pd.read_csv(MOF_OFF_R2SCAN / "R2SCAN_train_sample_ids.tsv", sep="\t")
    pbe = mof_off_ids().framework.value_counts()
    atoms = read(SAMPLE / "structures.extxyz", ":")
    rows = []
    for idx in np.flatnonzero(s.dataset.values == "MOF-off"):
        a = atoms[idx]
        rec = r2.iloc[int(s.source_frame[idx])]
        framework = rec.cif_name.split("_FSR")[0] + "_FSR"
        d = a.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        symbols = a.get_chemical_symbols()
        metals = [e for e in dict.fromkeys(symbols) if e not in NONMETAL]
        rows.append(dict(query=f"S{idx}", idx=int(idx), formula=a.get_chemical_formula(mode="hill", empirical=True),
                         natoms=len(a), metal=metals[0] if metals else "", min_dist=float(d.min()),
                         framework=framework, pbe_frames=int(pbe.get(framework, 0))))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-atoms", type=int, default=200)
    parser.add_argument("--per-metal", type=int, default=2)
    parser.add_argument("--extend", action="store_true", help="keep the existing targets and append new ones")
    args = parser.parse_args()
    c = candidates()
    q = pd.read_csv(SEARCH / "queries.csv")
    ok = c[~c.idx.isin(q.idx) & (c.pbe_frames >= args.min_frames) & (c.natoms <= args.max_atoms)
           & (c.min_dist >= 0.8) & (c.metal != "")]
    taken = q.metal.value_counts().to_dict()          # metals already covered by the search queries count
    rng = np.random.default_rng(0)
    pick = []
    if args.extend and EXTRA_TARGETS.exists():
        pick = [r for _, r in pd.read_csv(EXTRA_TARGETS).iterrows()]
        for r in pick:
            taken[r.metal] = taken.get(r.metal, 0) + 1
    for i in rng.permutation(len(ok)):
        r = ok.iloc[i]
        if len(pick) == args.n:
            break
        if r.idx in {p.idx for p in pick}:
            continue
        if taken.get(r.metal, 0) < args.per_metal and r.framework not in {p.framework for p in pick}:
            pick.append(r); taken[r.metal] = taken.get(r.metal, 0) + 1
    out = pd.DataFrame(pick)
    first = 3 if "round" not in out else out["round"].max()          # round 3 chose the first 20 targets
    out["round"] = out["round"].fillna(first + 1) if "round" in out else first
    out = out.astype({"round": int}).sort_values("idx")
    EXTRA_TARGETS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EXTRA_TARGETS, index=False)
    print(f"{len(c)} MOF-off sample structures, {len(ok)} qualify, {len(out)} chosen -> {EXTRA_TARGETS}")
    print(out[["query", "formula", "natoms", "metal", "pbe_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
