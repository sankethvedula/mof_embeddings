#!/usr/bin/env python
"""Facts about each MOF query's best AIMD 1000 K and AIMD 3000 K match, used to
judge whether the pick makes chemical sense. Writes data/mof_search/aimd_best_match_facts.csv.

Per (query, family): the match's formula, subset, frame, size, FGW distance and
ratio to the query's best match anywhere; from the FGW plan: share of MOF atoms
sent to the same element, where the MOF's metal atoms go, the dominant flow of
each MOF element; elements shared; atoms per cubic angstrom of both; C-H bond
medians; and whether the match is the top FGW hit over the full 50,000-candidate
rerank.
"""
import json

import numpy as np
import pandas as pd
from ase.db import connect

from mof_embeddings.config import ARCHIVE
from mof_embeddings.mof_search.compare_subset_terms import ch_bonds
from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel_fast

NONMETAL = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te", "I", "Xe"}
L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"


def main():
    const = json.loads((OUT / "constants.json").read_text())
    names = json.load(open(L + ".json"))["names"]
    label = np.load(L + ".npy", mmap_mode="r")
    best = pd.read_csv(OUT / "best_per_subset.csv")
    best = best[best.family.isin(["AIMD 1000 K", "AIMD 3000 K"])]
    queries = pd.read_csv(OUT / "queries.csv")
    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    rerank = pd.read_parquet(OUT / "rerank_results.parquet")
    top1 = rerank[rerank.fgw_rank == 1].set_index("query").frame

    def kernel(at):
        C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, const["sigma"])
        return len(C) * C / const["k0"]

    rows = []
    for _, r in best.sort_values(["query", "family"], key=lambda s: s.map(lambda v: int(v[1:]) if s.name == "query" else v)).iterrows():
        j = int(r["query"][1:]) - 1
        mof, f = qa[j], int(r.frame)
        at = db.get_atoms(id=f + 1)
        x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
        value, plan = fgw_distance(qf[j], kernel(mof), x, kernel(at), const["alpha"], const["m0"], return_plan=True)
        s1, s2 = np.array(mof.get_chemical_symbols()), np.array(at.get_chemical_symbols())
        same = float(plan[s1[:, None] == s2[None, :]].sum() / plan.sum())
        flows = []
        for e in dict.fromkeys(s1):
            row = plan[s1 == e].sum(0)
            tot = pd.Series(row).groupby(s2).sum().sort_values(ascending=False) / row.sum()
            flows.append(f"{e}→{tot.index[0]} {tot.iloc[0]:.0%}")
        metal = queries.metal[j]
        mrow = plan[s1 == metal].sum(0)
        mflow = (pd.Series(mrow).groupby(s2).sum().sort_values(ascending=False) / mrow.sum()).head(2)
        b1, b2 = ch_bonds(mof), ch_bonds(at)
        rows.append(dict(
            query=r["query"], mof_formula=queries.formula[j], mof_metal=metal, mof_atoms=len(mof),
            subset_family=r.family, omat_subset=names[label[f]], sr10m_frame=f, lmdb_id=f + 1,
            match_formula=at.get_chemical_formula(mode="hill", empirical=True), match_atoms=len(at),
            fgw=round(value, 4), vs_best_anywhere=round(float(r.vs_overall_best), 3),
            is_overall_top_hit=bool(top1.get(r["query"]) == f),
            shared_elements=" ".join(sorted(set(s1) & set(s2))),
            match_metals=" ".join(sorted(set(s2) - NONMETAL)) or "none",
            mof_atoms_to_same_element=round(same, 3),
            metal_goes_to="; ".join(f"{k} {v:.0%}" for k, v in mflow.items()),
            element_flows="; ".join(flows),
            mof_atoms_per_A3=round(len(mof) / mof.get_volume(), 4), match_atoms_per_A3=round(len(at) / at.get_volume(), 4),
            mof_CH_median_A=round(float(np.median(b1)), 3) if len(b1) else None,
            match_CH_median_A=round(float(np.median(b2)), 3) if len(b2) else None))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aimd_best_match_facts.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(df.drop(columns=["sr10m_frame", "lmdb_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
