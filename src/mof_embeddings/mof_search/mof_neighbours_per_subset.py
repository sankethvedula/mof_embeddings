#!/usr/bin/env python
"""For each MOF query, its 5 closest distinct OMat24 neighbours in AIMD 1000 K,
AIMD 3000 K and rattled relax (from the FGW scores already computed in
per_subset_scores.csv), one row per (MOF, neighbour) pair, with the facts behind
the match and a generated chemical explanation.

"Distinct" = different chemical formula: frames of the same material (e.g.
consecutive frames of one AIMD run) count once, keeping the closest frame; the
number of such frames among the candidates is reported.

Explanations are built from the FGW plan and the structures: what kind of
compound the neighbour is, where the MOF's metal atoms go, how much of the MOF
maps onto the same element, whether the MOF's ligand donor atoms (N, O, S) land
on donor atoms, packing density and C-H bond lengths. Verdict ("metal sites" =
at least 50% of the MOF's metal goes to one metal of the neighbour):
  sensible   metal sites -> the same metal and >= 35% of atoms to the same element, or
             -> another transition/heavy metal and >= 45% to the same element
  partly     metal sites -> any metal (incl. alkali), or >= 45% to the same element
  not chemically meaningful   otherwise
"""
import json

import numpy as np
import pandas as pd
from ase.db import connect

from mof_embeddings.config import ARCHIVE
from mof_embeddings.mof_search.compare_subset_terms import ch_bonds
from mof_embeddings.mof_search.common import OUT, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.structure_distance.structure_fgw import fgw_distance, periodic_kernel_fast

FAMILIES = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax"]
NONMETAL = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te", "I", "Xe"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"}
L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"


def metal_kind(e):
    if e in NONMETAL:
        return "non-metal"
    return "alkali/alkaline-earth metal" if e in ALKALI else "transition/heavy metal"


def compound_type(symbols):
    els = set(symbols)
    metals = sorted(els - NONMETAL)
    m = f" ({', '.join(metals)})" if metals else ""
    if {"C", "H"} <= els and metals:
        return "metal-organic compound" + m
    if {"C", "H"} <= els:
        return "hydrocarbon" if els == {"C", "H"} else "metal-free organic/molecular crystal"
    if "C" in els and "O" in els and metals:
        return "metal carbonate/oxalate-type salt" + m
    if metals and "O" in els and "P" in els:
        return "metal phosphate" + m
    if metals and "O" in els:
        return "metal oxide/oxysalt" + m
    if "H" in els and ("B" in els or "N" in els):
        return "light-element hydride (B/N-H)" + m
    if metals:
        return "inorganic compound" + m
    return "non-metal compound"


def explain(r):
    parts = [f"{r.match_formula} is a {r.match_type}."]
    tgt, pct = r.metal_main_target, r.metal_main_share
    if tgt == r.mof_metal:
        parts.append(f"The MOF's {r.mof_metal} maps onto {tgt} itself ({pct:.0%}): same metal.")
    elif metal_kind(tgt) == "transition/heavy metal":
        parts.append(f"The MOF's {r.mof_metal} maps onto the {tgt} sites ({pct:.0%}): a different metal in the metal position.")
    elif metal_kind(tgt).startswith("alkali"):
        parts.append(f"The MOF's {r.mof_metal} maps onto the {tgt} cation ({pct:.0%}), a metal but not a transition metal.")
    else:
        second = [x.rsplit(" ", 1) for x in r.metal_goes_to.split("; ")[1:]]
        if second and metal_kind(second[0][0]) != "non-metal" and float(second[0][1].rstrip("%")) >= 25:
            parts.append(f"The MOF's {r.mof_metal} is split between {tgt} ({pct:.0%}) and the {second[0][0]} metal sites ({second[0][1]}).")
        else:
            parts.append(f"The MOF's {r.mof_metal} has no metal counterpart and is sent to {tgt} ({pct:.0%}).")
    same = r.mof_atoms_to_same_element
    parts.append(f"{same:.0%} of the MOF's atoms map onto the same element"
                 + (f" (shared: {r.shared_elements})." if r.shared_elements else "."))
    if pd.notna(r.mof_donors_to_donors):
        d = r.mof_donors_to_donors
        parts.append(f"{d:.0%} of the MOF's ligand donor atoms (N, O, S) land on N/O/S/halide atoms"
                     + (", so the ligand shell is reproduced." if d >= 0.7 else
                        ", so the ligand shell is partly reproduced." if d >= 0.4 else
                        "; the rest go to H or C, so the ligand shell is not reproduced."))
    dens = r.match_atoms_per_A3 / r.mof_atoms_per_A3
    word = "Packing density is similar" if 0.9 <= dens <= 1.1 else ("The neighbour is denser" if dens > 1.1 else "The neighbour is less dense")
    parts.append(f"{word} ({r.match_atoms_per_A3:.3f} vs {r.mof_atoms_per_A3:.3f} atoms/A^3).")
    if pd.notna(r.match_CH_median_A):
        parts.append(f"Its C-H bonds are {'near equilibrium' if r.match_CH_median_A < 1.15 else 'stretched'} "
                     f"(median {r.match_CH_median_A:.2f} A).")
    if r.verdict == "not chemically meaningful":
        parts.append("Picked mainly for similar C/H environments and packing rather than shared chemistry.")
    if r.vs_best_for_this_mof >= 1.5:
        parts.append(f"Distance is {r.vs_best_for_this_mof:.1f}x this MOF's best match, so a weak neighbour.")
    return " ".join(parts)


def main():
    const = json.loads((OUT / "constants.json").read_text())
    names = json.load(open(L + ".json"))["names"]
    label = np.load(L + ".npy", mmap_mode="r")
    queries = pd.read_csv(OUT / "queries.csv")
    scores = pd.read_csv(OUT / "per_subset_scores.csv")
    scores = scores[scores.family.isin(FAMILIES)]
    best_any = pd.read_csv(OUT / "best_per_subset.csv").groupby("query").fgw.min()
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    formula = {f: db.get_atoms(id=int(f) + 1).get_chemical_formula(mode="hill", empirical=True)
               for f in scores.frame.unique()}
    scores["match_formula"] = scores.frame.map(formula)

    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    off = np.load(SR10M / "frame_atom_offsets.npy")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")

    def kernel(at):
        C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, const["sigma"])
        return len(C) * C / const["k0"]

    rows = []
    for qi, qrow in queries.iterrows():
        mof = qa[qi]; s1 = np.array(mof.get_chemical_symbols()); Kq = kernel(mof); b1 = ch_bonds(mof)
        for fam in FAMILIES:
            sub = scores[(scores["query"] == qrow["query"]) & (scores.family == fam)].sort_values("fgw")
            n_frames = sub.groupby("match_formula").size()
            top = sub.drop_duplicates("match_formula").head(5)
            for rank, (_, t) in enumerate(top.iterrows(), 1):
                f = int(t.frame); at = db.get_atoms(id=f + 1)
                x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
                value, plan = fgw_distance(qf[qi], Kq, x, kernel(at), const["alpha"], const["m0"], return_plan=True)
                s2 = np.array(at.get_chemical_symbols())
                same = float(plan[s1[:, None] == s2[None, :]].sum() / plan.sum())
                mrow = plan[s1 == qrow.metal].sum(0)
                mflow = pd.Series(mrow).groupby(s2).sum().sort_values(ascending=False) / mrow.sum()
                flows = []
                for e in dict.fromkeys(s1):
                    tot = pd.Series(plan[s1 == e].sum(0)).groupby(s2).sum().sort_values(ascending=False)
                    flows.append(f"{e}->{tot.index[0]} {tot.iloc[0] / tot.sum():.0%}")
                donors = np.isin(s1, ["N", "O", "S"])
                acceptors = np.isin(s2, ["N", "O", "S", "F", "Cl", "Br", "I", "P", "Se", "Te"])
                donor_share = float(plan[np.ix_(donors, acceptors)].sum() / plan[donors].sum()) if donors.any() else np.nan
                b2 = ch_bonds(at)
                rows.append(dict(
                    query=qrow["query"], mof_formula=qrow.formula, mof_metal=qrow.metal, mof_atoms=len(mof),
                    subset_family=fam, rank_in_subset=rank, omat_subset=names[label[f]],
                    match_formula=t.match_formula, match_atoms=len(at), frames_of_this_material=int(n_frames[t.match_formula]),
                    fgw=round(value, 4), vs_best_for_this_mof=round(value / best_any[qrow["query"]], 3),
                    match_type=compound_type(s2), shared_elements=" ".join(sorted(set(s1) & set(s2))),
                    match_metals=" ".join(sorted(set(s2) - NONMETAL)) or "none",
                    mof_atoms_to_same_element=round(same, 3),
                    mof_donors_to_donors=round(donor_share, 3),
                    metal_main_target=mflow.index[0], metal_main_share=round(float(mflow.iloc[0]), 3),
                    metal_goes_to="; ".join(f"{k} {v:.0%}" for k, v in mflow.head(2).items()),
                    element_flows="; ".join(flows),
                    mof_atoms_per_A3=round(len(mof) / mof.get_volume(), 4),
                    match_atoms_per_A3=round(len(at) / at.get_volume(), 4),
                    mof_CH_median_A=round(float(np.median(b1)), 3) if len(b1) else None,
                    match_CH_median_A=round(float(np.median(b2)), 3) if len(b2) else None,
                    sr10m_frame=f, lmdb_id=f + 1))
        print(f"  {qrow['query']} done", flush=True)
    df = pd.DataFrame(rows)
    on_metal = df.metal_main_target.map(lambda e: e not in NONMETAL) & (df.metal_main_share >= 0.5)
    same_metal = on_metal & (df.metal_main_target == df.mof_metal)
    heavy = on_metal & df.metal_main_target.map(lambda e: metal_kind(e) == "transition/heavy metal")
    same = df.mof_atoms_to_same_element
    sensible = (same_metal & (same >= 0.35)) | (heavy & (same >= 0.45))
    df["verdict"] = np.where(sensible, "sensible", np.where(on_metal | (same >= 0.45), "partly", "not chemically meaningful"))
    df["explanation"] = [explain(r) for r in df.itertuples()]
    cols = ["query", "mof_formula", "mof_metal", "mof_atoms", "subset_family", "rank_in_subset", "omat_subset",
            "match_formula", "match_atoms", "frames_of_this_material", "fgw", "vs_best_for_this_mof", "verdict", "explanation",
            "match_type", "shared_elements", "match_metals", "mof_atoms_to_same_element", "mof_donors_to_donors",
            "metal_goes_to", "element_flows",
            "mof_atoms_per_A3", "match_atoms_per_A3", "mof_CH_median_A", "match_CH_median_A", "sr10m_frame", "lmdb_id"]
    df[cols].to_csv(OUT / "mof_neighbours_per_subset.csv", index=False)
    print(df.groupby("subset_family").verdict.value_counts().unstack(fill_value=0).to_string())
    print(f"{len(df)} rows -> {OUT / 'mof_neighbours_per_subset.csv'}")


if __name__ == "__main__":
    main()
