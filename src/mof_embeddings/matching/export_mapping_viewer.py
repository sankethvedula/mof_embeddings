#!/usr/bin/env python
"""Bundle the small-cell -> supercell atom matchings (truth, plain OT, FGW) into
viewer/atom_mapping_viewer.html, an interactive page with linked 3D views.

Per material it exports: the supercell atoms; the small cell tiled 2x and
aligned onto the supercell (each copy tagged with its small-cell atom); every
method's match for each supercell atom; the feature-distance matrices that OT
works from; and the supercell's neighbour bonds with kept/broken flags. FGW is
exported for both geometry kernels (all periodic images / nearest image only);
embedding-only GW (gw_embedding_mapping.py) is added for each of its variants.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mof_embeddings.config import ROOT
from mof_embeddings.matching.fgw_atom_mapping import FEATURE_SETS, atom_features
from mof_embeddings.matching.ot_atom_mapping import BANKS, bank_stats

ROOT = Path(str(ROOT))
DATA = ROOT / "data/similar_structures"
KERNELS = ["periodic", "minimage"]
ALPHA = 0.25
# (key, label, feature set, kernel); truth and plain OT do not depend on the kernel
METHODS = [("truth", "True match", None, None)] + [
    (f"{fs}_{a}", f"{name} · {fs}", fs, kernel) for fs in FEATURE_SETS
    for a, name, kernels in [(0.0, "OT", [None]), (ALPHA, "FGW", KERNELS)] for kernel in kernels]
GW_VARIANTS = {"UMA": "GW · UMA", "MH1": "GW · MH1", "UMA+MH1": "GW · UMA+MH1",
               "UMA>MH1": "GW · UMA→MH1", "MH1>UMA": "GW · MH1→UMA"}


def bond_segment(lattice, frac, a, b):
    """Bond from atom a to the nearest periodic image of atom b, in Cartesian coordinates."""
    d = frac[b] - frac[a]
    d -= np.round(d)
    start = frac[a] @ lattice
    return np.round(np.concatenate([start, start + d @ lattice]), 4).tolist()


def main():
    frames = pd.read_csv(DATA / "frames.csv")
    runs = {k: dict(geom=json.loads((DATA / f"fgw_mapping_{k}/geometry.json").read_text()),
                    matches=pd.read_csv(DATA / f"fgw_mapping_{k}/atom_matches.csv"),
                    bonds=pd.read_csv(DATA / f"fgw_mapping_{k}/neighbour_scores.csv")) for k in KERNELS}
    gw_run = dict(geom=json.loads((DATA / "gw_embedding/geometry.json").read_text()),
                  matches=pd.read_csv(DATA / "gw_embedding/atom_matches.csv"),
                  bonds=pd.read_csv(DATA / "gw_embedding/neighbour_scores.csv"))
    emb = np.load(DATA / "embeddings.npz")
    off = emb["frame_offsets"]
    stats = {m: bank_stats(p) for m, p in BANKS.items()}

    materials = []
    for k in range(0, len(frames), 2):
        mat = frames.material[k]
        g = runs["periodic"]["geom"][mat]
        lattice = np.array(g["lattice"])
        truth = np.array(g["truth"])
        distance = {}
        for model in FEATURE_SETS:
            a = atom_features(model, emb, stats, off[k], off[k + 1])
            b = atom_features(model, emb, stats, off[k + 1], off[k + 2])
            d = cdist(a, b, "sqeuclidean")
            distance[model] = np.round(d / d.max(), 3).tolist()
        methods = []
        for key, label, model, kernel in METHODS:
            run = runs[kernel or "periodic"]
            gk, matches, bonds = run["geom"][mat], run["matches"], run["bonds"]
            if key == "truth":
                match, exact = truth, 1.0
                kept = bonds[(bonds.material == mat) & (bonds.model == "truth")].neighbours_kept.iloc[0]
            else:
                m, alpha = key.rsplit("_", 1)
                match = np.array(gk[f"fgw_{m}_{float(alpha)}"])
                sub = matches[(matches.material == mat) & (matches.model == m) & (matches.alpha == float(alpha))]
                exact = sub.exact.mean()
                kept = bonds[(bonds.material == mat) & (bonds.model == m) & (bonds.alpha == float(alpha))].neighbours_kept.iloc[0]
            same_el = np.mean(np.array(g["small_species"])[match] == np.array(g["large_species"]))
            methods.append(dict(key=key, label=label, model=model, kernel=kernel, match=match.tolist(),
                                exact=round(float(exact), 3), same_element=round(float(same_el), 3),
                                bonds_kept=round(float(kept), 3), bonds_flags=gk["bonds_kept"][key],
                                cross_distance=key != "truth"))
        gg = gw_run["geom"][mat]
        assert gg["bonds"] == g["bonds"]
        for variant, label in GW_VARIANTS.items():
            match = np.array(gg[f"gw_{variant}"])
            sub = gw_run["matches"][(gw_run["matches"].material == mat) & (gw_run["matches"].model == variant)]
            kept = gw_run["bonds"][(gw_run["bonds"].material == mat) & (gw_run["bonds"].model == variant)].neighbours_kept.iloc[0]
            methods.append(dict(key=f"GW_{variant}", label=label, model=variant if variant in FEATURE_SETS else "UMA",
                                kernel=None, match=match.tolist(), exact=round(float(sub.exact.mean()), 3),
                                same_element=round(float(sub.same_element.mean()), 3), bonds_kept=round(float(kept), 3),
                                bonds_flags=gg["bonds_kept"][f"GW_{variant}"], cross_distance=False))
        materials.append(dict(
            id=mat, parent_id=frames.parent_id[k], formula=g["formula"], spg=g["spg"], orbit=g["orbit"],
            small_species=g["small_species"], large_species=g["large_species"],
            lattice=np.round(lattice, 4).tolist(),
            large=np.round(np.array(g["large_frac"]) @ lattice, 4).tolist(),
            # aligned small cell: same order as the supercell; site j is a copy of small atom truth[j]
            aligned=np.round(np.array(g["aligned_frac"]) @ lattice, 4).tolist(),
            bond_index=[i for i, b in enumerate(g["bonds"]) if b[0] < b[1]],
            bond_segments=[bond_segment(lattice, np.array(g["large_frac"]), a, b) for a, b in g["bonds"] if a < b],
            distance=distance, methods=methods))

    template = (ROOT / "viewer/atom_mapping_viewer.template.html").read_text()
    out = ROOT / "viewer/atom_mapping_viewer.html"
    out.write_text(template.replace("/*__DATA__*/null", json.dumps(dict(alpha=ALPHA, materials=materials))))
    print("wrote", out, f"({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
