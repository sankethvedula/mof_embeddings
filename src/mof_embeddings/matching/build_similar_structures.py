#!/usr/bin/env python
"""Build a test set of the same material at different cell sizes, for trying
size-independent structure comparisons with UMA / MACE embeddings.

OMat24 generated each Alexandria parent material (parent_id `agmXXXXXXXXX_...`)
in several simulation families. `rattled-relax` uses the primitive-sized cell
(median 5 atoms, near equilibrium) and `rattled-300` a 2x supercell with small
displacements; `aimd-from-PBE-3000-*` uses a larger cell at 3000 K. All frames
come from OMat24 sr_1m, which has atom-level UMA-128 and MH1-96 embeddings.

Rattled frames are often strained as well as displaced, so the curated set keeps
only pairs whose volume per atom differs by less than --max-volume-change.

Outputs (in --out-dir):
  pairs_relax_r300.csv   every material with both a rattled-relax and a rattled-300 frame
  frames.csv             the curated set: one row per frame, with identities and row ranges
  structures.extxyz      the curated frames' geometries (same order as frames.csv)
  embeddings.npz         per-frame atom embeddings (UMA, MH1) and stored structure vectors
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from ase.io import write
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from mof_embeddings.config import ARCHIVE, DATA, PRETRAIN

A = Path(str(ARCHIVE))
IDENT = A / "derived/atomic_rbf_sr1m/identity"
UMA_ATOM = A / "embeddings/uma/atom/omat24_sr1m"
MH1_ATOM = A / "embeddings/mace_mh1/atom/omat24_sr1m/readout96"
SR1M_LMDB = f"{PRETRAIN}/data/processed/omat24/stratified_random_nested/sr_1m/train.lmdb"
STRUCTURE_BANKS = {  # full OMat24 structure banks, all in the same global row order
    "struct_uma128": A / "embeddings/uma/structure/omat24_full/all_uma_embeddings.npy",
    "struct_omat0_128": A / "embeddings/mace_omat0/structure/omat24_full/all_mace_omat0_descriptors.npy",
    "struct_mh1_96": A / "embeddings/mace_mh1/structure/readout96/omat24_full/all_mace_mh1_readout96.npy",
    "struct_mh1_1024": A / "embeddings/mace_mh1/structure/invariants1024/omat24_full/all_mace_mh1_inv1024.npy",
}
FAMILY = {"rattled-relax": "relax", "rattled-300": "r300", "rattled-300-subsampled": "r300",
          "aimd-from-PBE-3000-npt": "md3000", "aimd-from-PBE-3000-nvt": "md3000"}


def frame_table():
    prov = pd.read_parquet(IDENT / "frame_provenance.parquet", columns=["subdataset", "global_row"])
    parents = pd.read_parquet(IDENT / "parent_labels.parquet").parent_id.to_numpy()
    off = np.load(UMA_ATOM / "frame_atom_offsets.npy")
    f = pd.DataFrame({"frame": np.arange(len(off) - 1), "natoms": np.diff(off),
                      "atom_start": off[:-1], "atom_stop": off[1:],
                      "subdataset": prov.subdataset.to_numpy(), "global_row": prov.global_row.to_numpy(),
                      "parent_id": parents[np.load(IDENT / "frame_parent_label.npy")]})
    f["material"] = f.parent_id.str.split("_").str[0]
    f["family"] = f.subdataset.map(FAMILY)
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(f"{DATA}/similar_structures"))
    parser.add_argument("--n-materials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-volume-change", type=float, default=0.05)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="frames.csv files whose materials to skip")
    parser.add_argument("--allow-repeated-formulas", action="store_true")
    parser.add_argument("--with-md3000", action="store_true",
                        help="also add the material's 3000 K AIMD frame (the mapping scripts expect pairs only)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f = frame_table()
    db = connect(SR1M_LMDB, type="aselmdb", readonly=True)
    get = lambda frame: db.get_atoms(id=int(frame) + 1)
    first = f[f.family.notna()].sort_values("frame").groupby(["material", "family"]).first()

    # Every material with both a relaxed (small) and rattled-300 (larger) frame.
    pairs = first.frame.unstack()[["relax", "r300"]].dropna().astype(int)
    pairs = pairs.join(first.natoms.unstack()[["relax", "r300"]].add_prefix("natoms_"))
    pairs = pairs[pairs.natoms_relax < pairs.natoms_r300].rename(
        columns={"relax": "frame_relax", "r300": "frame_r300"})
    pairs.to_csv(args.out_dir / "pairs_relax_r300.csv")
    print(f"{len(pairs)} materials with a rattled-relax and a larger rattled-300 frame")

    # Curated set: distinct reduced formulas, similar volume per atom, verified supercell match;
    # with --with-md3000, a 3000 K AIMD frame of the same material is added when one exists.
    md = first.frame.unstack()["md3000"]
    cand = pairs[pairs.natoms_relax.between(2, 8)].sample(frac=1, random_state=args.seed)
    excluded = set().union(*[set(pd.read_csv(p).material) for p in args.exclude])
    cand = cand[~cand.index.isin(excluded)]
    ratio = cand.natoms_r300 / cand.natoms_relax
    cand = cand[np.isclose(ratio, np.round(ratio))]                   # whole-number supercells only
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8, attempt_supercell=True, primitive_cell=False)
    rows, atoms_list, seen = [], [], set()
    for mat, p in cand.iterrows():
        small, large = get(p.frame_relax), get(p.frame_r300)
        s_small, s_large = AseAtomsAdaptor.get_structure(small), AseAtomsAdaptor.get_structure(large)
        formula = s_small.composition.reduced_formula
        dv = abs(large.get_volume() / len(large) / (small.get_volume() / len(small)) - 1)
        repeated = formula in seen and not args.allow_repeated_formulas
        if repeated or s_large.composition.reduced_formula != formula or dv > args.max_volume_change:
            continue
        ok = matcher.fit(s_small, s_large)
        if not ok:
            continue
        seen.add(formula)
        n_kept = len({r["material"] for r in rows}) + 1
        frames = [("relax", p.frame_relax), ("r300", p.frame_r300)]
        if args.with_md3000 and not np.isnan(md.get(mat, np.nan)):
            frames.append(("md3000", int(md[mat])))
        for fam, fr in frames:
            at = get(fr)
            r = f.loc[fr]
            rows.append(dict(material=mat, parent_id=r.parent_id, formula=formula, family=fam,
                             subdataset=r.subdataset, frame=int(fr), natoms=int(r.natoms),
                             atom_start=int(r.atom_start), atom_stop=int(r.atom_stop), global_row=int(r.global_row),
                             volume_per_atom=at.get_volume() / len(at), supercell_match_to_relax=bool(ok) if fam != "relax" else None))
            atoms_list.append(at)
        print(f"  {mat} {formula:12s} " + " / ".join(f"{fam} {f.loc[fr].natoms}" for fam, fr in frames) + f" atoms; dV/atom {100*dv:.1f}%")
        if n_kept == args.n_materials:
            break

    frames = pd.DataFrame(rows)
    frames.to_csv(args.out_dir / "frames.csv", index=False)
    write(args.out_dir / "structures.extxyz", atoms_list)

    # Embeddings: atom sets per frame (ragged -> concatenated + offsets) and stored structure vectors.
    uma, mh1 = np.load(UMA_ATOM / "atom_latents.npy", mmap_mode="r"), np.load(MH1_ATOM / "atom_latents.npy", mmap_mode="r")
    z = np.load(UMA_ATOM / "atom_atomic_numbers.npy", mmap_mode="r")
    sl = [slice(a, b) for a, b in zip(frames.atom_start, frames.atom_stop)]
    out = dict(frame_offsets=np.concatenate([[0], np.cumsum(frames.natoms)]),
               atom_uma128=np.concatenate([uma[s] for s in sl]),
               atom_mh1_96=np.concatenate([mh1[s] for s in sl]),
               atomic_numbers=np.concatenate([z[s] for s in sl]))
    for key, path in STRUCTURE_BANKS.items():
        bank = np.load(path, mmap_mode="r")
        out[key] = np.stack([bank[g] for g in frames.global_row])
    for (s, a) in zip(sl, atoms_list):  # atom order in the banks must match the structures
        assert np.array_equal(np.asarray(z[s]), a.numbers)
    np.savez(args.out_dir / "embeddings.npz", **out)
    print(f"wrote {len(frames)} frames from {frames.material.nunique()} materials to {args.out_dir}")


if __name__ == "__main__":
    main()
