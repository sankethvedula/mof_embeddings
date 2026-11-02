#!/usr/bin/env python
"""Compare per-atom feature variants (UMA, and several MACE layers/heads) on the
50k-structure OMat24 sr_1m sample and the 94 same-material test pairs.

Variants (all per atom):
  UMA-128            archive bank
  MH1-96 readout     archive bank (hidden units of all 6 heads)
  MH1-16 omat head   channels 80-95 of the readout (the omat_pbe head's block)
  MH1-512 layer 1 / MH1-512 layer 2 / MH1-1024 both     new extraction (interaction invariants)
  OMAT0-128 layer 2 / OMAT0-256 both                   new extraction

Stages:
  check    Z-weighted sums of the new descriptors vs the archive's per-structure
           MH1-1024 and OMAT0-128 banks (relative error), and atom order vs the bank
  cluster  10-NN element vote and variance explained by element / element group,
           on all atoms of 5,000 sample structures (features z-scored per variant)
  match    plain OT and FGW (periodic kernel, alpha 0.25, 30 restarts) on the 94
           pairs: same element, exact site, neighbour bonds kept
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mof_embeddings.config import ARCHIVE, DATA, SCRATCH, use_local_packages

A = Path(str(ARCHIVE))
U = A / "embeddings/uma/atom/omat24_sr1m"
M = A / "embeddings/mace_mh1/atom/omat24_sr1m/readout96"
V = Path(f"{SCRATCH}/mace_variants")
OUT = Path(f"{DATA}/mace_variants")
VARIANTS = ["UMA-128", "MH1-96 readout", "MH1-16 omat head", "MH1-512 layer 1", "MH1-512 layer 2", "MH1-1024 both",
            "OMAT0-128 layer 2", "OMAT0-256 both"]


class Features:
    """Per-atom feature rows for any sr_1m frame in the extraction list, for every variant."""

    def __init__(self):
        self.frames = np.load(V / "frames.npy")
        self.pos = {int(f): i for i, f in enumerate(self.frames)}
        self.off = {m: np.load(V / f"{m}_offsets.npy") for m in ["mh1", "omat0"]}
        self.new = {m: np.load(V / f"{m}_atoms.npy", mmap_mode="r") for m in ["mh1", "omat0"]}
        self.bank_off = np.load(U / "frame_atom_offsets.npy")
        self.uma = np.load(U / "atom_latents.npy", mmap_mode="r")
        self.mh1r = np.load(M / "atom_latents.npy", mmap_mode="r")

    def get(self, variant, frame):
        f = int(frame)
        b = slice(self.bank_off[f], self.bank_off[f + 1])
        if variant == "UMA-128":
            return np.asarray(self.uma[b], dtype=np.float64)
        if variant == "MH1-96 readout":
            return np.asarray(self.mh1r[b], dtype=np.float64)
        if variant == "MH1-16 omat head":
            return np.asarray(self.mh1r[b][:, 80:96], dtype=np.float64)
        model = "mh1" if variant.startswith("MH1") else "omat0"
        i = self.pos[f]
        x = np.asarray(self.new[model][self.off[model][i]:self.off[model][i + 1]], dtype=np.float64)
        half = x.shape[1] // 2
        return {"MH1-512 layer 1": x[:, :half], "MH1-512 layer 2": x[:, half:], "MH1-1024 both": x,
                "OMAT0-128 layer 2": x[:, half:], "OMAT0-256 both": x}[variant]


def stats(F, frames):
    out = {}
    for v in VARIANTS:
        X = np.concatenate([F.get(v, f) for f in frames])
        out[v] = (X.mean(0), X.std(0) + 1e-8)
    return out


def check(F):
    gr = np.load(A / "derived/atomic_rbf_sr1m/identity/frame_global_row.npy")
    z_bank = np.load(U / "atom_atomic_numbers.npy", mmap_mode="r")
    banks = {"mh1": (np.load(A / "embeddings/mace_mh1/structure/invariants1024/omat24_full/all_mace_mh1_inv1024.npy", mmap_mode="r"), slice(None)),
             "omat0": (np.load(A / "embeddings/mace_omat0/structure/omat24_full/all_mace_omat0_descriptors.npy", mmap_mode="r"), slice(128, 256))}
    rng = np.random.default_rng(0)
    rows = []
    for f in rng.choice(F.frames, 200, replace=False):
        i = F.pos[int(f)]
        z = np.asarray(z_bank[F.bank_off[f]:F.bank_off[f + 1]])
        for m, (bank, cols) in banks.items():
            zn = np.load(V / f"{m}_atomic_numbers.npy", mmap_mode="r")[F.off[m][i]:F.off[m][i + 1]]
            x = np.asarray(F.new[m][F.off[m][i]:F.off[m][i + 1]], dtype=np.float64)[:, cols]
            H = (z[:, None] * x).sum(0)
            ref = np.asarray(bank[gr[f]], dtype=np.float64)
            rows.append(dict(model=m, atom_order_matches=bool(np.array_equal(zn, z)),
                             rel_error=float(np.linalg.norm(H - ref) / np.linalg.norm(ref))))
    df = pd.DataFrame(rows)
    res = df.groupby("model").agg(order_ok=("atom_order_matches", "mean"), median_rel_error=("rel_error", "median"),
                                  max_rel_error=("rel_error", "max"))
    print(res.to_string()); return res


def cluster(F, st):
    import torch
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    sample = F.frames[:50000]
    pick = np.sort(rng.choice(sample, 5000, replace=False))
    names = json.loads((U / "labels_element_group/row_labels.json").read_text())["names"]
    zb = np.load(U / "atom_atomic_numbers.npy", mmap_mode="r"); gb = np.load(U / "labels_element_group/row_labels.npy", mmap_mode="r")
    Z = np.concatenate([np.asarray(zb[F.bank_off[f]:F.bank_off[f + 1]]) for f in pick]).astype(int)
    G = np.concatenate([np.asarray(gb[F.bank_off[f]:F.bank_off[f + 1]]) for f in pick]).astype(int)
    rows = []
    for v in VARIANTS:
        mu, sd = st[v]
        X = np.concatenate([(F.get(v, f) - mu) / sd for f in pick])
        def eta2(y):
            tot = ((X - X.mean(0)) ** 2).sum()
            return 1 - sum(((X[y == e] - X[y == e].mean(0)) ** 2).sum() for e in np.unique(y)) / tot
        x = torch.from_numpy(X.astype(np.float32)).to(dev)
        nn = []
        for i in range(0, len(x), 2048):
            d = torch.cdist(x[i:i + 2048], x)
            d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
            nn.append(d.topk(10, largest=False).indices.cpu())
        nn = torch.cat(nn).numpy()
        vote = np.array([np.bincount(r).argmax() for r in Z[nn]])
        rows.append(dict(variant=v, width=X.shape[1], knn_element=float((vote == Z).mean()),
                         nbr_same_element=float((Z[nn] == Z[:, None]).mean()),
                         eta2_element=float(eta2(Z)), eta2_element_group=float(eta2(G))))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


_W = {}


def match_pair(task):
    """All variants x {OT, FGW} for one same-material pair (runs in a worker)."""
    d, k = task
    from ase.io import read
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    from scipy.spatial.distance import cdist
    from mof_embeddings.matching.fgw_atom_mapping import fgw, neighbour_kernel, round_plan, supercell_matrix
    F, st, restarts = _W["F"], _W["st"], _W["restarts"]
    frames = pd.read_csv(f"{DATA}/{d}/frames.csv")
    structures = read(f"{DATA}/{d}/structures.extxyz", f"{k}:{k + 2}")
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=8, attempt_supercell=True, primitive_cell=False)
    small = AseAtomsAdaptor.get_structure(structures[0]); large = AseAtomsAdaptor.get_structure(structures[1])
    small.add_site_property("orig", list(range(len(small))))
    aligned = matcher.get_s2_like_s1(large, small)
    truth = np.array([s.properties["orig"] for s in aligned])
    S, _ = supercell_matrix(small.lattice, large.lattice)
    tiled = small.copy(); tiled.make_supercell(S)
    orig = np.array([s.properties["orig"] for s in tiled])
    nn = np.where(np.eye(len(small), dtype=bool), np.inf, small.distance_matrix).min(1)
    sigma = np.median(nn)
    C1, C2 = neighbour_kernel(tiled, sigma, "periodic"), neighbour_kernel(large, sigma, "periodic")
    scale = max(C1.max(), C2.max()); C1, C2 = C1 / scale, C2 / scale
    D1, D2, cutoff = tiled.distance_matrix, large.distance_matrix, 1.3 * sigma
    pairs = (D2 < cutoff) & ~np.eye(len(large), dtype=bool)
    zt = np.array([s.specie.Z for s in tiled]); zl = np.array([s.specie.Z for s in large]); zs = np.array([s.specie.Z for s in small])
    rows = []
    for v in VARIANTS:
        mu, sd = st[v]
        a = (F.get(v, frames.frame[k]) - mu) / sd
        b = (F.get(v, frames.frame[k + 1]) - mu) / sd
        M = cdist(a[orig], b, "sqeuclidean"); M /= M.max()
        for method, alpha in [("OT", 0.0), ("FGW", 0.25)]:
            plan = fgw(M, C1, C2, alpha, zt, zl, restarts, np.random.default_rng(0))
            tiled_of = round_plan(plan, 1)
            m = orig[tiled_of]
            d1 = D1[tiled_of[:, None], tiled_of[None, :]][pairs]
            rows.append(dict(material=frames.material[k], set=d, variant=v, method=method,
                             same_element=float(np.mean(zs[m] == zs[truth])), exact=float(np.mean(m == truth)),
                             bonds_kept=float(np.mean(d1 < cutoff))))
    return rows


def match(F, st, restarts, procs):
    from multiprocessing import get_context
    use_local_packages()
    _W.update(F=F, st=st, restarts=restarts)
    tasks = [(d, k) for d in ["similar_structures", "similar_structures_heldout"]
             for k in range(0, len(pd.read_csv(f"{DATA}/{d}/frames.csv")), 2)]
    with get_context("fork").Pool(procs) as pool:
        res = pool.map(match_pair, tasks, chunksize=1)
    return pd.DataFrame([r for rows in res for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "cluster", "match", "all", "check+cluster"], default="all")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--procs", type=int, default=16, help="worker processes for the match stage")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    F = Features()
    if args.stage in ("check", "all", "check+cluster"):
        check(F).to_csv(OUT / "check_vs_archive.csv")
    st = stats(F, F.frames[:50000:10]) if args.stage != "check" else None
    if args.stage in ("cluster", "all", "check+cluster"):
        c = cluster(F, st); c.to_csv(OUT / "element_clustering.csv", index=False); print(c.round(3).to_string(index=False))
    if args.stage in ("match", "all"):
        m = match(F, st, args.restarts, args.procs); m.to_csv(OUT / "matching.csv", index=False)
        print(m.groupby(["variant", "method"])[["same_element", "exact", "bonds_kept"]].mean().unstack("method").reindex(VARIANTS).round(3).to_string())


if __name__ == "__main__":
    main()
