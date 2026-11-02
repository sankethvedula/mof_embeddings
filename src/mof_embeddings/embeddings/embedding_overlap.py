#!/usr/bin/env python
"""Do MOF-off atoms have close counterparts among OMat24 atoms in embedding space?

For UMA-128 and MH1-96 atom embeddings (z-scored with OMat24 statistics):
  reference  --n-ref random OMat24 sr_1m atoms
  queries    --n-query OMat24 atoms from structures NOT in the reference (the
             in-distribution baseline), and --n-query atoms of the MOF-off R2SCAN
             10% atom sample
For each query: Euclidean distance to its nearest reference atom, and that atom's
element. A MOF atom counts as "outside OMat24" when its nearest-neighbour
distance exceeds the 99th percentile of the OMat24 queries' distances.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element

from mof_embeddings.config import ARCHIVE, DATA
from mof_embeddings.matching.ot_atom_mapping import bank_stats

A = Path(f"{ARCHIVE}/embeddings")
BANKS = {
    "UMA": (A / "uma/atom/omat24_sr1m", A / "uma/atom/external_subsets/mof_off_r2scan"),
    "MH1": (A / "mace_mh1/atom/omat24_sr1m/readout96", A / "mace_mh1/atom/external_subsets/mof_off_r2scan/readout96"),
}


def load_rows(bank_dir, rows):
    x = np.load(bank_dir / "atom_latents.npy", mmap_mode="r")
    return np.asarray(x[np.sort(rows)], dtype=np.float32), np.sort(rows)


def nearest(queries, reference, chunk=512):
    q, r = torch.from_numpy(queries), torch.from_numpy(reference)
    dist, idx = [], []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], r)
        m = d.min(1)
        dist.append(m.values); idx.append(m.indices)
    return torch.cat(dist).numpy(), torch.cat(idx).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ref", type=int, default=1_000_000)
    parser.add_argument("--n-query", type=int, default=20_000)
    parser.add_argument("--out-dir", type=Path, default=Path(f"{DATA}/overlap"))
    args = parser.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))
    rng = np.random.default_rng(0)

    omat_dir = BANKS["UMA"][0]
    off = np.load(omat_dir / "frame_atom_offsets.npy")
    z_omat = np.load(omat_dir / "atom_atomic_numbers.npy", mmap_mode="r")
    frame_of = np.load(omat_dir / "atom_frame_index.npy", mmap_mode="r")
    query_frames = rng.choice(len(off) - 1, 5_000, replace=False)       # held out from the reference
    in_query = np.zeros(len(off) - 1, bool); in_query[query_frames] = True
    q_pool = np.concatenate([np.arange(off[f], off[f + 1]) for f in query_frames])
    q_omat = rng.choice(q_pool, min(args.n_query, len(q_pool)), replace=False)
    ref = np.sort(rng.choice(len(z_omat), args.n_ref + 200_000, replace=False))
    ref = ref[~in_query[np.asarray(frame_of[ref])]]                    # drop atoms of query structures
    ref = rng.choice(ref, args.n_ref, replace=False)

    mof_dir = BANKS["UMA"][1]
    z_mof = np.load(mof_dir / "atom_atomic_numbers.npy")
    q_mof = rng.choice(len(z_mof), args.n_query, replace=False)

    rows, summary = [], {}
    for model, (o_dir, m_dir) in BANKS.items():
        assert np.array_equal(np.load(m_dir / "atom_atomic_numbers.npy"), z_mof)
        mu, sd = bank_stats(o_dir / "atom_latents.npy")
        ref_x, ref_rows = load_rows(o_dir, ref)
        ref_x = ((ref_x - mu) / sd).astype(np.float32)
        z_ref = np.asarray(z_omat[ref_rows])
        for name, bank_dir, q, zq in [("OMat24", o_dir, q_omat, z_omat), ("MOF-off", m_dir, q_mof, z_mof)]:
            x, rows_sorted = load_rows(bank_dir, q)
            x = ((x - mu) / sd).astype(np.float32)
            dist, idx = nearest(x, ref_x)
            rows.append(pd.DataFrame(dict(model=model, dataset=name, element=np.asarray(zq[rows_sorted]),
                                          nn_dist=dist, nn_element=z_ref[idx])))
        print(f"{model} done", flush=True)

    df = pd.concat(rows)
    df["same_element_nn"] = df.element == df.nn_element
    df.to_parquet(args.out_dir / "nn_distances.parquet")
    for model, g in df.groupby("model"):
        base = g[g.dataset == "OMat24"].nn_dist
        thr = np.percentile(base, 99)
        mof = g[g.dataset == "MOF-off"]
        per_el = mof.groupby("element").agg(n=("nn_dist", "size"), median_nn=("nn_dist", "median"),
                                            outside=("nn_dist", lambda d: (d > thr).mean()),
                                            same_element_nn=("same_element_nn", "mean")).sort_values("n", ascending=False)
        per_el.index = [Element.from_Z(int(z)).symbol for z in per_el.index]
        summary[model] = dict(
            omat_median_nn=float(base.median()), threshold_p99=float(thr),
            mof_median_nn=float(mof.nn_dist.median()), mof_outside_fraction=float((mof.nn_dist > thr).mean()),
            omat_same_element_nn=float(g[g.dataset == "OMat24"].same_element_nn.mean()),
            mof_same_element_nn=float(mof.same_element_nn.mean()),
            per_element=per_el.head(10).round(3).reset_index().rename(columns={"index": "element"}).to_dict("records"))
        print(f"\n== {model}: OMat24→OMat24 median NN {base.median():.2f}, 99th pct {thr:.2f} | "
              f"MOF-off→OMat24 median NN {mof.nn_dist.median():.2f}, beyond OMat24 99th pct: {(mof.nn_dist > thr).mean():.1%} | "
              f"NN same element: OMat24 {summary[model]['omat_same_element_nn']:.1%}, MOF {summary[model]['mof_same_element_nn']:.1%}")
        print(per_el.head(10).round(3).to_string())
    (args.out_dir / "embedding_overlap_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
