#!/usr/bin/env python
"""Joint atom UMAPs of OMat24 sr_1m plus the external datasets that have atom
embeddings for both UMA and MACE-MH1 (MOF-off R2SCAN, MAD, MP-ALOE, MatPES 10%
samples), one fit per model with the settings of the earlier atom UMAPs (raw
features, Euclidean). AM is left out because it has no MH1 atom bank; MAD atoms
whose MH1 features are not finite (unsupported elements) are dropped from both.

Writes to --out-dir: rows.npz (dataset index, source atom row, source frame per
joint row, and the dataset names), umap_<model>.npy, fit_<model>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from mof_embeddings.config import ARCHIVE, SCRATCH

A = Path(f"{ARCHIVE}/embeddings")
UMA_FIT = Path(f"{ARCHIVE}/derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/fit.json")
DATASETS = [("OMat24", "omat24_sr1m"), ("MOF-off", "external_subsets/mof_off_r2scan"), ("MAD", "external_subsets/mad"),
            ("MP-ALOE", "external_subsets/mpaloe"), ("MatPES", "external_subsets/matpes_r2scan")]
BANK = {"UMA": lambda d: A / "uma/atom" / d, "MH1": lambda d: A / "mace_mh1/atom" / d / "readout96"}


def rows():
    """Joint row order: each dataset's atom bank in order, minus atoms without finite MH1 features."""
    ds, src, frame = [], [], []
    for k, (_, d) in enumerate(DATASETS):
        mh1 = np.load(BANK["MH1"](d) / "atom_latents.npy", mmap_mode="r")
        ok = np.ones(len(mh1), bool)
        for s in range(0, len(mh1), 2_000_000):
            ok[s:s + 2_000_000] = np.isfinite(np.asarray(mh1[s:s + 2_000_000])).all(1)
        fr = np.load(BANK["UMA"](d) / "atom_frame_index.npy")    # sr_1m frame, or frame in the external source file
        idx = np.flatnonzero(ok)
        ds.append(np.full(len(idx), k, np.int8)); src.append(idx); frame.append(fr[idx])
        print(f"{DATASETS[k][0]}: {len(idx):,} atoms ({(~ok).sum()} dropped)", flush=True)
    return np.concatenate(ds), np.concatenate(src), np.concatenate(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(f"{SCRATCH}/joint_umap"))
    parser.add_argument("--models", nargs="+", default=["UMA", "MH1"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds, src, frame = rows()
    tmp = args.out_dir / f"rows.{'_'.join(args.models)}.tmp.npz"            # jobs for different models may run together
    np.savez(tmp, dataset=ds, source_row=src, source_frame=frame, names=np.array([n for n, _ in DATASETS]))
    tmp.replace(args.out_dir / "rows.npz")
    from cuml.manifold import UMAP
    params = json.loads(UMA_FIT.read_text())["params"]
    for model in args.models:
        X = np.concatenate([np.asarray(np.load(BANK[model](d) / "atom_latents.npy", mmap_mode="r")[src[ds == k]], dtype=np.float32)
                            for k, (_, d) in enumerate(DATASETS)])
        t = time.time()
        xy = np.asarray(UMAP(**params).fit_transform(X), dtype=np.float32)
        assert xy.shape == (len(X), 2) and np.isfinite(xy).all()
        np.save(args.out_dir / f"umap_{model}.npy", xy)
        (args.out_dir / f"fit_{model}.json").write_text(json.dumps(dict(params=params, shape=list(X.shape), seconds=time.time() - t,
            datasets=[n for n, _ in DATASETS], geometry="raw features, Euclidean"), indent=2))
        print(f"{model}: {X.shape} fitted in {time.time() - t:.0f}s", flush=True)
        del X


if __name__ == "__main__":
    main()
