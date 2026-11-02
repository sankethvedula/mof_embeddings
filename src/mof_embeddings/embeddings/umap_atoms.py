#!/usr/bin/env python
"""cuML UMAP of the OMat24 sr_1m atom embeddings from UMA (raw128) or
MACE-MH1 (readout96).

UMAP parameters, geometry (raw features, Euclidean, no standardisation or PCA)
and cuML version match Aryan's UMA atomwise UMAP
(derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/fit.json), but the
fit uses the OMat24 sr_1m atoms only. Row i of the output is row i of the
chosen atom bank; both banks share the same atom order.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mof_embeddings.config import ARCHIVE

ARCHIVE = Path(str(ARCHIVE))
UMA_FIT = ARCHIVE / "derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/fit.json"
BANKS = {
    "mh1": dict(dir=ARCHIVE / "embeddings/mace_mh1/atom/omat24_sr1m/readout96", out="umap_mh1_readout96.npy",
                dim=96, check=lambda m: m["descriptor"]["hook"] == "model.readouts[-1].non_linearity forward output"
                and m["complete"]),
    "uma": dict(dir=ARCHIVE / "embeddings/uma/atom/omat24_sr1m", out="umap_uma_raw128.npy",
                dim=128, check=lambda m: (m["model"], m["task_name"], m["hook"]) == (
                    "uma-s-1p2", "omat", "output_heads.energyandforcehead.head.energy_block[-1] input[0]")),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(BANKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Fit only the first N atoms (smoke test)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_info = BANKS[args.model]

    import cuml
    from cuml.manifold import UMAP

    params = json.loads(UMA_FIT.read_text())["params"]
    meta = json.loads((bank_info["dir"] / "atom_latents.metadata.json").read_text())
    if not bank_info["check"](meta):
        raise ValueError(f"Unexpected {args.model} atom bank definition")

    path = bank_info["dir"] / "atom_latents.npy"
    bank = np.load(path, mmap_mode="r")
    if bank.shape[1] != bank_info["dim"]:
        raise ValueError(f"Expected {bank_info['dim']} features, found {bank.shape}")
    n = bank.shape[0] if args.max_rows is None else min(args.max_rows, bank.shape[0])
    x = np.ascontiguousarray(bank[:n], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite {args.model} atom embeddings")

    print(f"cuML {cuml.__version__}: fitting {args.model} {x.shape}; {params}", flush=True)
    start = time.monotonic()
    xy = np.asarray(UMAP(**params).fit_transform(x), dtype=np.float32)
    seconds = time.monotonic() - start
    if xy.shape != (n, 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid UMAP coordinates")

    np.save(args.output_dir / bank_info["out"], xy)
    (args.output_dir / f"fit_{args.model}.json").write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        params=params, cuml_version=cuml.__version__, fit_seconds=seconds, shape=list(xy.shape),
        input=str(path), rows="first %d atom rows of the bank (OMat24 sr_1m only)" % n,
        geometry=f"raw{bank_info['dim']}, Euclidean, no standardization or PCA",
        matched_to=str(UMA_FIT)), indent=2) + "\n")
    print(f"done in {seconds:.0f}s -> {args.output_dir / bank_info['out']}", flush=True)


if __name__ == "__main__":
    main()
