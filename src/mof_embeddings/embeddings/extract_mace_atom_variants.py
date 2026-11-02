#!/usr/bin/env python
"""Per-atom MACE invariant descriptors for a list of OMat24 sr_1m frames.

  mh1    mace-mh-1, omat_pbe head: invariant scalars of both interaction layers
         (512 + 512 = 1,024 per atom; the archive only stores their Z-weighted
         sum per structure as MH1-1024)
  omat0  mace-omat-0-medium: invariant scalars of both layers (128 + 128 = 256;
         the archive's OMAT0-128 is the last layer, summed per structure)

Descriptors come from mace's own MACECalculator.get_descriptors (invariants only,
all layers), computed in the model's float64 and stored as float32. Atom order
is the sr_1m LMDB order, which matches the archive's sr_1m atom banks.

Outputs in --out-dir: <model>_atoms.npy (atoms x features), <model>_offsets.npy,
frames.npy and <model>_meta.json. Work is saved in blocks of --block frames under
<out-dir>/blocks_<model>/ so a run cut short by --max-seconds (or the job's time
limit) resumes where it stopped; the final files are written once every block
exists. Run with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
(this environment's e3nn cannot import under torch's weights-only default).
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from ase.db import connect

from mof_embeddings.config import PRETRAIN

warnings.filterwarnings("ignore")
MODELS = {"mh1": ("mace-mh-1.model", "omat_pbe"), "omat0": ("mace-omat-0-medium.model", None)}
PRETRAINED = Path(f"{PRETRAIN}/models/pretrained")
SR1M_LMDB = f"{PRETRAIN}/data/processed/omat24/stratified_random_nested/sr_1m/train.lmdb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--frames", type=Path, required=True, help=".npy of sr_1m frame indices")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block", type=int, default=2500)
    parser.add_argument("--max-seconds", type=float, default=float("inf"), help="stop after the block that passes this")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from mace.calculators import MACECalculator

    fname, head = MODELS[args.model]
    kw = dict(model_paths=str(PRETRAINED / fname), device=args.device, default_dtype="float64")
    if head:
        kw["head"] = head
    calc = MACECalculator(**kw)
    frames = np.load(args.frames)
    db = connect(SR1M_LMDB, type="aselmdb", readonly=True)

    blocks = args.out_dir / f"blocks_{args.model}"; blocks.mkdir(exist_ok=True)
    starts = range(0, len(frames), args.block)
    t0 = time.time()
    for s in starts:
        out = blocks / f"{s:07d}.npz"
        if out.exists():
            continue
        if time.time() - t0 > args.max_seconds:
            print(f"{args.model}: stopping at frame {s} after {time.time() - t0:.0f}s; resubmit to continue", flush=True)
            return
        feats, counts, numbers = [], [], []
        for f in frames[s:s + args.block]:
            at = db.get_atoms(id=int(f) + 1)
            d = calc.get_descriptors(at, invariants_only=True, num_layers=-1)
            feats.append(np.asarray(d, dtype=np.float32)); counts.append(len(at)); numbers.append(at.numbers)
        np.savez(blocks / "tmp.npz", X=np.concatenate(feats), counts=counts, numbers=np.concatenate(numbers).astype(np.int16))
        (blocks / "tmp.npz").rename(out)
        print(f"{args.model}: {s + len(counts)}/{len(frames)} in {time.time() - t0:.0f}s", flush=True)
    parts = [np.load(blocks / f"{s:07d}.npz") for s in starts]
    X = np.concatenate([b["X"] for b in parts]); counts = np.concatenate([b["counts"] for b in parts])
    np.save(args.out_dir / f"{args.model}_atoms.npy", X)
    np.save(args.out_dir / f"{args.model}_offsets.npy", np.concatenate([[0], np.cumsum(counts)]))
    np.save(args.out_dir / f"{args.model}_atomic_numbers.npy", np.concatenate([b["numbers"] for b in parts]))
    np.save(args.out_dir / "frames.npy", frames)
    (args.out_dir / f"{args.model}_meta.json").write_text(json.dumps(dict(
        model=fname, head=head or "default", features=int(X.shape[1]), atoms=int(X.shape[0]), frames=int(len(frames)),
        layout="invariant scalars of interaction layer 1 then layer 2 (mace get_descriptors, invariants_only)"), indent=2))
    print(f"done: {X.shape}")


if __name__ == "__main__":
    main()
