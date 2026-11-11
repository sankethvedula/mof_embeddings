#!/usr/bin/env python
"""Per-frame energy, force and stress predictions of a MACE model on labelled LMDBs.

  $PY_MACE -m mof_embeddings.tta.evaluate --model M.model --lmdb target=PATH [--lmdb valid=PATH] --out eval.csv
  $PY_MACE -m mof_embeddings.tta.evaluate --model M.model --checkpoint run_epoch-5.pt ...   (weights of a
      training checkpoint loaded into the architecture of M.model)

One row per frame: set name, frame index, atoms, reference and predicted total energy,
force sum of absolute / squared component errors (for MAE / RMSE over all components),
mean absolute stress error (eV/A^3, Voigt). Summaries are computed by tta.compare.
"""
import argparse

import numpy as np
import pandas as pd


def model_from_checkpoint(model, checkpoint, out):
    """Write a .model with the architecture of `model` and the weights of a training checkpoint."""
    import torch
    m = torch.load(model, map_location="cpu", weights_only=False)
    m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    torch.save(m, out)
    return out


def evaluate(model, lmdbs, device, dtype):
    from mace.calculators import MACECalculator
    from mace.tools.fairchem_dataset import AseDBDataset
    calc = MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)
    rows = []
    for name, path in lmdbs:
        ds = AseDBDataset(config={"src": str(path)})
        try:
            for k, idx in enumerate(ds.ids):
                a = ds.get_atoms(idx)
                e_ref = a.get_potential_energy(); f_ref = a.get_forces(apply_constraint=False)
                s_ref = a.get_stress(voigt=True, apply_constraint=False)
                b = a.copy(); b.calc = calc
                e = b.get_potential_energy(); f = b.get_forces(); s = b.get_stress(voigt=True)
                df = f - f_ref
                rows.append(dict(set=name, frame=k, natoms=len(a), energy_ref=e_ref, energy_pred=e,
                                 force_abs_sum=float(np.abs(df).sum()), force_sq_sum=float((df ** 2).sum()),
                                 force_ref_rms=float(np.sqrt((f_ref ** 2).mean())),
                                 stress_mae=float(np.abs(s - s_ref).mean())))
        finally:
            del ds
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", help="training checkpoint (.pt) whose weights replace those of --model")
    parser.add_argument("--lmdb", action="append", required=True, help="NAME=PATH, repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    args = parser.parse_args()
    lmdbs = [x.split("=", 1) for x in args.lmdb]
    model = args.model
    if args.checkpoint:
        model = model_from_checkpoint(args.model, args.checkpoint, str(args.checkpoint).replace(".pt", ".model"))
    df = evaluate(model, lmdbs, args.device, args.dtype)
    df.to_csv(args.out, index=False)
    for name, g in df.groupby("set", sort=False):
        n = 3 * g.natoms.sum()
        print(f"{name}: {len(g)} frames  force MAE {g.force_abs_sum.sum() / n:.4f} eV/A  "
              f"RMSE {np.sqrt(g.force_sq_sum.sum() / n):.4f}  "
              f"energy MAE {np.mean(np.abs(g.energy_pred - g.energy_ref) / g.natoms) * 1e3:.1f} meV/atom")


if __name__ == "__main__":
    main()
