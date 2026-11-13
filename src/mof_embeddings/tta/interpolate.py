#!/usr/bin/env python
"""Weight interpolation between MACE-OMAT-0 and a fine-tuned model, scored on the target.

For a finished run (target, selection, lr) and a kept checkpoint epoch E, the model with weights
  w(lambda) = (1 - lambda) w_foundation + lambda w_finetuned(E)
is evaluated on the target's MOF-off PBE frames for each lambda (lambda = 1 is the fine-tuned model,
lambda = 0 the foundation model). Every float tensor is interpolated. The fine-tuned model is saved by
a newer MACE than the foundation file: the same parameters, scale/shift/E0s stored with another shape
and extra boolean *_zeroed flags (kept as fine-tuned); every fine-tuned float tensor has a foundation
twin. Nothing is retrained.

  $PY_MACE -m mof_embeddings.tta.interpolate --targets Q6 S0 ... [--selection fgwmid] [--lr 3e-5]
      [--epochs 5 59] [--lambdas 0.25 0.5 0.75] [--device cuda]
  -> <run dir>/interp/epoch<E>_l<lambda>.csv (per-frame predictions, as tta.evaluate)
Existing files are skipped. tta.effect --interp summarises them. Needs a GPU: the kept models were
saved on one (e3nn code generation).
"""
import argparse
import tempfile
from pathlib import Path

from mof_embeddings.tta.common import FOUNDATION, LR, SEED, TARGETS, out_dir, target_dir

LAMBDAS = [0.25, 0.5, 0.75]
EPOCHS = [5, 59]


def interp_path(target, selection, lr, epoch, lam):
    return out_dir(target, selection, lr) / "interp" / f"epoch{epoch}_l{lam:g}.csv"


def finetuned_model(target, selection, lr, epoch):
    run = out_dir(target, selection, lr) / "run" / "checkpoints"
    return next(run.glob(f"*_run-{SEED}_epoch-{epoch}.model"), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--selection", default="fgwmid")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from mof_embeddings.tta.evaluate import evaluate
    base = torch.load(FOUNDATION, map_location="cpu", weights_only=False).state_dict()
    for t in args.targets:
        for e in args.epochs:
            todo = [lam for lam in args.lambdas if not interp_path(t, args.selection, args.lr, e, lam).exists()]
            ft_path = finetuned_model(t, args.selection, args.lr, e)
            if not todo or ft_path is None:
                continue
            m = torch.load(ft_path, map_location="cpu", weights_only=False)
            ft = {k: v.clone() for k, v in m.state_dict().items()}
            # the fine-tuned model is saved by a newer MACE: same parameters, some stored with another
            # shape (scale, shift, E0s), extra boolean *_zeroed flags; every float tensor has a base twin
            shared = {k for k, v in ft.items() if v.is_floating_point() and k in base and base[k].numel() == v.numel()}
            missing = [k for k, v in ft.items() if v.is_floating_point() and k not in shared]
            assert not missing, f"{t}: no foundation weights for {missing[:3]}"
            for lam in todo:
                m.load_state_dict({k: ((1 - lam) * base[k].reshape(v.shape).to(v.device, v.dtype) + lam * v)
                                   if k in shared else v for k, v in ft.items()})
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "interp.model"
                    torch.save(m, path)
                    df = evaluate(path, [("target", target_dir(t) / "target.lmdb")], args.device, "float64")
                out = interp_path(t, args.selection, args.lr, e, lam)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                print(f"{t} epoch {e} lambda {lam:g}: {len(df)} frames", flush=True)


if __name__ == "__main__":
    main()
