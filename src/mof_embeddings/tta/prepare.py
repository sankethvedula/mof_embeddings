#!/usr/bin/env python
"""Write the mace_run_train config for one (target, selection) fine-tuning run.

Settings follow Aryan's MACE fine-tuning generator (group copy in finetune/aryan_method/, not tracked):
energy/force/stress loss with weights 1/10/10, lr 1e-4, rms_forces_scaling, EMA 0.99, AMSGrad,
float32, E0s of the foundation model. Changes for small training sets (20-100 structures): batch size 4, validation
batch 5, 2 data workers, no W&B, explicit seed, single-head fine-tuning (no replay data), and a
checkpoint every epoch, so the target can be scored along training (slurm/tta_finetune.slurm keeps
the scored epochs and deletes the rest). Validation uses the selection's own held-out structures
(ranks N+1..N+5 for N training structures), never the target; the best-validation epoch is the target-blind choice.

  $PY -m mof_embeddings.tta.prepare --target Q6 --selection fgw|fgw100|... [--epochs 60] [--lr 1e-4]
      [--seed 123] [--variant ro|pr|lora|sw]
Prints the config path; slurm/tta_finetune.slurm runs it. A variant (tta.common.VARIANTS) adds its
mace_run_train settings and sets its own number of epochs.
"""
import argparse
import json

import yaml

from mof_embeddings.tta.common import (E0S, FOUNDATION, LR, SEED, TARGETS, VARIANTS, out_dir, parse_selection,
                                       run_dir, target_dir)


def epochs_of(variant, epochs):
    """A variant's own number of epochs overrides the requested one."""
    return VARIANTS[variant].get("epochs", epochs)


def write_config(target, selection, epochs, lr, seed, variant=""):
    d = run_dir(target, selection)
    work = out_dir(target, selection, lr, seed, variant) / "run"
    work.mkdir(parents=True, exist_ok=True)
    e0 = json.loads(E0S.read_text())["atomic_energies"]
    e0 = {int(k): float(v) for k, v in e0.items()}
    name = f"tta_{target}_{selection}" + ("" if lr == LR else f"_lr{lr:g}") + ("" if seed == SEED else f"_s{seed}") \
        + (f"_v{variant}" if variant else "")
    cfg = {
        "name": name, "work_dir": str(work), "seed": seed,
        "foundation_model": str(FOUNDATION), "restart_latest": False, "multiheads_finetuning": False,
        "train_file": str(d / "train.lmdb"), "valid_file": str(d / "valid.lmdb"),
        "atomic_numbers": str(sorted(e0)), "E0s": str(e0),
        "energy_key": "energy", "forces_key": "forces", "stress_key": "stress",
        "error_table": "PerAtomMAEstressvirials", "compute_stress": True, "loss": "stress",
        "energy_weight": 1, "forces_weight": 10, "stress_weight": 10,
        "lr": lr, "scaling": "rms_forces_scaling",
        "batch_size": 4, "valid_batch_size": 5, "num_workers": 2, "pin_memory": True,
        "max_num_epochs": epochs_of(variant, epochs), "save_all_checkpoints": True,
        "ema": True, "ema_decay": 0.99, "amsgrad": True,
        "default_dtype": "float32", "device": "cuda", "wandb": False,
    }
    cfg.update(VARIANTS[variant]["cfg"])
    path = work / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--selection", help="fgw, random, global, fgwmid, fgwlow, or with a size: fgw100")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=LR, help="non-default values write to <selection>/lr<lr>/")
    parser.add_argument("--seed", type=int, default=SEED, help="non-default values write to <selection>/[lr<lr>_]s<seed>/")
    parser.add_argument("--variant", default="", choices=list(VARIANTS), help="training variant (tta.common.VARIANTS)")
    parser.add_argument("--print-dir", action="store_true", help="only print the run's output directory")
    parser.add_argument("--print-epochs", action="store_true", help="only print the run's number of epochs")
    parser.add_argument("--pending", nargs="+", metavar="T:S[:LR[:SEED[:VARIANT]]]",
                        help="print the runs among these whose last epoch is not scored yet")
    args = parser.parse_args()
    if args.pending:
        for r in args.pending:
            t, s, *rest = r.split(":")
            lr = float(rest[0]) if rest and rest[0] else args.lr
            seed = int(rest[1]) if len(rest) > 1 and rest[1] else args.seed
            variant = rest[2] if len(rest) > 2 else args.variant
            if s == "foundation":
                f = target_dir(t) / "foundation_eval.csv"       # pending unless newer than every set
                sets = list(target_dir(t).glob("*/train.lmdb"))
                if not f.exists() or any(x.stat().st_mtime > f.stat().st_mtime for x in sets):
                    print(r)
            elif not (out_dir(t, s, lr, seed, variant) / f"eval_epoch{epochs_of(variant, args.epochs) - 1}.csv").exists():
                print(r)
        return
    if args.target is None or args.selection is None:
        parser.error("--target and --selection are required")
    parse_selection(args.selection)
    if args.print_dir:
        print(out_dir(args.target, args.selection, args.lr, args.seed, args.variant)); return
    if args.print_epochs:
        print(epochs_of(args.variant, args.epochs)); return
    print(write_config(args.target, args.selection, args.epochs, args.lr, args.seed, args.variant))


if __name__ == "__main__":
    main()
