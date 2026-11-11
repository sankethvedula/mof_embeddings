#!/usr/bin/env python
"""Write the mace_run_train config for one (target, selection) fine-tuning run.

Settings follow Aryan's MACE fine-tuning generator (group copy in finetune/aryan_method/, not tracked):
energy/force/stress loss with weights 1/10/10, lr 1e-4, rms_forces_scaling, EMA 0.99, AMSGrad,
float32, E0s of the foundation model. Changes for a 20-structure run: batch size 4, validation
batch 5, 2 data workers, no W&B, explicit seed, single-head fine-tuning (no replay data), and a
checkpoint every epoch, so the target can be scored along training (slurm/tta_finetune.slurm keeps
the scored epochs and deletes the rest). Validation uses the selection's own held-out structures
(ranks 21-25), never the target; the best-validation epoch is the target-blind choice.

  $PY -m mof_embeddings.tta.prepare --target Q6 --selection fgw [--epochs 60] [--lr 1e-4]
Prints the config path; slurm/tta_finetune.slurm runs it.
"""
import argparse
import json

import yaml

from mof_embeddings.tta.common import E0S, FOUNDATION, SELECTIONS, TARGETS, run_dir


def write_config(target, selection, epochs, lr, seed):
    d = run_dir(target, selection)
    work = d / "run"
    work.mkdir(parents=True, exist_ok=True)
    e0 = json.loads(E0S.read_text())["atomic_energies"]
    e0 = {int(k): float(v) for k, v in e0.items()}
    name = f"tta_{target}_{selection}"
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
        "max_num_epochs": epochs, "save_all_checkpoints": True,
        "ema": True, "ema_decay": 0.99, "amsgrad": True,
        "default_dtype": "float32", "device": "cuda", "wandb": False,
    }
    path = work / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--selection", choices=SELECTIONS, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(write_config(args.target, args.selection, args.epochs, args.lr, args.seed))


if __name__ == "__main__":
    main()
