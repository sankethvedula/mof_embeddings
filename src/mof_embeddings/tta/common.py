"""Paths, targets and the run layout shared by the test-time adaptation modules.

Targets are MOF-off structures of the structure-FGW sample (data/structure_fgw/sample), named
  Q1-Q10   the queries of the 10M FGW search (data/mof_search/queries.csv)
  S<idx>   further MOF-off sample structures chosen by tta.targets (data/tta/targets_extra.csv),
           whose candidate pools come from the global vector (tta.select_alpha)

Layout under $SCRATCH/tta/:
  e0s_mace-omat-0-medium.json            per-element E0s of the foundation model
  <target>/target.lmdb, target.csv        every MOF-off PBE frame of the target framework
  <target>/<selection>/train.lmdb, valid.lmdb, sets.csv
                                          selection = method + training-set size: e.g. fgw, global,
                                          random (20 structures) or fgw100 (100 structures)
  <target>/<selection>/run/               MACE work_dir (config, checkpoints, models, logs)
  <target>/<selection>/eval_epoch<E>.csv  per-frame predictions after epoch E
  <target>/<selection>/lr1e-05/, lr3e-05_s7/, s7/, lr3e-05_vro/   the same for a non-default learning
                                          rate, seed and/or training variant (VARIANTS)
  <target>/foundation_eval.csv            per-frame predictions of the foundation model
"""
import re
from pathlib import Path

from mof_embeddings.config import DATA, PRETRAIN, ROOT, SCRATCH

FOUNDATION = ROOT / "finetune" / "mace-omat-0-medium.model"
TTA = Path(f"{SCRATCH}/tta")
E0S = TTA / "e0s_mace-omat-0-medium.json"
MOF_OFF_PBE = Path(f"{PRETRAIN}/data/processed/mof_off/PBE")
MOF_OFF_R2SCAN = Path(f"{PRETRAIN}/data/processed/mof_off/R2SCAN")
SR10M_LMDB = f"{PRETRAIN}/data/processed/omat24/stratified_random_nested/sr_10m/train.lmdb"
SEARCH = DATA / "mof_search"
SAMPLE = DATA / "structure_fgw" / "sample"
EXTRA_TARGETS = DATA / "tta" / "targets_extra.csv"
METHODS = ["fgw", "random", "global", "fgwmid", "fgwlow"]
# FGW weight on geometry for the FGW-ranked selections (the search used 0.5); fgwmid and fgwlow, and
# every FGW selection of an S target, are ranked within a candidate pool by tta.select_alpha
METHOD_ALPHA = {"fgw": 0.5, "fgwmid": 0.25, "fgwlow": 0.1}
PILOT = ["Q6", "Q3", "Q4", "Q1"]          # queries with meaningful FGW neighbours (first round)
SEARCH_TARGETS = PILOT + ["Q2", "Q5", "Q7", "Q8", "Q9", "Q10"]
N_TRAIN, N_VALID = 20, 5
LR = 1e-4                                 # default learning rate (Aryan's setting)
SEED = 123                                # default seed
# Training variants (mace_run_train settings on top of tta.prepare's); "" is the default protocol.
VARIANTS = {
    "": dict(desc="all weights trained", cfg={}),
    "ro": dict(desc="readouts only (embedding, interactions, products frozen)", cfg={"freeze": 6}, epochs=30),
    "pr": dict(desc="products and readouts (embedding, interactions frozen)", cfg={"freeze": 5}, epochs=30),
    "lora": dict(desc="LoRA rank 4 on all layers, merged after training",
                 cfg={"lora": True, "lora_rank": 4, "lora_alpha": 1.0}, epochs=30),
    "sw": dict(desc="stress weight 1000 instead of 10", cfg={"stress_weight": 1000}, epochs=30),
}


def extra_targets():
    import pandas as pd
    return pd.read_csv(EXTRA_TARGETS)["query"].tolist() if EXTRA_TARGETS.exists() else []


TARGETS = SEARCH_TARGETS + extra_targets()


def targets_table():
    """query, idx (row of the structure-FGW sample), formula, natoms, metal for every target."""
    import pandas as pd
    t = pd.read_csv(SEARCH / "queries.csv")
    if EXTRA_TARGETS.exists():
        t = pd.concat([t, pd.read_csv(EXTRA_TARGETS)], ignore_index=True)
    return t


def target_query(target, mu, sd):
    """z-scored UMA atom features, atomic numbers and structure of a target's query frame."""
    import numpy as np
    from ase.io import read
    idx = int(targets_table().set_index("query").idx[target])
    emb = np.load(SAMPLE / "embeddings.npz")
    off = emb["frame_offsets"]
    x = (emb["atom_uma128"][off[idx]:off[idx + 1]] - mu) / sd
    z = emb["atomic_numbers"][off[idx]:off[idx + 1]]
    atoms = read(SAMPLE / "structures.extxyz", index=idx)
    assert (atoms.numbers == z).all()
    return x, z, atoms


def selection_name(method, n_train=N_TRAIN):
    return method if n_train == N_TRAIN else f"{method}{n_train}"


def parse_selection(selection):
    """'fgw' -> ('fgw', 20); 'global100' -> ('global', 100)."""
    m = re.fullmatch(r"({})(\d*)".format("|".join(sorted(METHODS, key=len, reverse=True))), selection)
    if m is None:
        raise ValueError(f"unknown selection {selection!r}")
    return m.group(1), int(m.group(2) or N_TRAIN)


def target_dir(target):
    return TTA / target


def run_dir(target, selection):
    """Training and validation databases of a selection (shared by all learning rates and seeds)."""
    return TTA / target / selection


def out_dir(target, selection, lr=LR, seed=SEED, variant=""):
    """MACE work_dir parent and scored epochs of one run; non-default learning rate / seed / variant in a
    subfolder."""
    parts = ([f"lr{lr:g}"] if lr != LR else []) + ([f"s{seed}"] if seed != SEED else []) + \
            ([f"v{variant}"] if variant else [])
    return run_dir(target, selection) / "_".join(parts) if parts else run_dir(target, selection)


def run_params(path):
    """(learning rate, seed, variant) of a run directory made by out_dir."""
    lr, seed, variant = LR, SEED, ""
    if re.fullmatch(r"(lr[0-9.e+-]+)?(_?s\d+)?(_?v[a-z0-9]+)?", path.name):
        for tok in path.name.split("_"):
            if tok.startswith("lr"):
                lr = float(tok[2:])
            elif tok.startswith("s"):
                seed = int(tok[1:])
            elif tok.startswith("v"):
                variant = tok[1:]
    return lr, seed, variant
