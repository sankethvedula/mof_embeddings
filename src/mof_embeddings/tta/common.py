"""Paths and the run layout shared by the test-time adaptation modules.

Layout under $SCRATCH/tta/:
  e0s_mace-omat-0-medium.json            per-element E0s of the foundation model
  <target>/target.lmdb, target.csv        every MOF-off PBE frame of the target framework
  <target>/<selection>/train.lmdb, valid.lmdb, sets.csv
  <target>/<selection>/run/               MACE work_dir (config, checkpoints, models, logs)
  <target>/<selection>/eval.csv           per-frame predictions of the fine-tuned model
  <target>/foundation_eval.csv            per-frame predictions of the foundation model
"""
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
SELECTIONS = ["fgw", "random", "global"]
TARGETS = ["Q6", "Q3", "Q4", "Q1"]
N_TRAIN, N_VALID = 20, 5


def target_dir(target):
    return TTA / target


def run_dir(target, selection):
    return TTA / target / selection
