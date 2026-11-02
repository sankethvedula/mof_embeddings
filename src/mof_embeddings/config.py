"""Locations of inputs and outputs.

Defaults match the Della cluster setup; override any of them with environment
variables (MOF_EMB_ROOT, MOF_EMB_ARCHIVE, MOF_EMB_PRETRAIN, MOF_EMB_SCRATCH,
MOF_EMB_PYDEPS).

  ROOT      repository root; DATA and RUNS live under it (not tracked by git)
  ARCHIVE   shared embedding archive (read-only)
  PRETRAIN  Aryan's pretrain_analysis tree: processed structures, pretrained models
  SCRATCH   scratch space for large intermediates (MACE variants, joint UMAPs)
  PYDEPS    directory of locally installed packages (POT), added to sys.path on request
"""
import os
import sys
from pathlib import Path


def _path(var, default):
    return Path(os.environ.get(var, default)).expanduser()


ROOT = _path("MOF_EMB_ROOT", Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ARCHIVE = _path("MOF_EMB_ARCHIVE", "/scratch/gpfs/ROSENGROUP/shared/aryan/saved_embeddings")
PRETRAIN = _path("MOF_EMB_PRETRAIN", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
SCRATCH = _path("MOF_EMB_SCRATCH", "/scratch/gpfs/GUILLERMOS/sv4956")
PYDEPS = _path("MOF_EMB_PYDEPS", ROOT / ".pydeps")


def use_local_packages():
    """Make packages installed with `pip install --target $PYDEPS` (POT) importable."""
    if PYDEPS.is_dir() and str(PYDEPS) not in sys.path:
        sys.path.insert(0, str(PYDEPS))
