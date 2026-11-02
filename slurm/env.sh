# Sourced by every job script (and usable interactively): run from the repository root.
#   source slurm/env.sh
# PY runs everything except the MACE descriptor extraction, which needs PY_MACE.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID:-$$}}"
PY="${PY:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python}"
PY_MACE="${PY_MACE:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env_mace/bin/python}"
mkdir -p "$ROOT/logs"
cd "$ROOT"
