#!/bin/bash
# MOF -> OMat24 sr_10m search. STAGE=prefilter or rerank; run as an array, then merge.
#     STAGE=prefilter sbatch --array=0-249 slurm/mof_search.slurm
#     STAGE=rerank    sbatch --array=0-99  slurm/mof_search.slurm
#SBATCH --job-name=mof_search
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=1
case "${STAGE:?set STAGE}" in
  prefilter) "$PY" -B -m mof_embeddings.mof_search.prefilter --task "$SLURM_ARRAY_TASK_ID" --n-tasks "${N_TASKS:-250}" ;;
  rerank)    "$PY" -B -m mof_embeddings.mof_search.rerank --task "$SLURM_ARRAY_TASK_ID" --n-tasks "${N_TASKS:-100}" ;;
esac
