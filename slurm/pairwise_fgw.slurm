#!/bin/bash
# Pairwise structure-FGW matrix (100 MOF-off + 1,000 OMat24) as an array; merge with --merge afterwards.
#     sbatch --array=0-49 slurm/pairwise_fgw.slurm
#SBATCH --job-name=pairwise_fgw
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=1
"$PY" -B -m mof_embeddings.structure_distance.compute_pairwise_fgw \
    --sigma 3.0 --alphas 0.0 0.5 --task "$SLURM_ARRAY_TASK_ID" --n-tasks "${N_TASKS:-50}"
