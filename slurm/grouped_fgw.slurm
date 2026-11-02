#!/bin/bash
# Grouped FGW matrices (UMA features, constants of the MOF search). KIND=random or organic.
#     KIND=random sbatch --array=0-19 slurm/grouped_fgw.slurm
#SBATCH --job-name=grouped_fgw
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=1
D="data/grouped_fgw/${KIND:?set KIND}"
"$PY" -B -m mof_embeddings.structure_distance.compute_pairwise_fgw \
    --sample "$D/sample" --out "$D/pairwise" --sigma 3.0 --alphas 0.0 0.5 --features UMA \
    --constants data/mof_search/constants.json \
    --task "$SLURM_ARRAY_TASK_ID" --n-tasks "${N_TASKS:-20}"
