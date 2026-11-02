#!/bin/bash
# Pooled global UMA vectors (Z-weighted mean, mean) for all 10M sr_10m frames and their distances to the MOF queries.
#     sbatch slurm/global_pool.slurm
#SBATCH --job-name=global_pool
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=1
"$PY" -B -m mof_embeddings.mof_search.global_vs_fgw --stage pool --procs 16 --chunks 400
