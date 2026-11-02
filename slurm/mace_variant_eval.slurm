#!/bin/bash
# Archive check, element clustering (GPU), matching (16 CPU workers), then per-variant UMAPs (GPU).
#SBATCH --job-name=mace_variant_eval
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=16
"$PY" -B -m mof_embeddings.embeddings.evaluate_mace_variants --stage all --procs 16
"$PY" -B -m mof_embeddings.embeddings.umap_variants
