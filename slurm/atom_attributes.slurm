#!/bin/bash
# Per-atom force magnitude, structure size and volume per atom for the joint-UMAP datasets.
#SBATCH --job-name=atom_attributes
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=1
"$PY" -B -m mof_embeddings.embeddings.atom_attributes --procs 16
