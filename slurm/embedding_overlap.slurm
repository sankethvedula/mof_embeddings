#!/bin/bash
# Nearest-neighbour overlap of MOF-off atoms with OMat24 atoms in UMA / MH1 embedding space.
#SBATCH --job-name=emb_overlap
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PY" -B -m mof_embeddings.embeddings.embedding_overlap
