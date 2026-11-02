#!/bin/bash
# Run one module on a CPU node:  sbatch slurm/run_python.slurm -m mof_embeddings.<subpackage>.<module> [args...]
#SBATCH --job-name=python
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PY" -B "$@"
