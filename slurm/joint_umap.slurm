#!/bin/bash
# Joint atom UMAP for one model (OMat24 sr_1m + MOF-off, MAD, MP-ALOE, MatPES samples).
#     MODEL=UMA sbatch slurm/joint_umap.slurm ;  MODEL=MH1 sbatch slurm/joint_umap.slurm
#SBATCH --job-name=joint_umap
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=8
"$PY" -B -m mof_embeddings.embeddings.joint_atom_umap --models "${MODEL:?set MODEL}"
