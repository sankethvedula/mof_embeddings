#!/bin/bash
# cuML UMAP of the OMat24 sr_1m atom embeddings (18.7M atoms) for one model, then the element comparison figure.
#     MODEL=uma sbatch slurm/umap_atoms.slurm ;  MODEL=mh1 sbatch slurm/umap_atoms.slurm
#     MODEL=uma MAX_ROWS=1000000 OUT_DIR=runs/atom_umap_test sbatch --time=00:20:00 slurm/umap_atoms.slurm
#SBATCH --job-name=atom_umap
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
MODEL="${MODEL:?set MODEL=uma or MODEL=mh1}"
OUT_DIR="${OUT_DIR:-runs/atom_umap_sr1m}"
extra=()
[ -n "${MAX_ROWS:-}" ] && extra+=(--max-rows "$MAX_ROWS")
"$PY" -B -m mof_embeddings.embeddings.umap_atoms --model "$MODEL" --output-dir "$OUT_DIR" "${extra[@]}"
"$PY" -B -m mof_embeddings.embeddings.plot_umap_comparison --run-dir "$OUT_DIR"
