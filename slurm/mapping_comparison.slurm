#!/bin/bash
# Plain OT, FGW (all-images periodic kernel) and embedding-only GW on a set of small-cell / supercell pairs.
#     DATA_DIR=data/similar_structures_heldout sbatch slurm/mapping_comparison.slurm
#SBATCH --job-name=mapping_compare
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
DATA_DIR="${DATA_DIR:?set DATA_DIR}"
"$PY" -B -m mof_embeddings.matching.ot_atom_mapping --data-dir "$DATA_DIR"
"$PY" -B -m mof_embeddings.matching.fgw_atom_mapping --data-dir "$DATA_DIR" --kernel periodic
"$PY" -B -m mof_embeddings.matching.gw_embedding_mapping --data-dir "$DATA_DIR"
