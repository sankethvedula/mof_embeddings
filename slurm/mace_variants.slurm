#!/bin/bash
# Per-atom MACE descriptors (MH1 1,024 or OMAT0 256) for 50k sr_1m frames + the benchmark frames.
# Sized for the 1-hour gpu-test QOS: stops after ~50 min and resubmits itself until every block is done;
# the second model to finish submits slurm/mace_variant_eval.slurm.
#     MODEL=mh1 sbatch slurm/mace_variants.slurm ;  MODEL=omat0 sbatch slurm/mace_variants.slurm
#SBATCH --job-name=mace_variants
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
D="${MOF_EMB_SCRATCH:-/scratch/gpfs/GUILLERMOS/sv4956}/mace_variants"
"$PY_MACE" -B -m mof_embeddings.embeddings.extract_mace_atom_variants \
    --model "${MODEL:?set MODEL}" --frames "$D/frame_list.npy" --out-dir "$D" --max-seconds 2900 --block 1000
if [ ! -f "$D/${MODEL}_meta.json" ]; then
    sbatch --export=ALL,MODEL="$MODEL" slurm/mace_variants.slurm
elif [ -f "$D/mh1_meta.json" ] && [ -f "$D/omat0_meta.json" ] && mkdir "$D/.eval_submitted" 2>/dev/null; then
    sbatch slurm/mace_variant_eval.slurm      # mkdir: only the second model to finish submits
fi
