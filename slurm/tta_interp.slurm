#!/bin/bash
# Weight interpolation between MACE-OMAT-0 and fine-tuned models (tta.interpolate) on one GPU.
#   TARGETS="Q6 S0 ..." [SELECTION=fgwmid] [LR=3e-5] [WORKERS=6] [ARGS="--lambdas 0.25 0.5 0.75"] \
#       sbatch slurm/tta_interp.slurm
# The targets are split over WORKERS processes sharing the GPU; finished files are skipped, so the job
# can be resubmitted.
#SBATCH --job-name=tta_interp
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=2
read -ra T <<< "${TARGETS:?set TARGETS}"
W=${WORKERS:-6}
for ((w = 0; w < W; w++)); do
    mine=()
    for ((i = w; i < ${#T[@]}; i += W)); do mine+=("${T[i]}"); done
    [ ${#mine[@]} -gt 0 ] || continue
    "$PY_MACE" -B -m mof_embeddings.tta.interpolate --targets "${mine[@]}" --selection "${SELECTION:-fgwmid}" \
        --lr "${LR:-3e-5}" ${ARGS:-} > "logs/tta_interp_${SLURM_JOB_ID}_w$w.log" 2>&1 &
done
fail=0
for p in $(jobs -p); do wait "$p" || fail=1; done
echo "done $(date +%T), failures: $fail"
exit $fail
