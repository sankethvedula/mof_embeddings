#!/bin/bash
# Run many test-time adaptation runs inside one GPU job, several at a time on the same GPU.
#     RUNS="Q7:fgw Q7:fgwlow:1e-5 Q8:fgw100 Q8:foundation" sbatch slurm/tta_batch.slurm
#     PARALLEL=4 RUNS="..." sbatch --time=00:59:00 slurm/tta_batch.slurm
# Each entry is TARGET:SELECTION[:LR[:SEED[:VARIANT]]] (defaults $LR or 1e-4, $SEED or 123, default protocol). Runs start
# in the order given, up to PARALLEL at a time (default 1), each only when MIN_FREE_GB (default 12) of GPU
# memory is free; a 20-structure run peaks at about 2-10 GB.
# A run whose last epoch is already scored (eval_epoch<EPOCHS-1>.csv) is skipped, so the job can be
# resubmitted or chained (sbatch --dependency=afterany:<id>) after a timeout. No new run starts after
# BUDGET_MIN minutes (default: time limit minus 12 min), so runs are not cut off mid-way.
# SETUP="command" runs first on the same node (e.g. tta.select_alpha and tta.build_sets).
# Each run's output goes to logs/tta_runs/; the job fails if any run failed.
#SBATCH --job-name=tta_batch
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
PARALLEL=${PARALLEL:-1}
MIN_FREE_GB=${MIN_FREE_GB:-12}        # start a run only with this much free GPU memory (a run peaks at ~10 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
START=$(date +%s)
LIMIT_MIN=$(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F'[-:]' '{if (NF==2) m=$1; else if (NF==3) m=$2+60*$1; else m=$3+60*$2+1440*$1; print m}')
BUDGET_MIN=${BUDGET_MIN:-$(( LIMIT_MIN - 12 ))}
if [ -n "${SETUP:-}" ]; then          # e.g. build the training sets on this node before the runs
    echo "== setup: $SETUP ($(date +%T))"; bash -c "$SETUP" || { echo "setup FAILED"; exit 1; }; echo "== setup done $(date +%T)"
fi
PENDING=$("$PY" -B -m mof_embeddings.tta.prepare --pending ${RUNS:?set RUNS} --epochs "${EPOCHS:-60}")
echo "$(echo $PENDING | wc -w) of $(echo $RUNS | wc -w) runs pending; parallel $PARALLEL; new runs until minute $BUDGET_MIN"
mkdir -p logs/tta_runs
declare -A pid_run=()
failed=()

reap_finished() {   # record the outcome of every run that has ended
    local pid rc r
    for pid in "${!pid_run[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"; rc=$?; r=${pid_run[$pid]}; unset "pid_run[$pid]"
        if [ $rc -eq 0 ]; then echo "== $r: done $(date +%T)"; else echo "== $r: FAILED $(date +%T)"; failed+=("$r"); fi
    done
}
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1; }

for r in $PENDING; do
    IFS=: read -r t s lr seed variant <<< "$r"; lr=${lr:-${LR:-1e-4}}; seed=${seed:-${SEED:-123}}
    while :; do
        reap_finished
        n=${#pid_run[@]}
        if [ "$n" -ge "$PARALLEL" ] || { [ "$n" -gt 0 ] && [ "$(gpu_free_mb)" -lt $(( MIN_FREE_GB * 1024 )) ]; }; then
            sleep 15; continue
        fi
        break
    done
    if (( $(date +%s) - START > BUDGET_MIN * 60 )); then echo "== time budget reached, not starting $r and later"; break; fi
    echo "== $r: start $(date +%T)"
    VARIANT=${variant:-} SEED=$seed LR=$lr TARGET=$t SELECTION=$s TTA_THREADS=${TTA_THREADS:-2} \
        bash "$SLURM_SUBMIT_DIR/slurm/tta_finetune.slurm" > "logs/tta_runs/${t}_${s}_lr${lr}_s${seed}${variant:+_v$variant}.log" 2>&1 &
    pid_run[$!]=$r
    sleep 45                                 # let the new run allocate before checking free memory again
done
while [ ${#pid_run[@]} -gt 0 ]; do reap_finished; sleep 15; done
[ ${#failed[@]} -eq 0 ] || { echo "failed runs: ${failed[*]}"; exit 1; }
