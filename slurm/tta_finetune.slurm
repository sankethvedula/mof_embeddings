#!/bin/bash
# Test-time adaptation run: fine-tune MACE-OMAT-0 on one selection's structures for one target,
# then evaluate the fine-tuned model on the target's MOF-off PBE frames and the validation set.
#     TARGET=Q6 SELECTION=fgw sbatch slurm/tta_finetune.slurm          (fgw, random, global: 20 structures;
#                                                                       fgw100, random100, global100: 100)
#     TARGET=Q6 SELECTION=foundation sbatch slurm/tta_finetune.slurm   (evaluate the foundation model on the
#                                                                       target and every selection's sets)
#     LR=1e-5 TARGET=Q6 SELECTION=fgw sbatch ...    (learning rate other than 1e-4: outputs in $SEL/lr1e-05/)
#     SEED=7 LR=3e-5 TARGET=Q6 SELECTION=fgw sbatch ...   (seed other than 123: outputs in $SEL/lr3e-05_s7/)
#     VARIANT=ro LR=3e-5 TARGET=Q6 SELECTION=fgw sbatch ... (training variant, tta.common.VARIANTS: $SEL/lr3e-05_vro/)
#SBATCH --job-name=tta
#SBATCH --account=guillermos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
export OMP_NUM_THREADS=${TTA_THREADS:-4} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MACE_BIN="$(dirname "$PY_MACE")"
T="${MOF_EMB_SCRATCH:-/scratch/gpfs/GUILLERMOS/sv4956}/tta/${TARGET:?set TARGET}"
SEL="${SELECTION:?set SELECTION}"
if [ "$SEL" = foundation ]; then
    "$PY_MACE" -B -m mof_embeddings.tta.evaluate --model finetune/mace-omat-0-medium.model \
        --lmdb "target=$T/target.lmdb" \
        $(for d in "$T"/*/train.lmdb; do s=$(basename "$(dirname "$d")"); echo "--lmdb valid_$s=$T/$s/valid.lmdb --lmdb train_$s=$d"; done) \
        --out "$T/foundation_eval.csv"
    exit 0
fi
PREP=(--target "$TARGET" --selection "$SEL" --epochs "${EPOCHS:-60}" --lr "${LR:-1e-4}" --seed "${SEED:-123}"
      --variant "${VARIANT:-}")
CONFIG=$("$PY" -B -m mof_embeddings.tta.prepare "${PREP[@]}")
LAST=$(( $("$PY" -B -m mof_embeddings.tta.prepare "${PREP[@]}" --print-epochs) - 1 ))
OUT="$(dirname "$(dirname "$CONFIG")")"
cd "$(dirname "$CONFIG")"
"$MACE_BIN/mace_run_train" --config="$CONFIG"
cd "$SLURM_SUBMIT_DIR"
MODEL=$(find "$(dirname "$CONFIG")" -name "$(basename "$CONFIG" .yaml)*.model" ! -name "*compiled*" -newer "$CONFIG" | head -n 1)
CKPT="$(dirname "$CONFIG")/checkpoints"
for E in $(printf '%s\n' ${EVAL_EPOCHS:-0 1 2 5 10 20 40 59} "$LAST" | sort -nu); do   # always score the last epoch
    [ "$E" -le "$LAST" ] || continue
    C=$(ls "$CKPT"/*_epoch-"$E".pt 2>/dev/null | head -n 1) || true
    [ -n "$C" ] || continue
    "$PY_MACE" -B -m mof_embeddings.tta.evaluate --model "$MODEL" --checkpoint "$C" \
        --lmdb "target=$T/target.lmdb" --lmdb "valid=$T/$SEL/valid.lmdb" --lmdb "train=$T/$SEL/train.lmdb" \
        --out "$OUT/eval_epoch$E.csv"
done
find "$CKPT" -name "*.pt" -delete                    # scored epochs are kept as .model files
