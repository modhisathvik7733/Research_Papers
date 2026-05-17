#!/usr/bin/env bash
# 0010 protocol — run on the A100 AFTER setup.sh.
#   bash run_0010.sh smoke      # FIRST: tiny, validates wiring (~minutes). NOT a result.
#   bash run_0010.sh full       # the pre-registered run (real GPU hours, real $)
# Stage1 dense retrofit is shared across arms; stage2 sparse differs only by
# optimiser. Honest prior (README): smf-localpc most likely == smf-adam here.
set -euo pipefail

MODE="${1:-smoke}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMF_DIR="${SMF_DIR:-$HERE/SMF-ICML}"
OUT="${OUT:-$HERE/out_${MODE}}"
cd "$SMF_DIR"
mkdir -p "$OUT"

# --- A100-80GB throughput knobs (all verified train_sparse args) ----------
# 0.5B base, only ~44M trainable (sparse) -> activations dominate; bs is
# very safe to push. WATCH `nvidia-smi`: if VRAM < ~60GB and util < ~85%,
# raise PD_TRAIN/PD_EVAL (2x at a time). bf16 + TF32 (sitecustomize) on.
PD_TRAIN="${PD_TRAIN:-128}"      # per-device train batch (try 256 if headroom)
PD_EVAL="${PD_EVAL:-256}"        # eval is forward-only -> larger
GA="${GA:-1}"                    # no accumulation: keep the GPU continuously fed
NPROC="${NPROC:-16}"             # CPU workers for tokenize/filter/background-DF
DLW="${DLW:-8}"                  # dataloader workers
BG_BS="${BG_BS:-64}"            # *** the big one: background-DF was bs=1 ***
PERF="--per-device-train-batch-size $PD_TRAIN \
--per-device-eval-batch-size $PD_EVAL --gradient-accumulation-steps $GA \
--dataloader-num-workers $DLW --num-proc $NPROC \
--background-batch-size $BG_BS --dtype bf16"

if [ "$MODE" = "smoke" ]; then
  SEEDS="0"; RETRO_N="--sample-size 1500 --eval-sample-size 200 --dtype bf16"
  # smoke MUST also shrink the TF-IDF background-DF pass (else ~25 min/arm).
  SPARSE_N="--sample-size 400 --eval-sample-size 150 --num-train-epochs 1 \
--background-sample-size 400 --background-max-batches 40 $PERF"
  EVAL_N="--limit 40 --dtype bf16"
elif [ "$MODE" = "full" ]; then
  SEEDS="0 1 2"                       # >=3 pre-registered; raise if budget allows
  RETRO_N="--dtype bf16"
  SPARSE_N="--num-train-epochs 2 $PERF"   # real background-DF, but bs=64 -> fast
  EVAL_N="--limit 500 --dtype bf16"
else
  echo "MODE must be smoke|full"; exit 1
fi

# ---- stage 1: dense retrofit (ONCE, shared by all arms) -------------------
RETRO="$OUT/retrofit"
if [ ! -f "$RETRO/memory_config.json" ]; then
  echo "=== stage1 dense retrofit ==="
  python -m smf_rebuild.train_retrofit --output-dir "$RETRO" $RETRO_N
fi

# ---- stage 2: sparse phase, per optimiser x seed -------------------------
for SEED in $SEEDS; do
  for OPT in adamw localpc; do
    SP="$OUT/sparse_${OPT}_s${SEED}"
    echo "=== stage2 sparse opt=$OPT seed=$SEED ==="
    python smf_localpc.py --optimizer "$OPT" -- \
      --init-checkpoint "$RETRO" --output-dir "$SP" \
      --dataset-preset medmcqa --sparse-scoring tfidf \
      --seed "$SEED" $SPARSE_N
    echo "=== eval opt=$OPT seed=$SEED ==="
    python -m smf_rebuild.eval_tasks \
      --memory-checkpoint "$SP" \
      --memory-config "$SP/memory_config.json" \
      --tasks medmcqa,wikitext,triviaqa \
      --output "$OUT/eval_${OPT}_s${SEED}.json" $EVAL_N
  done
done

echo "DONE ($MODE). Analyse: python $HERE/analyze.py $OUT"
echo "GPU tip: in another shell run  watch -n1 nvidia-smi  — if VRAM<60GB"
echo " and util<85% during sparse training, re-run with: PD_TRAIN=256 \\"
echo " PD_EVAL=512 bash run_0010.sh $MODE  (push until ~70-75GB used)."
echo "(smoke = wiring validation only, NOT a result; run 'full' for the"
echo " pre-registered verdict, then analyze.py applies the paired rule.)"
