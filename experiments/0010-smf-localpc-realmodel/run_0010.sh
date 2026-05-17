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

# --- A100 throughput knobs (verified train_sparse args) -------------------
# MEMORY MODEL (the real bottleneck): Qwen-2.5-0.5B vocab ~= 151,936, so the
# OOM driver is the logits tensor ~ B * S * 151936, plus an fp32 upcast for
# the loss + its grad  =>  ~ B*S*152k*~6 bytes. NOT the 0.5B body. So size
# by B*S, not by param count. Rule of thumb on 80GB (bf16): keep
#   B * S  <~ 90,000   ( ~35GB logits, big headroom; OOM was B*S=131,072 ).
# Tune via nvidia-smi: each +16 train batch @ S=512 ~= +2.5GB. Push
# PD_TRAIN up (×2) until ~70GB used; that is "using the GPU", not OOMing it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MAXLEN="${MAXLEN:-512}"          # seq len: halves logits vs 1024, plenty here
PD_TRAIN="${PD_TRAIN:-64}"       # B*S = 64*512 = 32768  (~0.25x the OOM cfg)
PD_EVAL="${PD_EVAL:-64}"         # same logits cost as train; not "free"
GA="${GA:-1}"
NPROC="${NPROC:-4}"              # box has 4 vCPU — match it (16 oversubscribed)
DLW="${DLW:-4}"
BG_BS="${BG_BS:-64}"             # background-DF batch (was 1 -> the 25min wall)
PERF="--per-device-train-batch-size $PD_TRAIN \
--per-device-eval-batch-size $PD_EVAL --gradient-accumulation-steps $GA \
--dataloader-num-workers $DLW --num-proc $NPROC --max-length $MAXLEN \
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
echo "GPU tip: watch -n1 nvidia-smi during *sparse training*. Safe to push"
echo " (S=512): PD_TRAIN=96 then 128 then 160 -> B*S 49k/65k/82k (<90k cap)."
echo " e.g.  PD_TRAIN=128 PD_EVAL=128 bash run_0010.sh $MODE  ; aim ~70GB."
echo "(smoke = wiring validation only, NOT a result; run 'full' for the"
echo " pre-registered verdict, then analyze.py applies the paired rule.)"
