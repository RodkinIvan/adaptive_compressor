#!/usr/bin/env bash

set -euo pipefail

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LEVELS="${NUM_LEVELS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_STEPS="${MAX_STEPS:-10000}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-100}"
MAX_DOCUMENTS="${MAX_DOCUMENTS:-5000}"
VAL_MAX_DOCUMENTS="${VAL_MAX_DOCUMENTS:-1000}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-adaptive_compressor}"
OUTPUT="${OUTPUT:-checkpoints/baseline.pt}"

CUDA_PREFIX=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_PREFIX=(env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

"${CUDA_PREFIX[@]}" python -m adaptive_compressor.train \
  --model-type baseline \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-levels "${NUM_LEVELS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-steps "${MAX_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --max-documents "${MAX_DOCUMENTS}" \
  --val-max-documents "${VAL_MAX_DOCUMENTS}" \
  --val-max-batches "${VAL_MAX_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output "${OUTPUT}" \
  "$@"
