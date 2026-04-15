# Adaptive Compressor

Minimal PyTorch prototype of a hierarchical byte-level language model with adaptive border selection, plus a simple byte-level GRU baseline.

## Core idea

1. Level 0 models raw UTF-8 bytes and promotes positions whose next-byte prediction is wrong.
2. Higher levels model only the promoted border embeddings and promote positions whose next-embedding prediction MSE is above a threshold.
3. Each sequence always starts with a border at position `0`, so every compressed token owns a span in the level below.
4. The decoder mirrors the compressor and reconstructs lower-level sequences by broadcast-adding each parent embedding over its child span.

By default, border selection is causal:
1. Level 0 promotes positions whose byte-prediction entropy is above `--byte-entropy-threshold`.
2. Higher levels promote positions whose predicted next-embedding uncertainty is above `--meta-uncertainty-threshold`.
3. The original teacher-forced border rule is still available with `--border-mode teacher_forced`, but it leaks target information and should not be used for fair LM evaluation.
4. `--threshold` is kept for the legacy teacher-forced meta-border MSE threshold.

## Baseline

Use `--model-type baseline` for a simple byte model: one stacked GRU over bytes with total depth equal to the adaptive stack. It uses only the final next-byte loss.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch datasets
pip install wandb
```

## Train

```bash
python -m adaptive_compressor.train \
  --model-type adaptive \
  --border-mode uncertainty \
  --sequence-length 128 \
  --batch-size 8 \
  --hidden-size 128 \
  --num-levels 3 \
  --threshold 0.1 \
  --byte-entropy-threshold 5.0 \
  --meta-uncertainty-threshold 0.1 \
  --eval-every 20 \
  --max-steps 200
```

```bash
python -m adaptive_compressor.train \
  --model-type baseline \
  --sequence-length 128 \
  --batch-size 8 \
  --hidden-size 128 \
  --num-levels 3 \
  --threshold 0.1 \
  --eval-every 20 \
  --max-steps 200
```

The script uses `Salesforce/wikitext` with config `wikitext-103-raw-v1` by default, logs to the Weights & Biases project `adaptive_compressor`, evaluates on the validation split every `--eval-every` steps using a small subset, and writes a checkpoint to `checkpoints/adaptive_compressor.pt`.

Pass `--disable-wandb` to run without online logging.
If `--wandb-run-name` is not provided, the default run name is `${model_type}_L${sequence_length}_B${batch_tokens // 1000}k`.
If you pass `--border-mode teacher_forced`, training will warn that the adaptive routing uses target-dependent borders and therefore leaks future information.

## Inference

The current inference path is intentionally simple and causal: it recomputes the hierarchy from the current prefix at every generation step. This is slower than a cached scheduler, but it is the cleanest way to verify causal behavior.

```bash
python -m adaptive_compressor.infer checkpoints/adaptive_compressor.pt \
  --prompt "The meaning of compression is" \
  --max-new-bytes 128 \
  --temperature 1.0 \
  --top-k 0
```

To check whether prefix-only logits match full-sequence logits on a prompt prefix:

```bash
python -m adaptive_compressor.infer checkpoints/adaptive_compressor.pt \
  --prompt "The meaning of compression is" \
  --check-causality \
  --causality-max-positions 64 \
  --max-new-bytes 16
```

## Files

1. `adaptive_compressor/model.py` - hierarchical encoder/decoder model.
2. `adaptive_compressor/routing.py` - border selection and span routing helpers.
3. `adaptive_compressor/data.py` - WikiText byte dataset.
4. `adaptive_compressor/train.py` - small training entry point.
