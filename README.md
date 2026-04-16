# Adaptive Compressor

Minimal PyTorch prototype of a hierarchical byte-level language model with adaptive border selection, plus a residual byte-level GRU baseline.

## Core idea

1. Level 0 models raw UTF-8 bytes and promotes positions whose next-byte prediction is wrong.
2. Higher levels model only the promoted border embeddings and promote positions whose next-embedding prediction MSE is above a threshold.
3. Each sequence always starts with a border at position `0`, so every compressed token owns a span in the level below.
4. The decoder mirrors the compressor and reconstructs lower-level sequences by broadcast-adding each parent embedding over its child span.

By default, border selection is causal:
1. Level 0 accumulates byte-prediction entropy within the current span and promotes a border once the cumulative sum exceeds `--byte-entropy-threshold`.
2. Higher levels accumulate predicted next-embedding uncertainty within the current span and promote a border once the cumulative sum exceeds `--meta-uncertainty-threshold`.
3. The original teacher-forced border rule is still available with `--border-mode teacher_forced`, but it leaks target information and should not be used for fair LM evaluation.
4. `--threshold` is kept for the legacy teacher-forced meta-border MSE threshold.
5. A small entropy regularizer can keep the byte model from becoming too overconfident everywhere without directly changing the routing rule.

## Baseline

Use `--model-type baseline` for a non-hierarchical byte model built from residual 2-layer GRU blocks. It uses only the final next-byte loss, but matches the adaptive model's trainable parameter count exactly.

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
  --byte-entropy-threshold 20.0 \
  --meta-uncertainty-threshold 1.0 \
  --entropy-floor 0.0 \
  --entropy-reg-weight 0.001 \
  --eval-every 20 \
  --max-steps 200
```

Or use the helper scripts:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_adaptive.sh --max-steps 200 --batch-size 16
CUDA_VISIBLE_DEVICES=1 scripts/run_baseline.sh --max-steps 200 --batch-size 16
```

The scripts accept common hyperparameters through environment variables such as `SEQUENCE_LENGTH`, `BATCH_SIZE`, `HIDDEN_SIZE`, `NUM_LEVELS`, and also forward any extra CLI flags you append.

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
For comparison to the residual baseline, prefer `byte_encoder_bpb` rather than `byte_decoder_bpb`, because the adaptive decoder adds extra depth even when the hierarchy collapses.

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

1. `adaptive_compressor/models/` - split model package with shared modules, adaptive model, and baseline.
2. `adaptive_compressor/routing.py` - border selection and span routing helpers.
3. `adaptive_compressor/data.py` - WikiText byte dataset.
4. `adaptive_compressor/train.py` - small training entry point.

## Visualization

Open `docs/model_visualization.html` for a colleague-facing explanation of the current architecture, the cumulative border rule, and the exact cached inference loop.
