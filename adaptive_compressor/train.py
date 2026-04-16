"""Train a minimal adaptive compressor on WikiText bytes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ByteDatasetConfig, load_wikitext_byte_dataset
from .model import AdaptiveCompressorConfig, build_model
from .runtime import choose_device

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type", choices=["adaptive", "baseline"], default="adaptive"
    )
    parser.add_argument(
        "--border-mode",
        choices=["uncertainty", "teacher_forced"],
        default="uncertainty",
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--byte-entropy-threshold", type=float, default=20.0)
    parser.add_argument("--meta-uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--entropy-floor", type=float, default=0.0)
    parser.add_argument("--entropy-reg-weight", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-documents", type=int, default=5000)
    parser.add_argument("--val-max-documents", type=int, default=1000)
    parser.add_argument("--val-max-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="adaptive_compressor")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/adaptive_compressor.pt")
    )
    return parser.parse_args()


def format_border_stats(border_counts: list[int]) -> str:
    if not border_counts:
        return "none"
    return ", ".join(f"L{idx + 1}:{count}" for idx, count in enumerate(border_counts))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def default_wandb_run_name(args: argparse.Namespace) -> str:
    batch_tokens = args.batch_size * args.sequence_length
    return f"{args.model_type}_L{args.sequence_length}_B{batch_tokens // 1000}k"


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_byte_encoder_loss = 0.0
    total_meta_loss = 0.0
    total_uncertainty_loss = 0.0
    total_entropy_reg_loss = 0.0
    total_byte_decoder_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            byte_ids = batch.to(device)
            outputs = model(byte_ids)

            total_loss += outputs["loss"].item()
            total_byte_encoder_loss += outputs["byte_encoder_loss"].item()
            total_meta_loss += outputs["meta_loss"].item()
            total_uncertainty_loss += outputs["uncertainty_loss"].item()
            total_entropy_reg_loss += outputs["entropy_reg_loss"].item()
            total_byte_decoder_loss += outputs["byte_decoder_loss"].item()
            batch_count += 1

            if batch_count >= max_batches:
                break

    model.train()

    if batch_count == 0:
        return {
            "val/loss": 0.0,
            "val/byte_encoder_loss": 0.0,
            "val/byte_encoder_bpb": 0.0,
            "val/meta_loss": 0.0,
            "val/uncertainty_loss": 0.0,
            "val/entropy_reg_loss": 0.0,
            "val/byte_decoder_loss": 0.0,
            "val/byte_decoder_bpb": 0.0,
        }

    scale = 1.0 / batch_count
    return {
        "val/loss": total_loss * scale,
        "val/byte_encoder_loss": total_byte_encoder_loss * scale,
        "val/byte_encoder_bpb": (total_byte_encoder_loss * scale) / math.log(2.0),
        "val/meta_loss": total_meta_loss * scale,
        "val/uncertainty_loss": total_uncertainty_loss * scale,
        "val/entropy_reg_loss": total_entropy_reg_loss * scale,
        "val/byte_decoder_loss": total_byte_decoder_loss * scale,
        "val/byte_decoder_bpb": (total_byte_decoder_loss * scale) / math.log(2.0),
    }


def maybe_init_wandb(
    args: argparse.Namespace, config: AdaptiveCompressorConfig, parameter_count: int
) -> bool:
    if args.disable_wandb:
        return False
    if wandb is None:
        print("wandb is not installed; continuing without experiment logging")
        return False

    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or default_wandb_run_name(args),
            config={
                **vars(args),
                "parameter_count": parameter_count,
                "model_config": vars(config),
            },
        )
    except Exception as exc:  # pragma: no cover - depends on external auth state
        print(f"wandb initialization failed; continuing without logging: {exc}")
        return False

    return True


def main() -> None:
    args = parse_args()
    device = choose_device()

    dataset = load_wikitext_byte_dataset(
        ByteDatasetConfig(
            sequence_length=args.sequence_length,
            max_documents=args.max_documents,
        )
    )
    val_dataset = load_wikitext_byte_dataset(
        ByteDatasetConfig(
            split="validation",
            sequence_length=args.sequence_length,
            max_documents=args.val_max_documents,
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model_config = AdaptiveCompressorConfig(
        hidden_size=args.hidden_size,
        num_levels=args.num_levels,
        threshold=args.threshold,
        border_mode=args.border_mode,
        byte_entropy_threshold=args.byte_entropy_threshold,
        meta_uncertainty_threshold=args.meta_uncertainty_threshold,
        entropy_floor=args.entropy_floor,
        entropy_reg_weight=args.entropy_reg_weight,
        dropout=args.dropout,
    )
    model = build_model(args.model_type, model_config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parameter_count = count_parameters(model)
    print(f"model_type={args.model_type} parameters={parameter_count}")
    if args.model_type == "adaptive" and args.border_mode == "teacher_forced":
        print(
            "warning: teacher_forced border mode uses target-dependent routing and leaks future information"
        )
    wandb_enabled = maybe_init_wandb(args, model_config, parameter_count)

    model.train()
    step = 0
    progress = tqdm(
        total=args.max_steps, desc=f"train:{args.model_type}", dynamic_ncols=True
    )
    while step < args.max_steps:
        for batch in dataloader:
            byte_ids = batch.to(device)
            outputs = model(byte_ids)

            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            optimizer.step()

            step += 1
            progress.update(1)
            if step % args.log_every == 0 or step == 1:
                metrics = {
                    "train/loss": outputs["loss"].item(),
                    "train/byte_encoder_loss": outputs["byte_encoder_loss"].item(),
                    "train/byte_encoder_bpb": outputs["byte_encoder_loss"].item()
                    / math.log(2.0),
                    "train/meta_loss": outputs["meta_loss"].item(),
                    "train/uncertainty_loss": outputs["uncertainty_loss"].item(),
                    "train/entropy_reg_loss": outputs["entropy_reg_loss"].item(),
                    "train/byte_decoder_loss": outputs["byte_decoder_loss"].item(),
                    "train/byte_decoder_bpb": outputs["byte_decoder_loss"].item()
                    / math.log(2.0),
                    "train/parameter_count": float(parameter_count),
                }
                for level_idx, border_count in enumerate(outputs["border_counts"]):
                    metrics[f"train/border_count_level_{level_idx + 1}"] = float(
                        border_count
                    )

                progress.set_postfix(
                    loss=f"{metrics['train/loss']:.4f}",
                    bpb=f"{metrics['train/byte_decoder_bpb']:.3f}",
                    enc=f"{metrics['train/byte_encoder_bpb']:.3f}",
                    meta=f"{metrics['train/meta_loss']:.4f}",
                    unc=f"{metrics['train/uncertainty_loss']:.4f}",
                    reg=f"{metrics['train/entropy_reg_loss']:.4f}",
                    dec=f"{metrics['train/byte_decoder_loss']:.4f}",
                )

                tqdm.write(
                    f"step={step} "
                    f"loss={metrics['train/loss']:.4f} "
                    f"bpb={metrics['train/byte_decoder_bpb']:.3f} "
                    f"enc_bpb={metrics['train/byte_encoder_bpb']:.3f} "
                    f"meta={metrics['train/meta_loss']:.4f} "
                    f"unc={metrics['train/uncertainty_loss']:.4f} "
                    f"reg={metrics['train/entropy_reg_loss']:.4f} "
                    f"dec={metrics['train/byte_decoder_loss']:.4f} "
                    f"borders={format_border_stats(outputs['border_counts'])}"
                )
                if wandb_enabled:
                    wandb.log(metrics, step=step)

            if args.eval_every > 0 and (
                step % args.eval_every == 0 or step == args.max_steps
            ):
                val_metrics = evaluate(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    max_batches=args.val_max_batches,
                )
                progress.set_postfix(
                    loss=f"{outputs['loss'].item():.4f}",
                    val=f"{val_metrics['val/loss']:.4f}",
                    val_bpb=f"{val_metrics['val/byte_decoder_bpb']:.3f}",
                )
                tqdm.write(
                    f"eval step={step} "
                    f"val_loss={val_metrics['val/loss']:.4f} "
                    f"val_bpb={val_metrics['val/byte_decoder_bpb']:.3f} "
                    f"val_enc_bpb={val_metrics['val/byte_encoder_bpb']:.3f} "
                    f"val_meta={val_metrics['val/meta_loss']:.4f} "
                    f"val_unc={val_metrics['val/uncertainty_loss']:.4f} "
                    f"val_reg={val_metrics['val/entropy_reg_loss']:.4f} "
                    f"val_dec={val_metrics['val/byte_decoder_loss']:.4f}"
                )
                if wandb_enabled:
                    wandb.log(val_metrics, step=step)

            if step >= args.max_steps:
                break

    progress.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config,
            "args": vars(args),
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
