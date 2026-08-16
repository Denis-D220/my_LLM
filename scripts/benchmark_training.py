r"""Measure what this GPU can actually do, then recommend a training command.

Run this once on a fresh machine before committing to a long job.  It answers
the four questions that decide the run, using the real model and the real
shards rather than estimates:

    1. what micro-batch size fits in memory
    2. how many tokens per second that yields
    3. which peak learning rate descends fastest without instability
    4. how long one epoch will therefore take

    python scripts/benchmark_training.py \
        --dataset data/tokenized/v0.1 \
        --device cuda \
        --precision bf16

Phase 1 doubles the micro-batch until it stops fitting or stops getting
faster.  Out-of-memory is an expected outcome here, not a failure -- finding
the ceiling is the point -- so it is caught and reported rather than raised.

Phase 2 trains a *freshly initialized* model for a short burst at each
candidate learning rate, from an identical seed, so the only difference
between trials is the learning rate.  It reports the loss reduction and the
largest gradient norm seen: a rate that descends fast but spikes the gradient
norm is worse than a slightly slower one that does not, because the spike is
what eventually produces a NaN eight hours into a run.

Both phases are short.  The whole thing should take a few minutes, against a
pretraining job measured in hours.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch

from llm.data.dataset import PretrainingDataset
from llm.model import ModelConfig
from llm.model.transformer import Transformer
from llm.training import Trainer, TrainingConfig


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_LEARNING_RATES = (1e-4, 3e-4, 6e-4, 1e-3)
TARGET_EFFECTIVE_TOKENS = 524_288  # ~0.5M tokens/step, a common budget


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark throughput and probe learning rates on real data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Micro-batch sizes to try, ascending.",
    )
    parser.add_argument("--throughput-steps", type=positive_int, default=6)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEARNING_RATES),
    )
    parser.add_argument("--probe-steps", type=positive_int, default=40)
    parser.add_argument(
        "--target-tokens-per-step",
        type=positive_int,
        default=TARGET_EFFECTIVE_TOKENS,
        help="Effective batch to reach via gradient accumulation.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--skip-lr-probe", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args(argv)


def reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def build_trainer(
    dataset,
    validation,
    *,
    micro_batch: int,
    accumulation: int,
    learning_rate: float,
    steps: int,
    args: argparse.Namespace,
) -> Trainer:
    torch.manual_seed(args.seed)
    model = Transformer(ModelConfig())
    return Trainer(
        model,
        dataset,
        validation,
        config=TrainingConfig(
            max_steps=steps,
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            learning_rate=learning_rate,
            min_learning_rate=learning_rate / 10,
            warmup_steps=min(5, steps),
            eval_interval=0,
            checkpoint_interval=0,
            log_interval=10**9,
            seed=args.seed,
            precision=args.precision,
            device=args.device,
        ),
    )


def benchmark_throughput(
    dataset, validation, args: argparse.Namespace, device: torch.device
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    print()
    print("PHASE 1 -- micro-batch sweep")
    print("-" * 78)
    print(f"{'micro':>7}{'tokens/step':>14}{'s/step':>10}{'tokens/s':>12}{'peak VRAM':>14}")
    print("-" * 78)

    for micro_batch in sorted(args.batch_sizes):
        if micro_batch > len(dataset):
            continue

        reset_memory(device)
        try:
            trainer = build_trainer(
                dataset,
                validation,
                micro_batch=micro_batch,
                accumulation=1,
                learning_rate=1e-4,
                steps=args.throughput_steps + 1,
                args=args,
            )

            trainer.train_step()  # warmup: allocator, autotune, kernel selection
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            started = time.perf_counter()
            for _ in range(args.throughput_steps):
                trainer.train_step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) / args.throughput_steps

        except torch.cuda.OutOfMemoryError:
            print(f"{micro_batch:>7}{'OUT OF MEMORY':>50}")
            reset_memory(device)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"{micro_batch:>7}{'OUT OF MEMORY':>50}")
            reset_memory(device)
            break

        tokens = micro_batch * ModelConfig().context_length
        memory = peak_memory_bytes(device)
        results.append(
            {
                "micro_batch_size": micro_batch,
                "tokens_per_step": tokens,
                "seconds_per_step": elapsed,
                "tokens_per_second": tokens / elapsed,
                "peak_memory_bytes": memory,
            }
        )

        memory_text = f"{memory / 2**30:,.2f} GiB" if memory else "n/a"
        print(
            f"{micro_batch:>7}{tokens:>14,}{elapsed:>10.3f}"
            f"{tokens / elapsed:>12,.0f}{memory_text:>14}"
        )

        del trainer
        reset_memory(device)

    return results


def probe_learning_rates(
    dataset, validation, args: argparse.Namespace, micro_batch: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    print()
    print(f"PHASE 2 -- learning-rate probe ({args.probe_steps} steps, micro-batch {micro_batch})")
    print("-" * 78)
    print(f"{'lr':>10}{'first loss':>13}{'final loss':>13}{'reduction':>12}{'max grad':>12}{'':>8}")
    print("-" * 78)

    for learning_rate in args.learning_rates:
        trainer = build_trainer(
            dataset,
            validation,
            micro_batch=micro_batch,
            accumulation=1,
            learning_rate=learning_rate,
            steps=args.probe_steps,
            args=args,
        )
        history = trainer.train()

        losses = [m.loss for m in history]
        grads = [m.grad_norm for m in history]
        finite = all(math.isfinite(v) for v in losses + grads)
        first = sum(losses[:3]) / min(3, len(losses))
        final = sum(losses[-3:]) / min(3, len(losses))

        verdict = "ok" if finite and final < first else ("DIVERGED" if not finite else "no progress")
        results.append(
            {
                "learning_rate": learning_rate,
                "first_loss": first,
                "final_loss": final,
                "reduction": first - final,
                "max_grad_norm": max(grads) if grads else float("nan"),
                "finite": finite,
                "verdict": verdict,
            }
        )

        print(
            f"{learning_rate:>10.1e}{first:>13.4f}{final:>13.4f}"
            f"{first - final:>12.4f}{max(grads):>12.3f}{verdict:>8}"
        )

        del trainer

    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(
                f"ERROR: --device cuda requested but torch {torch.__version__} "
                "reports no CUDA. Install a CUDA build.",
                file=sys.stderr,
            )
            return 1

        config = ModelConfig()
        train_dataset = PretrainingDataset(
            args.dataset / "train" / "manifest.json",
            context_length=config.context_length,
            expected_vocab_size=config.vocab_size,
            verify_checksums=not args.skip_checksums,
        )
        validation_dataset = PretrainingDataset(
            args.dataset / "validation" / "manifest.json",
            context_length=config.context_length,
            expected_vocab_size=config.vocab_size,
            verify_checksums=not args.skip_checksums,
        )

        print()
        print("TRAINING BENCHMARK")
        print("=" * 78)
        print(f"Device:        {device}")
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            print(f"GPU:           {properties.name}")
            print(f"VRAM:          {properties.total_memory / 2**30:,.1f} GiB")
            print(f"Capability:    {properties.major}.{properties.minor}")
            print(f"BF16 support:  {torch.cuda.is_bf16_supported()}")
        print(f"torch:         {torch.__version__}  (cuda {torch.version.cuda})")
        print(f"Precision:     {args.precision}")
        print(f"Parameters:    {config.parameter_count():,}")
        print(f"Train windows: {len(train_dataset):,}")

        if args.precision == "bf16" and device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                print(
                    "WARNING: this GPU does not report BF16 support; results "
                    "below may fall back or be slow.",
                    file=sys.stderr,
                )

        throughput = benchmark_throughput(train_dataset, validation_dataset, args, device)
        if not throughput:
            print("ERROR: no micro-batch size completed a step.", file=sys.stderr)
            return 2

        best = max(throughput, key=lambda r: r["tokens_per_second"])
        largest = max(throughput, key=lambda r: r["micro_batch_size"])

        learning_rates = (
            []
            if args.skip_lr_probe
            else probe_learning_rates(
                train_dataset, validation_dataset, args, best["micro_batch_size"]
            )
        )

        # ------------------------------------------------------------------
        # recommendation
        # ------------------------------------------------------------------
        micro = best["micro_batch_size"]
        tokens_per_micro = micro * config.context_length
        accumulation = max(1, round(args.target_tokens_per_step / tokens_per_micro))
        tokens_per_step = tokens_per_micro * accumulation

        train_tokens = len(train_dataset) * config.context_length
        steps_per_epoch = train_tokens // tokens_per_step
        seconds_per_epoch = train_tokens / best["tokens_per_second"]

        usable = [r for r in learning_rates if r["verdict"] == "ok"]
        recommended_lr = (
            max(usable, key=lambda r: r["reduction"])["learning_rate"]
            if usable
            else None
        )

        print()
        print("=" * 78)
        print("RESULTS")
        print("-" * 78)
        print(f"Fastest micro-batch:   {micro}  ({best['tokens_per_second']:,.0f} tokens/s)")
        print(f"Largest that fit:      {largest['micro_batch_size']}")
        if best["peak_memory_bytes"]:
            print(f"Peak VRAM at that size:{best['peak_memory_bytes'] / 2**30:>7,.2f} GiB")
        print(f"Accumulation for ~{args.target_tokens_per_step:,} tokens/step: {accumulation}")
        print(f"Effective tokens/step: {tokens_per_step:,}")
        print(f"Steps per epoch:       {steps_per_epoch:,}")
        print(
            f"Estimated epoch time:  {seconds_per_epoch / 3600:,.2f} hours "
            f"({seconds_per_epoch / 60:,.0f} min)"
        )
        if recommended_lr is not None:
            print(f"Recommended peak LR:   {recommended_lr:.1e}")
        elif learning_rates:
            print("Recommended peak LR:   none of the probed rates made progress")

        print()
        print("Suggested command:")
        print()
        lr_text = f"{recommended_lr:.1e}" if recommended_lr else "<choose from phase 2>"
        print(f"  python scripts/train_model.py \\")
        print(f"      --dataset {args.dataset} \\")
        print(f"      --checkpoint-dir checkpoints/base-v0.1 \\")
        print(f"      --max-steps {steps_per_epoch} \\")
        print(f"      --micro-batch-size {micro} \\")
        print(f"      --gradient-accumulation-steps {accumulation} \\")
        print(f"      --learning-rate {lr_text} \\")
        print(f"      --warmup-steps {max(10, steps_per_epoch // 50)} \\")
        print(f"      --precision {args.precision} \\")
        print(f"      --device {args.device} \\")
        print(f"      --report data/audits/pretrain-v0.1.json")
        print()

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "device": str(device),
                        "gpu": (
                            torch.cuda.get_device_properties(device).name
                            if device.type == "cuda"
                            else None
                        ),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "precision": args.precision,
                        "parameters": config.parameter_count(),
                        "throughput": throughput,
                        "learning_rates": learning_rates,
                        "recommendation": {
                            "micro_batch_size": micro,
                            "gradient_accumulation_steps": accumulation,
                            "tokens_per_step": tokens_per_step,
                            "steps_per_epoch": steps_per_epoch,
                            "estimated_epoch_hours": seconds_per_epoch / 3600,
                            "learning_rate": recommended_lr,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"Report: {args.report}")

        return 0

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
