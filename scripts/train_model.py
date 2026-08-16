r"""Pretrain the base language model on the frozen token shards.

    python scripts\train_model.py `
        --dataset data\tokenized\v0.1 `
        --checkpoint-dir checkpoints\base-v0.1 `
        --max-steps 100000 `
        --micro-batch-size 8 `
        --gradient-accumulation-steps 8 `
        --precision bf16 `
        --device cuda

Resuming is the default behaviour: if ``--checkpoint-dir`` already holds
checkpoints, the highest-step one is loaded and training continues from it.
Pass ``--no-resume`` to start over.

Safety rails
------------
Three things are checked before a single step runs, because each is a mistake
that produces a plausible-looking run and a worthless model:

* the model's ``vocab_size`` and ``context_length`` must match the dataset
  manifest -- a mismatch trains against ids the model has no rows for;
* the dataset's stream digests are recorded into every checkpoint, so resuming
  against a different corpus is refused rather than silently accepted;
* a smoke evaluation runs before training, and its loss should sit near
  ``ln(vocab_size)`` for a fresh model. A materially lower value means the
  weights are not actually fresh; a higher one means something is wrong with
  the data path.
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


DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the 32.7M-parameter base model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)

    parser.add_argument("--max-steps", type=positive_int, default=1_000)
    parser.add_argument("--micro-batch-size", type=positive_int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=1)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=positive_int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=positive_int, default=10)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start from step 0.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip shard SHA-256 verification when opening the dataset.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Fail early and legibly on an unavailable device.

    ``torch`` otherwise raises a bare ``AssertionError`` from deep inside
    ``.to()``, after the dataset has already been opened and verified. Checking
    here turns a stack trace into a sentence that says what to install.
    """

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        detail = (
            "this PyTorch build has no CUDA support compiled in"
            if torch.version.cuda is None
            else "no CUDA device is visible"
        )
        raise RuntimeError(
            f"--device cuda was requested but {detail} "
            f"(torch {torch.__version__}). Install a CUDA build of PyTorch on a "
            "machine with an NVIDIA GPU, or pass --device cpu."
        )

    if device.type == "cpu" and requested != "cpu":
        raise RuntimeError(f"unsupported device: {requested!r}")

    return device


def warn_about_precision(device: torch.device, precision: str) -> None:
    if precision == "bf16" and device.type == "cpu":
        print(
            "WARNING: bf16 on CPU is typically slower than fp32 and is not the "
            "path this flag exists for. Use --precision fp32 on CPU.",
            file=sys.stderr,
        )
    if precision == "fp16":
        print(
            "WARNING: fp16 has no gradient scaler here. Prefer bf16, which keeps "
            "fp32's exponent range and does not need one.",
            file=sys.stderr,
        )


def load_datasets(
    dataset_dir: Path, *, context_length: int, vocab_size: int, verify: bool
):
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train = PretrainingDataset(
        dataset_dir / "train" / "manifest.json",
        context_length=context_length,
        expected_vocab_size=vocab_size,
        verify_checksums=verify,
    )
    validation = PretrainingDataset(
        dataset_dir / "validation" / "manifest.json",
        context_length=context_length,
        expected_vocab_size=vocab_size,
        verify_checksums=verify,
    )
    return manifest, train, validation


def dataset_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    splits = manifest.get("splits", {})
    return {
        "train_stream_sha256": splits.get("train", {}).get("stream_sha256"),
        "validation_stream_sha256": splits.get("validation", {}).get("stream_sha256"),
        "tokenizer_state_sha256": manifest.get("tokenizer", {}).get("state_sha256"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    try:
        # Before anything expensive: opening the dataset verifies 2.2 GB of
        # shard checksums, and there is no point paying for that only to
        # discover the device is unusable.
        device = resolve_device(args.device)
        warn_about_precision(device, args.precision)

        model_config = ModelConfig()

        manifest, train_dataset, validation_dataset = load_datasets(
            args.dataset,
            context_length=model_config.context_length,
            vocab_size=model_config.vocab_size,
            verify=not args.skip_checksums,
        )
        model_config.validate_against_dataset(manifest)

        training_config = TrainingConfig(
            max_steps=args.max_steps,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            seed=args.seed,
            precision=args.precision,
            device=args.device,
        )

        torch.manual_seed(args.seed)
        model = Transformer(model_config)

        trainer = Trainer(
            model,
            train_dataset,
            validation_dataset,
            config=training_config,
            checkpoint_dir=args.checkpoint_dir,
            dataset_identity=dataset_identity(manifest),
        )

        resumed_from = None
        if args.checkpoint_dir is not None and not args.no_resume:
            resumed_from = trainer.resume_latest()

        tokens_per_step = (
            training_config.effective_batch_size * model_config.context_length
        )

        print()
        print("PRETRAINING")
        print("=" * 88)
        print(f"Dataset:            {args.dataset}")
        print(f"Train windows:      {len(train_dataset):,}")
        print(f"Validation windows: {len(validation_dataset):,}")
        print(f"Parameters:         {model.parameter_count():,}")
        print(f"Device:             {training_config.device}  precision {args.precision}")
        print(
            f"Batch:              {training_config.micro_batch_size} x "
            f"{training_config.gradient_accumulation_steps} accum x "
            f"{model_config.context_length} ctx = {tokens_per_step:,} tokens/step"
        )
        print(
            f"Schedule:           peak {args.learning_rate:.2e} -> "
            f"{args.min_learning_rate:.2e}, warmup {args.warmup_steps}, "
            f"total {args.max_steps}"
        )
        print(
            f"Planned tokens:     {tokens_per_step * args.max_steps:,} "
            f"({tokens_per_step * args.max_steps / 1e9:.3f}B)"
        )
        if resumed_from is not None:
            print(f"Resumed from:       {resumed_from} at step {trainer.step:,}")
        print()

        if trainer.step == 0:
            baseline = trainer.evaluate(batches=min(4, args.eval_batches))
            uniform = math.log(model_config.vocab_size)
            print(f"Pre-training validation loss: {baseline:.4f} "
                  f"(uniform baseline {uniform:.4f})")
            if baseline < uniform - 1.0:
                print(
                    "WARNING: a fresh model should score near the uniform "
                    "baseline. This model appears to be already trained.",
                    file=sys.stderr,
                )
            print()

        print(Trainer.log_header())
        print("-" * 88)

        wall_started = time.perf_counter()
        history = trainer.train(progress=True)
        elapsed = time.perf_counter() - wall_started

        if args.checkpoint_dir is not None:
            final = trainer.save(args.checkpoint_dir / f"step-{trainer.step:08d}.pt")
            print()
            print(f"Final checkpoint: {final}")

        final_validation = (
            trainer.evaluate() if validation_dataset is not None else None
        )
        total_tokens = trainer.tokens_processed

        print()
        print("=" * 88)
        print(f"Steps completed:    {trainer.step:,}")
        print(f"Tokens processed:   {total_tokens:,}")
        print(f"Elapsed:            {elapsed:,.1f}s")
        if elapsed > 0 and history:
            observed = sum(m.tokens_processed for m in history[-1:]) and (
                sum(m.seconds for m in history)
            )
            throughput = sum(
                m.tokens_per_second * m.seconds for m in history
            ) / max(observed, 1e-9)
            print(f"Throughput:         {throughput:,.0f} tokens/s")
        if history:
            print(f"First loss:         {history[0].loss:.4f}")
            print(f"Last loss:          {history[-1].loss:.4f}")
        if final_validation is not None:
            print(f"Validation loss:    {final_validation:.4f}")
            print(f"Validation ppl:     {math.exp(min(final_validation, 20)):,.2f}")

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "started_at_utc": started.isoformat(),
                        "elapsed_seconds": elapsed,
                        "dataset": str(args.dataset),
                        "dataset_identity": dataset_identity(manifest),
                        "model_config": model_config.to_dict(),
                        "training_config": training_config.to_dict(),
                        "parameters": model.parameter_count(),
                        "resumed_from": str(resumed_from) if resumed_from else None,
                        "steps_completed": trainer.step,
                        "tokens_processed": total_tokens,
                        "final_validation_loss": final_validation,
                        "history": [m.to_dict() for m in history],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print()
            print(f"Report: {args.report}")

        return 0

    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
