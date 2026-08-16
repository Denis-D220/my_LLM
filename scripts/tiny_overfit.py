r"""Prove the full-scale model can memorize a microscopic dataset.

The decisive gate before any real training run.  Every component test says the
model computes what it claims; this says the assembled thing can *learn*.  A
32.7M-parameter Transformer that cannot memorize eight short sequences has a
bug in the model, the loss, the gradients, or the optimizer wiring, and no
amount of real data will fix it.

    python scripts\tiny_overfit.py `
        --examples 8 `
        --sequence-length 64 `
        --steps 200 `
        --learning-rate 1e-3

Expected shape of the run::

    step   0   loss ~10.09     (== ln(24000), a uniform 24k-way guess)
    step  50   loss   6-8
    step 100   loss   2-5
    step 200   loss  < 0.5

The exact curve does not matter.  What matters is that it falls, that nothing
goes non-finite, and that the model ends up predicting the batch it was trained
on.  ``tests/model/test_overfit.py`` runs the same experiment at small scale as
a fast permanent regression guard.

Data here is deliberately *random token ids*, not real text.  Random sequences
have no structure to generalize from, so the only way to drive the loss to zero
is genuine memorization -- which is exactly the capability being tested.  Real
text would let a partially-broken model score well by learning token frequency
alone.

The reported tokens/second is measured on this machine at this batch shape and
is the first honest input to estimating what full pretraining will cost.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

from llm.model import ModelConfig
from llm.model.transformer import Transformer, causal_lm_loss


DEFAULT_LOSS_TARGET = 0.5


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the full-scale model on a tiny fixed batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--examples", type=positive_int, default=8)
    parser.add_argument("--sequence-length", type=positive_int, default=64)
    parser.add_argument("--steps", type=positive_int, default=200)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=positive_int, default=10)
    parser.add_argument(
        "--loss-target",
        type=positive_float,
        default=DEFAULT_LOSS_TARGET,
        help="Run fails if the final loss does not fall below this.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    config = ModelConfig()

    if args.sequence_length > config.context_length:
        print(
            f"ERROR: --sequence-length {args.sequence_length} exceeds the frozen "
            f"context length {config.context_length}",
            file=sys.stderr,
        )
        return 1

    model = Transformer(config).to(device)
    parameters = model.parameter_count()

    generator = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.examples, args.sequence_length + 1),
        generator=generator,
    )
    inputs = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    uniform = math.log(config.vocab_size)
    tokens_per_step = args.examples * args.sequence_length

    print()
    print("TINY OVERFIT")
    print("=" * 72)
    print(f"Parameters:        {parameters:,}")
    print(f"Device:            {device}")
    print(f"Batch:             {args.examples} x {args.sequence_length} "
          f"= {tokens_per_step:,} tokens/step")
    print(f"Optimizer:         AdamW lr={args.learning_rate} betas=(0.9, 0.95) "
          f"weight_decay={args.weight_decay}")
    print(f"Uniform baseline:  ln({config.vocab_size:,}) = {uniform:.4f}")
    print()
    print(f"{'step':>6}{'loss':>12}{'grad norm':>14}{'tok/s':>12}{'elapsed':>10}")
    print("-" * 72)

    history: list[float] = []
    started = time.perf_counter()
    non_finite = False

    for step in range(args.steps):
        step_started = time.perf_counter()

        logits = model(inputs)
        loss = causal_lm_loss(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()

        value = loss.item()
        history.append(value)
        if not math.isfinite(value):
            non_finite = True
            print(f"{step:>6}{value:>12}  NON-FINITE LOSS -- stopping")
            break

        if step % args.log_every == 0 or step == args.steps - 1:
            step_seconds = time.perf_counter() - step_started
            print(
                f"{step:>6}{value:>12.4f}{float(grad_norm):>14.3f}"
                f"{tokens_per_step / step_seconds:>12,.0f}"
                f"{time.perf_counter() - started:>9.1f}s"
            )

    elapsed = time.perf_counter() - started

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    final = history[-1] if history else float("nan")
    memorized = math.isfinite(final) and final < args.loss_target

    print("-" * 72)
    print(f"Initial loss:      {history[0]:.4f}  (uniform {uniform:.4f})")
    print(f"Final loss:        {final:.4f}")
    print(f"Reduction:         {history[0] / final:,.1f}x" if final > 0 else "")
    print(f"Token accuracy:    {accuracy:.2%}")
    print(f"Steps:             {len(history)}")
    print(f"Elapsed:           {elapsed:.1f}s")
    print(f"Throughput:        {len(history) * tokens_per_step / elapsed:,.0f} tokens/s")
    print()

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "parameters": parameters,
                    "device": str(device),
                    "examples": args.examples,
                    "sequence_length": args.sequence_length,
                    "steps": len(history),
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "uniform_baseline": uniform,
                    "initial_loss": history[0] if history else None,
                    "final_loss": final,
                    "token_accuracy": accuracy,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": len(history) * tokens_per_step / elapsed,
                    "loss_history": history,
                    "memorized": memorized,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Report: {args.report}")
        print()

    if non_finite:
        print("RESULT: FAIL -- loss became non-finite")
        return 2
    if not memorized:
        print(
            f"RESULT: FAIL -- final loss {final:.4f} did not reach "
            f"{args.loss_target}. Do not start real training; something in the "
            "model, loss, gradients or optimizer is wrong."
        )
        return 2

    print("RESULT: PASS -- the model memorized the batch.")
    print("Architecture and training mechanics are proven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
