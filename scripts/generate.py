r"""Talk to the trained model.

Just run it:

    python scripts\generate.py

No arguments needed.  It finds the newest checkpoint and the frozen tokenizer
on its own, then opens an interactive prompt.  Every flag below exists to
override that, not to make it work.

    python scripts\generate.py --sweep            probe prompts x 3 presets
    python scripts\generate.py --prompt "..."     one-shot
    python scripts\generate.py --checkpoint X     a specific checkpoint

What to expect from a 32.7M base model
--------------------------------------
Not answers.  This is a *base* language model trained on next-token prediction
over filtered web text; it continues patterns, it does not respond to
requests.  "Newton's second law states that" gets continued plausibly.
"Explain X" gets continued as though it appeared in a document, not obeyed.

Worth checking: coherent English, sound local grammar, technical vocabulary in
the right places, no immediate repetition loops.  Judging it against a chat
assistant is judging it against something it was never trained to be.

Checkpoint formats
------------------
Accepts a full training checkpoint (a payload with ``model_state_dict``, which
also carries the architecture and step count) or a bare ``state_dict``.  In the
second case the frozen v0.1 architecture is assumed, since nothing in the file
says otherwise.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_imports() -> None:
    """Make ``llm`` importable even without ``pip install -e .``.

    The whole point of this script is that running the file works.  Requiring
    an editable install first would defeat that, and the failure it produces
    (``ModuleNotFoundError: llm``) tells a reader nothing about the fix.
    """

    try:
        import llm  # noqa: F401
    except ModuleNotFoundError:
        source = REPO_ROOT / "src"
        if source.is_dir():
            sys.path.insert(0, str(source))


_bootstrap_imports()

import torch  # noqa: E402

from llm.generation import SamplingConfig, StreamingDecoder, generate  # noqa: E402
from llm.model import ModelConfig  # noqa: E402
from llm.model.transformer import Transformer, causal_lm_loss  # noqa: E402
from llm.tokenizer import Tokenizer  # noqa: E402


PROBE_PROMPTS = (
    "The purpose of a voltage regulator is",
    "In Python, a dictionary is",
    "Newton's second law states that",
    "A database transaction should",
    "The operating system is responsible for",
)

PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {"temperature": 0.0, "top_k": None},
    "t0.7-k40": {"temperature": 0.7, "top_k": 40},
    "t0.8-k50": {"temperature": 0.8, "top_k": 50},
}

#: Where a checkpoint plausibly lives, best guess first.
CHECKPOINT_DIRS = (
    "artifacts/base-v0.1",
    "checkpoints/base-v0.1",
    "checkpoints",
    "artifacts",
)

#: Likewise for the tokenizer; E011 is the frozen one the corpus was built with.
TOKENIZER_CANDIDATES = ("artifacts/tokenizer-E011/tokenizer.json",)

DEFAULT_DATASET = "data/tokenized/v0.1"
STATE_DICT_PREFIXES = ("module.", "_orig_mod.")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def _step_number(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else -1


def discover_checkpoint(root: Path = REPO_ROOT) -> Path | None:
    """Find the most advanced checkpoint available.

    Ranked by step number encoded in the filename first, then modification
    time.  A run that wrote ``step-00002067.pt`` should win over one that wrote
    ``step-00000500.pt`` later, because the step count is what actually orders
    training progress.
    """

    found: list[Path] = []
    for relative in CHECKPOINT_DIRS:
        directory = root / relative
        if not directory.is_dir():
            continue
        found.extend(directory.glob("*.pt"))
        found.extend(directory.glob("*/*.pt"))

    unique = {path.resolve(): path for path in found if path.is_file()}
    if not unique:
        return None

    return max(unique.values(), key=lambda p: (_step_number(p), p.stat().st_mtime))


def discover_tokenizer(root: Path = REPO_ROOT) -> Path | None:
    for relative in TOKENIZER_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate

    # Fall back to the highest-numbered tokenizer generation present.
    generations = sorted(
        (p for p in (root / "artifacts").glob("tokenizer-*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for directory in reversed(generations):
        candidate = directory / "tokenizer.json"
        if candidate.is_file():
            return candidate
    return None


def report_missing_checkpoint(root: Path) -> None:
    print("No checkpoint found. Looked in:", file=sys.stderr)
    for relative in CHECKPOINT_DIRS:
        directory = root / relative
        mark = "exists but holds no .pt" if directory.is_dir() else "missing"
        print(f"  {directory}  ({mark})", file=sys.stderr)
    print(
        "\nIf the model was trained on a remote machine it still has to be "
        "downloaded. A 32.7M-parameter checkpoint is roughly 390 MB with "
        "optimizer state, or ~131 MB for weights alone.\n"
        "Then either drop it into artifacts/base-v0.1/ or pass --checkpoint.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def strip_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    """Undo wrappers that rename every key (DDP, torch.compile)."""

    cleaned = {}
    for key, value in state.items():
        for prefix in STATE_DICT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_model(path: Path, device: torch.device) -> tuple[Transformer, dict[str, Any]]:
    """Load either a training checkpoint or a bare state dict."""

    payload = torch.load(path, map_location=device, weights_only=True)
    metadata: dict[str, Any] = {}

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = {
            "step": payload.get("step"),
            "tokens_processed": payload.get("tokens_processed"),
            "created_at_utc": payload.get("created_at_utc"),
            "dataset_identity": payload.get("dataset_identity"),
        }
        recorded = payload.get("model_config")
        if isinstance(recorded, dict) and recorded:
            known = set(ModelConfig().to_dict())
            unknown = sorted(set(recorded) - known)
            if unknown:
                print(f"NOTE: ignoring unknown config fields: {unknown}", file=sys.stderr)
            config = ModelConfig(**{k: v for k, v in recorded.items() if k in known})
        else:
            config = ModelConfig()
    else:
        state = payload
        config = ModelConfig()
        print(
            "NOTE: bare state dict with no recorded architecture; "
            "assuming the frozen v0.1 ModelConfig.",
            file=sys.stderr,
        )

    model = Transformer(config)
    model.load_state_dict(strip_prefixes(state))
    model.to(device).eval()
    return model, metadata


@torch.no_grad()
def evaluate(model, dataset_dir: Path, device, batches: int, batch_size: int) -> dict:
    """Validation loss and perplexity on the frozen validation split."""

    from llm.data.dataset import PretrainingDataset
    from llm.training.data import DeterministicBatchSampler, collate_windows

    dataset = PretrainingDataset(
        dataset_dir / "validation" / "manifest.json",
        context_length=model.config.context_length,
        expected_vocab_size=model.config.vocab_size,
        verify_checksums=False,
    )
    sampler = DeterministicBatchSampler(len(dataset), batch_size, seed=2027)
    count = min(batches, len(sampler))

    total = 0.0
    for index in range(count):
        inputs, targets = collate_windows(
            dataset, sampler.indices_for_step(index), device
        )
        total += float(causal_lm_loss(model(inputs), targets))

    loss = total / count
    return {
        "validation_windows": len(dataset),
        "batches": count,
        "batch_size": batch_size,
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
    }


# --------------------------------------------------------------------------
# interactive session
# --------------------------------------------------------------------------


REPL_HELP = """
Commands
  /help                 show this
  /settings             show current decoding settings
  /temp <float>         sampling temperature (0 = greedy)
  /topk <int|off>       keep only the k most likely tokens
  /topp <float|off>     nucleus sampling threshold
  /tokens <int>         max new tokens per response
  /seed <int|off>       fix the sampler for reproducible output
  /greedy               shortcut for /temp 0
  /probe                run the built-in probe prompts
  /exit                 leave (Ctrl-C and Ctrl-D also work)

This is a base language model: it continues text, it does not answer
questions. Phrase prompts as the start of a sentence rather than as a request.
  good:  Ohm's law states that
  poor:  What is Ohm's law?
"""


def describe(config: SamplingConfig) -> str:
    mode = "greedy" if config.is_greedy else "sampling"
    return (
        f"{mode}: temperature={config.temperature}, top_k={config.top_k}, "
        f"top_p={config.top_p}, max_new_tokens={config.max_new_tokens}, "
        f"seed={config.seed}"
    )


def stream_generation(model, tokenizer, prompt: str, config: SamplingConfig):
    """Generate while printing tokens as they arrive."""

    decoder = StreamingDecoder(tokenizer)

    def emit(token_id: int) -> None:
        piece = decoder.push(token_id)
        if piece:
            print(piece, end="", flush=True)

    print(prompt, end="", flush=True)
    result = generate(model, tokenizer, prompt, config, on_token=emit)

    trailing = decoder.flush()
    if trailing:
        print(trailing, end="")
    print()

    notes = []
    if result.stopped_on_eos:
        notes.append("EOS")
    if result.decode_was_lossy:
        notes.append("lossy decode")
    print(
        f"  ({result.generated_token_count} tokens"
        + (f", {', '.join(notes)}" if notes else "")
        + ")\n"
    )
    return result


def apply_command(
    config: SamplingConfig, line: str, model=None, tokenizer=None, report=None
) -> tuple[SamplingConfig, bool]:
    """Interpret a /command. Returns (config, should_exit)."""

    parts = line.strip().split()
    command = parts[0].lower()
    argument = parts[1] if len(parts) > 1 else None

    def number(cast, name):
        if argument is None:
            print(f"  usage: {command} <{name}>")
            return None
        try:
            return cast(argument)
        except ValueError:
            print(f"  {argument!r} is not a valid {name}")
            return None

    try:
        if command in ("/exit", "/quit", "/q"):
            return config, True
        if command == "/help":
            print(REPL_HELP)
        elif command == "/settings":
            print(f"  {describe(config)}")
        elif command == "/probe":
            if model is None:
                print("  unavailable")
            else:
                print()
                for prompt in PROBE_PROMPTS:
                    result = stream_generation(model, tokenizer, prompt, config)
                    if report is not None:
                        report["generations"].append(result.to_dict())
        elif command == "/greedy":
            config = replace(config, temperature=0.0)
            print(f"  {describe(config)}")
        elif command == "/temp":
            value = number(float, "float")
            if value is not None:
                config = replace(config, temperature=value)
                print(f"  {describe(config)}")
        elif command == "/topk":
            if argument == "off":
                config = replace(config, top_k=None)
            else:
                value = number(int, "int")
                config = replace(config, top_k=value) if value is not None else config
            print(f"  {describe(config)}")
        elif command == "/topp":
            if argument == "off":
                config = replace(config, top_p=None)
            else:
                value = number(float, "float")
                config = replace(config, top_p=value) if value is not None else config
            print(f"  {describe(config)}")
        elif command == "/tokens":
            value = number(int, "int")
            if value is not None:
                config = replace(config, max_new_tokens=value)
                print(f"  {describe(config)}")
        elif command == "/seed":
            if argument == "off":
                config = replace(config, seed=None)
            else:
                value = number(int, "int")
                config = replace(config, seed=value) if value is not None else config
            print(f"  {describe(config)}")
        else:
            print(f"  unknown command {command!r}; /help for the list")
    except (ValueError, TypeError) as exc:
        # SamplingConfig validates on construction, so a rejected value is
        # reported here rather than ending the session.
        print(f"  rejected: {exc}")

    return config, False


def run_repl(model, tokenizer, config: SamplingConfig, report: dict[str, Any]) -> None:
    print("-" * 78)
    print("Interactive. Type a prompt, or /help for commands, /exit to leave.")
    print(f"  {describe(config)}")
    print()

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # Piped stdin often carries a UTF-8 BOM on the first line, and
        # str.strip() does not remove U+FEFF because it is not whitespace.
        # Left in, it hides the leading "/" and turns a command into a prompt.
        line = line.lstrip("﻿​‎‏")

        if not line.strip():
            continue
        if line.strip().startswith("/"):
            config, should_exit = apply_command(config, line, model, tokenizer, report)
            if should_exit:
                break
            continue

        try:
            result = stream_generation(model, tokenizer, line, config)
        except KeyboardInterrupt:
            print("\n  [interrupted]\n")
            continue
        report["generations"].append(result.to_dict())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from the trained base model. Run with no "
        "arguments for an interactive session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)

    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the prompt loop even after --prompt or --sweep (it is the "
        "default when neither is given).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Never open the prompt loop.",
    )

    parser.add_argument("--max-new-tokens", type=positive_int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-bos", action="store_true")

    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--eval-batches", type=positive_int, default=20)
    parser.add_argument("--eval-batch-size", type=positive_int, default=4)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def finish(report: dict[str, Any], args: argparse.Namespace) -> int:
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Report: {args.report}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        checkpoint = args.checkpoint or discover_checkpoint()
        if checkpoint is None:
            report_missing_checkpoint(REPO_ROOT)
            return 1
        if not checkpoint.is_file():
            print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
            return 1

        tokenizer_path = args.tokenizer or discover_tokenizer()
        if tokenizer_path is None or not tokenizer_path.is_file():
            print(
                "ERROR: no tokenizer found. Expected "
                f"{REPO_ROOT / TOKENIZER_CANDIDATES[0]}",
                file=sys.stderr,
            )
            return 1

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("ERROR: --device cuda requested but no CUDA is available.", file=sys.stderr)
            return 1

        tokenizer = Tokenizer.load(tokenizer_path)
        model, metadata = load_model(checkpoint, device)

        def relative(path: Path) -> str:
            try:
                return str(path.relative_to(REPO_ROOT))
            except ValueError:
                return str(path)

        print()
        print("GENERATION")
        print("=" * 78)
        print(f"Checkpoint:  {relative(checkpoint)}")
        print(f"Tokenizer:   {relative(tokenizer_path)}  (vocab {tokenizer.vocab_size:,})")
        print(f"Parameters:  {model.parameter_count():,}")
        print(f"Device:      {device}")
        if metadata.get("step") is not None:
            line = f"Trained to:  step {metadata['step']:,}"
            if metadata.get("tokens_processed"):
                line += f"  ({metadata['tokens_processed']:,} tokens)"
            print(line)
        print()

        report: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer_path),
            "parameters": model.parameter_count(),
            "checkpoint_metadata": metadata,
            "generations": [],
        }

        dataset_dir = args.dataset
        if dataset_dir is None and (REPO_ROOT / DEFAULT_DATASET).is_dir():
            dataset_dir = REPO_ROOT / DEFAULT_DATASET
        if dataset_dir is not None and (dataset_dir / "validation").is_dir():
            print("Validation pass")
            print("-" * 78)
            metrics = evaluate(
                model, dataset_dir, device, args.eval_batches, args.eval_batch_size
            )
            print(f"  loss        {metrics['loss']:.4f}")
            print(f"  perplexity  {metrics['perplexity']:,.2f}")
            print(f"  (over {metrics['batches']} batches of {metrics['batch_size']})")
            print()
            report["validation"] = metrics

        base = dict(
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            seed=args.seed,
            add_bos=not args.no_bos,
        )
        config = SamplingConfig(
            **base,
            temperature=0.0 if args.greedy else args.temperature,
            top_k=None if args.greedy else (args.top_k if args.top_k > 0 else None),
        )

        explicit_prompts: list[str] = list(args.prompt or [])
        if args.prompt_file:
            explicit_prompts.extend(
                line.strip()
                for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        if args.sweep:
            for prompt in explicit_prompts or list(PROBE_PROMPTS):
                print("-" * 78)
                for label, preset in PRESETS.items():
                    print(f"  [{label}]")
                    result = stream_generation(
                        model, tokenizer, prompt, SamplingConfig(**{**base, **preset})
                    )
                    report["generations"].append({"preset": label, **result.to_dict()})

        for prompt in explicit_prompts:
            print("-" * 78)
            result = stream_generation(model, tokenizer, prompt, config)
            report["generations"].append(result.to_dict())

        # A bare run means "let me talk to it". Naming prompts or --sweep means
        # "do this and stop", unless --interactive says otherwise.
        did_batch_work = bool(explicit_prompts) or args.sweep
        if not args.no_interactive and (args.interactive or not did_batch_work):
            run_repl(model, tokenizer, config, report)

        return finish(report, args)

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
