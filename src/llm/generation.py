"""Autoregressive text generation.

Stateless and deliberately simple: the whole prefix is re-encoded on every
step.  That is O(n^2) in the length of the output and a KV cache would fix it,
but a cache is an optimization whose bugs look exactly like a badly trained
model -- subtly wrong continuations, no error anywhere.  Correct-and-slow comes
first; the cache can be added later and validated *against* this.

Two details that decide whether output looks like the training data
-------------------------------------------------------------------
**BOS is prepended.** Every document in the corpus was framed as
``<bos> content <eos>``, so the model has only ever seen text that starts after
a BOS. Generating from a bare prompt puts it in a state it never trained on,
and the first few tokens come out noticeably worse. ``add_bos=True`` is the
default for that reason.

**The context is cropped from the left.** Once the prompt plus what has been
generated exceeds ``context_length``, the model is fed only the most recent
2048 tokens. Without this, generation raises the moment it crosses the
boundary -- RoPE refuses positions it has no table for, which is the correct
behaviour there and the wrong behaviour here.

Sampling
--------
``temperature == 0`` means greedy: take the argmax. Otherwise logits are
divided by the temperature, optionally restricted to the ``top_k`` most likely
tokens and/or the smallest set whose cumulative probability reaches ``top_p``,
then sampled from. Both filters keep at least one candidate, so no combination
of settings can produce an empty distribution.

Undecodable ids are masked
--------------------------
The model's output layer spans the whole vocabulary, but a tokenizer does not
necessarily assign every id in that range -- when the merge budget exceeds what
the training corpus can support, some ids are reserved and unassigned. The
model can still emit them, and ``decode`` rightly refuses, so a single such
sample would abort an otherwise fine generation.

Those ids are therefore masked out of the distribution before sampling. This
is not a workaround for a decode bug: an id the tokenizer cannot represent is
not a valid output, and excluding it is more honest than sampling it and
failing afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SamplingConfig:
    """How to turn logits into the next token."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = None
    seed: int | None = None
    stop_on_eos: bool = True
    add_bos: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(
            self.max_new_tokens, bool
        ):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")

        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a number")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")

        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
                raise TypeError("top_k must be an integer or None")
            if self.top_k <= 0:
                raise ValueError(f"top_k must be > 0, got {self.top_k}")

        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                raise TypeError("top_p must be a number or None")
            if not 0.0 < self.top_p <= 1.0:
                raise ValueError(f"top_p must be within (0, 1], got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    """A completed generation and how it ended."""

    prompt: str
    completion: str
    prompt_token_count: int
    generated_token_count: int
    stopped_on_eos: bool
    token_ids: list[int] = field(default_factory=list)
    #: True when the ids could not be decoded cleanly even after trimming a
    #: truncated tail, so the text contains U+FFFD replacements.
    decode_was_lossy: bool = False

    @property
    def text(self) -> str:
        return self.prompt + self.completion

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text"] = self.text
        return payload


#: How many trailing tokens to drop while looking for a clean UTF-8 boundary.
#: A single character is at most four bytes, but one BPE token can carry
#: several bytes, so a small margin covers realistic truncation.
MAX_TRAILING_TRIM = 8


def _token_bytes(tokenizer, token_id: int) -> bytes:
    special = tokenizer.id_to_special_token.get(token_id)
    if special is not None:
        return special.encode("utf-8")
    data = tokenizer.bpe.vocab.get(token_id)
    return bytes(data) if data is not None else b""


def decode_generated(
    tokenizer, token_ids: Sequence[int], *, skip_special_tokens: bool = True
) -> tuple[str, bool]:
    """Decode generated ids into text; return ``(text, was_lossy)``.

    Byte-level BPE emits *bytes*, so stopping after a fixed number of tokens
    can land halfway through a multi-byte character.  That is ordinary
    behaviour for a generation that hit ``max_new_tokens``, not corruption, but
    the tokenizer's strict decoder correctly refuses it.

    So: try the strict decode, then retry dropping up to
    :data:`MAX_TRAILING_TRIM` trailing tokens to find a clean character
    boundary.  Only if that fails does it fall back to a lossy byte decode, and
    the caller is told, because at that point the ids really are malformed
    rather than merely truncated.
    """

    ids = list(token_ids)
    if not ids:
        return "", False

    for drop in range(0, min(len(ids), MAX_TRAILING_TRIM) + 1):
        head = ids[: len(ids) - drop]
        if not head:
            break
        try:
            return (
                tokenizer.decode(head, skip_special_tokens=skip_special_tokens),
                False,
            )
        except ValueError:
            continue

    raw = b"".join(
        _token_bytes(tokenizer, token_id)
        for token_id in ids
        if not (skip_special_tokens and token_id in tokenizer.id_to_special_token)
    )
    return raw.decode("utf-8", errors="replace"), True


class StreamingDecoder:
    """Turn a stream of token ids into text as it arrives.

    Printing each token as it is generated is what makes an interactive session
    feel responsive, but tokens cannot simply be decoded one at a time: a
    byte-level BPE token may carry a *fragment* of a multi-byte character, and
    decoding it alone either raises or emits a replacement character that then
    never gets corrected.

    So bytes are buffered and only whole characters are released.  A partial
    character stays in the buffer until the token completing it arrives.
    ``flush`` reports whatever is left at the end, lossily, since by then no
    further token is coming to complete it.
    """

    def __init__(self, tokenizer, *, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self._buffer = bytearray()

    def push(self, token_id: int) -> str:
        """Add one token; return whatever text is now complete (often "")."""

        if self.skip_special_tokens and token_id in self.tokenizer.id_to_special_token:
            return ""

        self._buffer.extend(_token_bytes(self.tokenizer, token_id))

        try:
            text = self._buffer.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start == 0:
                return ""
            text = self._buffer[: exc.start].decode("utf-8")
            del self._buffer[: exc.start]
            return text

        self._buffer.clear()
        return text

    def flush(self) -> str:
        """Release any trailing incomplete bytes, replacing what cannot decode."""

        if not self._buffer:
            return ""
        text = self._buffer.decode("utf-8", errors="replace")
        self._buffer.clear()
        return text


def undecodable_token_ids(tokenizer) -> list[int]:
    """Ids inside the model's output range that the tokenizer cannot decode.

    Content ids come from the learned BPE vocabulary and special ids from the
    reserved table; anything in ``[0, vocab_size)`` in neither is unassigned.
    """

    assigned = set(tokenizer.bpe.vocab) | set(tokenizer.id_to_special_token)
    return [i for i in range(tokenizer.vocab_size) if i not in assigned]


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask everything outside the ``top_k`` highest logits."""

    k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches ``top_p``."""

    ordered, indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)

    # Drop tokens once the mass is already covered, but never the first one:
    # a single token above top_p would otherwise leave nothing to sample.
    remove = cumulative - F.softmax(ordered, dim=-1) >= top_p
    remove[..., 0] = False

    ordered = ordered.masked_fill(remove, float("-inf"))
    return ordered.gather(-1, indices.argsort(-1))


def select_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> int:
    """Choose one token id from a ``(vocab_size,)`` logit vector."""

    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got {tuple(logits.shape)}")

    if config.temperature == 0.0:
        return int(torch.argmax(logits))

    scaled = logits.float() / config.temperature
    if config.top_k is not None:
        scaled = apply_top_k(scaled, config.top_k)
    if config.top_p is not None:
        scaled = apply_top_p(scaled, config.top_p)

    probabilities = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator))


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids: list[int],
    config: SamplingConfig,
    *,
    eos_id: int | None = None,
    device: torch.device | None = None,
    on_token: Callable[[int], None] | None = None,
    forbidden_ids: Sequence[int] | None = None,
) -> tuple[list[int], bool]:
    """Continue ``prompt_ids``; return ``(new_ids, stopped_on_eos)``."""

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    context_length = model.config.context_length
    target_device = device or next(model.parameters()).device

    forbidden = (
        torch.tensor(sorted(set(forbidden_ids)), dtype=torch.long)
        if forbidden_ids
        else None
    )

    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

    was_training = model.training
    model.eval()

    produced: list[int] = []
    running = list(prompt_ids)
    stopped = False

    try:
        for _ in range(config.max_new_tokens):
            # Left-crop: the model can only be given positions it has RoPE
            # tables for, and the most recent tokens are the relevant ones.
            window = running[-context_length:]
            inputs = torch.tensor([window], dtype=torch.long, device=target_device)

            # Sampling happens on CPU so a fixed seed reproduces regardless of
            # which device the weights happen to live on.
            logits = model(inputs)[0, -1].float().cpu()

            if forbidden is not None:
                logits = logits.index_fill(0, forbidden, float("-inf"))
                if not torch.isfinite(logits).any():
                    raise ValueError(
                        "every token id was masked out; the tokenizer and model "
                        "vocabularies do not overlap"
                    )

            next_id = select_next_token(logits, config, generator)

            if eos_id is not None and next_id == eos_id and config.stop_on_eos:
                stopped = True
                break

            produced.append(next_id)
            running.append(next_id)
            if on_token is not None:
                on_token(next_id)
    finally:
        if was_training:
            model.train()

    return produced, stopped


def generate(
    model,
    tokenizer,
    prompt: str,
    config: SamplingConfig | None = None,
    *,
    device: torch.device | None = None,
    on_token: Callable[[int], None] | None = None,
) -> GenerationResult:
    """Generate a continuation of ``prompt``."""

    settings = config or SamplingConfig()
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=settings.add_bos,
        add_eos=False,
        # Prompt text is content, not control tokens. Parsing "<|eos|>" typed
        # by a user into a real EOS would let a prompt end its own generation.
        parse_special_tokens=False,
    )
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")

    eos_id = tokenizer.special_token_to_id.get("<|eos|>")

    new_ids, stopped = generate_ids(
        model,
        prompt_ids,
        settings,
        eos_id=eos_id,
        device=device,
        on_token=on_token,
        forbidden_ids=undecodable_token_ids(tokenizer),
    )

    completion, was_lossy = decode_generated(tokenizer, new_ids)

    return GenerationResult(
        prompt=prompt,
        completion=completion,
        prompt_token_count=len(prompt_ids),
        generated_token_count=len(new_ids),
        stopped_on_eos=stopped,
        token_ids=new_ids,
        decode_was_lossy=was_lossy,
    )
