"""Public byte-level BPE tokenizer used by the LLM project.

The :class:`Tokenizer` is the stable boundary between text and the neural
network.  It owns:

* NFC text normalization;
* case preservation;
* byte-level BPE training/application;
* reserved special tokens;
* encode/decode round trips;
* deterministic JSON serialization.

The neural network should depend on this API rather than on BPE internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import json
import re

try:  # Package import: llm.tokenizer.tokenizer
    from .bpe import BYTE_VOCAB_SIZE, BPETrainer
    from .normalizer import UNICODE_NORMALIZATION_FORM, normalize_text
except ImportError:  # Standalone development from the same directory
    from bpe import BYTE_VOCAB_SIZE, BPETrainer
    from normalizer import UNICODE_NORMALIZATION_FORM, normalize_text


TOKENIZER_FORMAT = "llm_byte_level_bpe_tokenizer"
TOKENIZER_FORMAT_VERSION = 1

DEFAULT_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|tool|>",
    "<|tool_result|>",
]


class Tokenizer:
    """Case-preserving, Unicode-safe byte-level BPE tokenizer."""

    def __init__(
        self,
        *,
        bpe: BPETrainer,
        vocab_size: int,
        special_tokens: Sequence[str] | None = None,
    ):
        if not isinstance(bpe, BPETrainer):
            raise TypeError("bpe must be a BPETrainer")
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")

        special_tokens = list(special_tokens or [])
        self._validate_special_tokens(special_tokens)

        minimum_size = BYTE_VOCAB_SIZE + len(special_tokens)
        if vocab_size < minimum_size:
            raise ValueError(
                f"vocab_size must be at least {minimum_size} for 256 byte "
                f"tokens plus {len(special_tokens)} special tokens"
            )

        # The BPE content id space ends immediately before the reserved special
        # token block.  This keeps every token id inside [0, vocab_size).
        content_vocab_limit = vocab_size - len(special_tokens)
        if bpe.vocab_size != content_vocab_limit:
            raise ValueError(
                "BPE vocab size does not match tokenizer content vocabulary: "
                f"expected {content_vocab_limit}, got {bpe.vocab_size}"
            )

        self.bpe = bpe
        self._vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.content_vocab_limit = content_vocab_limit

        self.special_token_to_id: dict[str, int] = {
            token: content_vocab_limit + index
            for index, token in enumerate(special_tokens)
        }
        self.id_to_special_token: dict[int, str] = {
            token_id: token
            for token, token_id in self.special_token_to_id.items()
        }

        # Match the longest special token first if one token is a prefix of
        # another.  re.escape prevents token characters from becoming regex.
        if special_tokens:
            ordered = sorted(special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in ordered) + ")"
            )
        else:
            self._special_pattern = None

    # ------------------------------------------------------------------
    # Construction / training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 24_000,
        special_tokens: Sequence[str] | None = None,
        *,
        min_pair_frequency: int = 1,
    ) -> "Tokenizer":
        """Train a tokenizer from text.

        ``vocab_size`` is the **total** vocabulary size, including special
        tokens.  For example, with 9 special tokens and ``vocab_size=24000``,
        ids ``0..23990`` are reserved for bytes/BPE content and ids
        ``23991..23999`` are special tokens.
        """

        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")

        if special_tokens is None:
            special_tokens = list(DEFAULT_SPECIAL_TOKENS)
        else:
            special_tokens = list(special_tokens)

        cls._validate_special_tokens(special_tokens)

        content_vocab_limit = vocab_size - len(special_tokens)
        if content_vocab_limit < BYTE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size={vocab_size} is too small: byte-level BPE needs "
                f"256 content ids plus {len(special_tokens)} special ids"
            )

        # Normalize once before BPE training.  Keeping each input item separate
        # prevents merges from crossing document boundaries.
        normalized_texts: list[str] = []
        for text in texts:
            normalized_texts.append(normalize_text(text))

        bpe = BPETrainer(
            vocab_size=content_vocab_limit,
            min_pair_frequency=min_pair_frequency,
        ).train(normalized_texts)

        return cls(
            bpe=bpe,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        parse_special_tokens: bool = True,
    ) -> list[int]:
        """Normalize and encode text into integer token ids.

        Parameters
        ----------
        text:
            Unicode input text.
        add_bos / add_eos:
            Optionally add the corresponding reserved token ids.  A clear
            ``ValueError`` is raised if the requested special token was not
            reserved by this tokenizer.
        parse_special_tokens:
            When true (default), literal reserved strings such as
            ``<|assistant|>`` map to their single special ids.  When false,
            those characters are encoded as ordinary UTF-8/BPE content.
        """

        normalized = normalize_text(text)
        token_ids: list[int] = []

        if add_bos:
            token_ids.append(self._require_special_id("<|bos|>"))

        if (
            parse_special_tokens
            and self._special_pattern is not None
            and normalized
        ):
            # Keep delimiters in the split output because the regex contains a
            # capture group.
            pieces = self._special_pattern.split(normalized)
            for piece in pieces:
                if not piece:
                    continue
                special_id = self.special_token_to_id.get(piece)
                if special_id is not None:
                    token_ids.append(special_id)
                else:
                    token_ids.extend(
                        self.bpe.encode_bytes(piece.encode("utf-8", errors="strict"))
                    )
        else:
            token_ids.extend(
                self.bpe.encode_bytes(normalized.encode("utf-8", errors="strict"))
            )

        if add_eos:
            token_ids.append(self._require_special_id("<|eos|>"))

        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token ids back to Unicode text.

        The output is the normalized text represented by the ids.  By default,
        special ids decode to their literal strings so the core round-trip
        invariant also holds for chat-formatted text.
        """

        output = bytearray()

        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(
                    f"token ids must be integers, got {type(token_id).__name__}"
                )
            if token_id < 0 or token_id >= self._vocab_size:
                raise ValueError(
                    f"token id {token_id} is outside tokenizer vocabulary "
                    f"[0, {self._vocab_size})"
                )

            special = self.id_to_special_token.get(token_id)
            if special is not None:
                if not skip_special_tokens:
                    output.extend(special.encode("utf-8"))
                continue

            # Some ids between learned_vocab_size and content_vocab_limit may
            # be unassigned when the training corpus is too small to learn all
            # requested merges.  Rejecting them catches corrupted model output
            # instead of decoding arbitrary bytes.
            token_bytes = self.bpe.vocab.get(token_id)
            if token_bytes is None:
                raise ValueError(
                    f"token id {token_id} is reserved/unassigned in the content "
                    "vocabulary"
                )
            output.extend(token_bytes)

        try:
            return bytes(output).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "token id sequence does not form valid UTF-8; it may be "
                "truncated or corrupted"
            ) from exc

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_special_tokens(tokens: Sequence[str]) -> None:
        seen: set[str] = set()
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("special tokens must be strings")
            if not token:
                raise ValueError("special tokens cannot be empty")
            # Validate strict UTF-8 and normalize.  Requiring callers to supply
            # already-normalized forms avoids surprising lookup differences.
            normalized = normalize_text(token)
            if normalized != token:
                raise ValueError(
                    f"special token must already be NFC/LF normalized: {token!r}"
                )
            if token in seen:
                raise ValueError(f"duplicate special token: {token!r}")
            seen.add(token)

    def _require_special_id(self, token: str) -> int:
        try:
            return self.special_token_to_id[token]
        except KeyError as exc:
            raise ValueError(f"special token {token!r} is not reserved") from exc

    def token_to_id(self, token: str) -> int | None:
        """Return a reserved special-token id, or ``None`` if not special."""

        return self.special_token_to_id.get(token)

    def id_to_token(self, token_id: int) -> str | bytes | None:
        """Inspect one token without changing tokenizer state.

        Returns a special-token string, raw bytes for a byte/BPE token, or
        ``None`` for an unassigned vocabulary id.
        """

        if token_id in self.id_to_special_token:
            return self.id_to_special_token[token_id]
        return self.bpe.vocab.get(token_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return deterministic JSON-serializable tokenizer state."""

        return {
            "format": TOKENIZER_FORMAT,
            "format_version": TOKENIZER_FORMAT_VERSION,
            "vocab_size": self._vocab_size,
            "normalization": {
                "encoding": "UTF-8",
                "unicode_form": UNICODE_NORMALIZATION_FORM,
                "preserve_case": True,
                "case_folding": False,
                "line_endings": "LF",
            },
            "special_tokens": self.special_tokens,
            "bpe": self.bpe.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """Serialize the complete tokenizer into one UTF-8 JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load a tokenizer saved by :meth:`save`."""

        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")
        if payload.get("format") != TOKENIZER_FORMAT:
            raise ValueError("unsupported or missing tokenizer format")
        if payload.get("format_version") != TOKENIZER_FORMAT_VERSION:
            raise ValueError(
                f"unsupported tokenizer format version: "
                f"{payload.get('format_version')!r}"
            )

        normalization = payload.get("normalization", {})
        expected_normalization = {
            "encoding": "UTF-8",
            "unicode_form": UNICODE_NORMALIZATION_FORM,
            "preserve_case": True,
            "case_folding": False,
            "line_endings": "LF",
        }
        if normalization != expected_normalization:
            raise ValueError(
                "tokenizer normalization policy is incompatible with this code: "
                f"{normalization!r}"
            )

        bpe = BPETrainer.from_dict(payload["bpe"])

        return cls(
            bpe=bpe,
            vocab_size=int(payload["vocab_size"]),
            special_tokens=payload.get("special_tokens", []),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total model vocabulary size, including special/unassigned ids."""

        return self._vocab_size

    @property
    def learned_content_vocab_size(self) -> int:
        """Number of usable byte/BPE content token ids learned so far."""

        return self.bpe.learned_vocab_size

    @property
    def special_token_count(self) -> int:
        return len(self.special_tokens)

    def __len__(self) -> int:
        return self._vocab_size

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self._vocab_size}, "
            f"learned_content_vocab_size={self.learned_content_vocab_size}, "
            f"special_tokens={len(self.special_tokens)})"
        )