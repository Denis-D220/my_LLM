"""Text normalization utilities for the LLM tokenizer.

Tokenizer v0.1 normalization policy
-----------------------------------
* Input must be a Python ``str`` containing valid Unicode.
* Unicode is normalized with NFC.
* Letter case is preserved exactly; no lowercasing/case folding is applied.
* CRLF and legacy CR line endings are converted to LF.
* Spaces, tabs, newlines, punctuation, symbols, and technical notation are
  otherwise preserved.

The tokenizer is designed to be reversible *after normalization*:

    tokenizer.decode(tokenizer.encode(text)) == normalize_text(text)
"""

from __future__ import annotations

import unicodedata


UNICODE_NORMALIZATION_FORM = "NFC"


def normalize_text(text: str) -> str:
    """Normalize text before byte-level BPE tokenization.

    Parameters
    ----------
    text:
        Input Unicode text.

    Returns
    -------
    str
        NFC-normalized text with LF line endings.

    Raises
    ------
    TypeError
        If *text* is not a ``str``.
    UnicodeEncodeError
        If *text* contains an invalid/unpaired Unicode surrogate that cannot
        be represented as strict UTF-8.

    Notes
    -----
    This function deliberately does **not**:

    * lowercase text;
    * case-fold text;
    * collapse spaces;
    * collapse blank lines;
    * strip leading/trailing whitespace;
    * replace technical Unicode symbols with ASCII approximations.

    Those transformations could destroy information important to a technical
    language model (for example ``MHz`` vs ``mHz`` or code indentation).
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Validate that the Python string is representable as strict UTF-8.  A
    # Python str can technically contain an unpaired surrogate; our byte-level
    # tokenizer intentionally rejects such malformed Unicode rather than
    # silently replacing it.
    text.encode("utf-8", errors="strict")

    # Canonical normalization keeps technical compatibility characters more
    # faithfully than NFKC while making canonically equivalent sequences use a
    # stable representation (e.g. e + combining acute -> é where applicable).
    normalized = unicodedata.normalize(UNICODE_NORMALIZATION_FORM, text)

    # Normalize platform-specific line endings without disturbing any other
    # whitespace.  Order matters: consume CRLF first, then lone CR.
    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")

    return normalized

