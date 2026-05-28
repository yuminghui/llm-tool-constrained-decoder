"""
Vocabulary token index for fast candidate lookup during constrained decoding.

Preprocesses a tokenizer's vocabulary into a first-character index,
enabling O(1) retrieval of tokens starting with a given character.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TokenEntry:
    """A single token in the vocabulary index."""
    token_id: int
    token_str: str
    first_char: str  # empty string for zero-length tokens


class TokenIndex:
    """First-character index over a tokenizer's vocabulary.

    Usage::

        index = TokenIndex(tokenizer)
        candidates = index.get_candidates({'a', 'b', '{'})
        # -> all tokens whose first decoded character is 'a', 'b', or '{'
    """

    def __init__(self, tokenizer):
        """Build the index from ``tokenizer``.

        Args:
            tokenizer: A HuggingFace tokenizer (PreTrainedTokenizer / PreTrainedTokenizerFast).
        """
        self._tokenizer = tokenizer
        self._first_char_to_tokens: Dict[str, List[TokenEntry]] = {}
        self._id_to_entry: Dict[int, TokenEntry] = {}
        self._build(tokenizer)

    # ---- properties -------------------------------------------------------

    @property
    def id_to_entry(self) -> Dict[int, TokenEntry]:
        return self._id_to_entry

    @property
    def first_char_to_tokens(self) -> Dict[str, List[TokenEntry]]:
        return self._first_char_to_tokens

    # ---- build ------------------------------------------------------------

    def _build(self, tokenizer) -> None:
        """Iterate over the full vocabulary and build the index."""
        vocab_size = tokenizer.vocab_size
        vocab = tokenizer.get_vocab()

        # For fast tokenizers, build a reverse mapping: token_str -> set of ids
        # For slow tokenizers, decode each id individually.
        # We use the fast path when possible.

        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            # HuggingFace tokenizers support batch conversion
            for token_id in range(vocab_size):
                self._add_token(token_id, tokenizer)
        else:
            # Fallback: iterate vocab dict
            for token_str, token_id in vocab.items():
                self._add_token_by_str(token_id, token_str)

    def _add_token(self, token_id: int, tokenizer) -> None:
        """Decode a single token and add it to the index."""
        try:
            # Decode the token in isolation (not as part of a sequence)
            token_str = tokenizer.decode([token_id])
        except Exception:
            token_str = ''

        self._add_token_by_str(token_id, token_str)

    def _add_token_by_str(self, token_id: int, token_str: str) -> None:
        """Add a token to the index by its string representation."""
        first_char = token_str[0] if token_str else ''

        entry = TokenEntry(
            token_id=token_id,
            token_str=token_str,
            first_char=first_char,
        )

        self._first_char_to_tokens.setdefault(first_char, []).append(entry)
        self._id_to_entry[token_id] = entry

    # ---- query ------------------------------------------------------------

    def get_candidates(self, allowed_first_chars: set) -> List[TokenEntry]:
        """Return all token entries whose first character is in *allowed_first_chars*."""
        candidates: List[TokenEntry] = []
        for ch in allowed_first_chars:
            candidates.extend(self._first_char_to_tokens.get(ch, ()))
        return candidates

    def get_entry(self, token_id: int) -> Optional[TokenEntry]:
        """Return the TokenEntry for *token_id*, or None."""
        return self._id_to_entry.get(token_id)
