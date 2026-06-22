"""
HuggingFace LogitsProcessor for tool-constrained decoding.

Plugs into ``model.generate(logits_processor=[...])`` to mask tokens that
would lead the DFA into a dead-end state.
"""

from __future__ import annotations

import logging
from typing import Dict, Set, Tuple, Optional

import torch
from transformers import LogitsProcessor

from .char_fsm import DFAState, dfa_walk
from .token_index import TokenIndex

logger = logging.getLogger(__name__)


class ToolConstraintLogitsProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` that enforces a tool-call DFA constraint.

    At each generation step:

    1. Advance the internal DFA state for any newly generated tokens.
    2. If the DFA has already visited an accept state, force EOS.
    3. Otherwise, for each token in the vocabulary:
       - Check if the token string is a valid continuation from the current DFA state.
       - If not, set its logit to -inf.
    4. Cache results per ``(dfa_state_id, token_str)`` for efficiency.

    Usage::

        processor = ToolConstraintLogitsProcessor(
            dfa_start=dfa_start,
            token_index=token_index,
            prompt_length=prompt_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = model.generate(input_ids, logits_processor=[processor], ...)
    """

    def __init__(
        self,
        dfa_start: DFAState,
        token_index: TokenIndex,
        prompt_length: int,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        all_dfa_states: Optional[list] = None,
        allowed_start_chars: Optional[Set[str]] = None,
    ):
        """
        Args:
            dfa_start: Start state of the tool-call DFA.
            token_index: Pre-built vocabulary index.
            prompt_length: Length of the prompt in tokens (used to detect new tokens).
            eos_token_id: Tokenizer's EOS token id.
            pad_token_id: Tokenizer's PAD token id (masked unconditionally).
            all_dfa_states: All DFA states (for pre-computing per-state valid masks).
            allowed_start_chars: Optional pre-computed first-character set for optimization.
        """
        self.dfa_start = dfa_start
        self.current_dfa_state = dfa_start
        self.token_index = token_index
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.last_input_len = prompt_length
        self.has_accepted = False

        # Pre-compute allowed start chars for the start state
        self.allowed_start_chars = (
            allowed_start_chars or set(dfa_start.transitions.keys())
        )

        # Cache: (dfa_state_id, token_str) -> (is_valid, next_state_id)
        self._cache: Dict[Tuple[int, str], Tuple[bool, int]] = {}

        # Cache miss stats for debugging
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-compute per-state valid-token masks (major speedup).
        # Instead of iterating the full vocabulary at every decoding step,
        # we build boolean masks once and apply them with a single
        # tensor operation per step.
        self._state_masks: Dict[int, torch.Tensor] = {}
        if all_dfa_states is not None:
            self._build_state_masks(all_dfa_states, token_index)

    # ------------------------------------------------------------------
    # Mask pre-computation
    # ------------------------------------------------------------------

    def _build_state_masks(
        self, all_dfa_states: list, token_index: TokenIndex
    ) -> None:
        """Pre-compute valid-token boolean masks for every DFA state.

        Walks each DFA state against the vocabulary once.  The resulting
        masks are applied in O(1) at each generation step, replacing the
        O(vocab_size) per-step loop.
        """
        from .char_fsm import CATCHALL

        for dfa_state in all_dfa_states:
            allowed_first = set(dfa_state.transitions.keys())
            has_catchall = CATCHALL in allowed_first

            # Collect valid token IDs for this state
            valid_ids: set = set()

            for first_ch, entries in token_index.first_char_to_tokens.items():
                if first_ch not in allowed_first and not has_catchall:
                    continue
                for entry in entries:
                    # Walk the DFA with this token's string
                    state = dfa_state
                    ok = True
                    for ch in entry.token_str:
                        nxt = state.next_state(ch)
                        if nxt is None:
                            ok = False
                            break
                        state = nxt
                    if ok and state.can_reach_accept:
                        valid_ids.add(entry.token_id)

            # Build a boolean tensor (CPU, converted to GPU on use)
            mask = torch.zeros(
                len(token_index.id_to_entry), dtype=torch.bool
            )
            for tid in valid_ids:
                if tid < len(mask):
                    mask[tid] = True

            self._state_masks[dfa_state.id] = mask
            logger.debug(
                "State %d mask: %d / %d tokens valid",
                dfa_state.id, len(valid_ids), len(token_index.id_to_entry),
            )

    # ------------------------------------------------------------------
    # LogitsProcessor interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,    # (batch_size, seq_len)
        scores: torch.FloatTensor,      # (batch_size, vocab_size)
    ) -> torch.FloatTensor:
        """Called by HuggingFace ``generate()`` at each decoding step."""
        # --- Step 1: advance state for new tokens ---
        current_len = input_ids.shape[1]
        if current_len > self.last_input_len:
            new_token_ids = input_ids[0, self.last_input_len:current_len]
            for tid in new_token_ids:
                self._advance_state(tid.item())
            self.last_input_len = current_len

        # --- Step 2: after accept, force EOS ---
        if self.has_accepted:
            scores[0, :] = -float('inf')
            if self.eos_token_id < scores.shape[1]:
                scores[0, self.eos_token_id] = 0.0
            return scores

        # --- Step 3: mask invalid tokens ---
        vocab_size = scores.shape[1]
        state_id = self.current_dfa_state.id

        # Use pre-computed mask when available (fast path)
        if state_id in self._state_masks:
            mask = self._state_masks[state_id]
            # Trim or pad mask to match vocab_size
            if len(mask) >= vocab_size:
                mask = mask[:vocab_size]
            else:
                pad = torch.zeros(vocab_size - len(mask), dtype=torch.bool)
                mask = torch.cat([mask, pad])
            scores[0, ~mask.to(scores.device)] = -float('inf')
        else:
            # Slow path — fallback (rarely hit once masks are built)
            allowed_first = set(self.current_dfa_state.transitions.keys())
            from .char_fsm import CATCHALL
            for token_id in range(vocab_size):
                entry = self.token_index.id_to_entry.get(token_id)
                if entry is None or not entry.token_str:
                    scores[0, token_id] = -float('inf')
                    continue
                if entry.first_char not in allowed_first and CATCHALL not in allowed_first:
                    scores[0, token_id] = -float('inf')
                    continue
                if not self._is_token_valid(entry.token_str):
                    scores[0, token_id] = -float('inf')

        # --- Step 4: special token handling ---
        if self.eos_token_id < vocab_size:
            scores[0, self.eos_token_id] = -float('inf')
        if self.pad_token_id is not None and self.pad_token_id < vocab_size:
            scores[0, self.pad_token_id] = -float('inf')

        return scores

    # ------------------------------------------------------------------
    # Internal state tracking
    # ------------------------------------------------------------------

    def _advance_state(self, token_id: int) -> None:
        """Walk the DFA forward by the string of *token_id*."""
        entry = self.token_index.id_to_entry.get(token_id)
        if entry is None:
            return

        token_str = entry.token_str
        if not token_str:
            return

        state = self.current_dfa_state
        for ch in token_str:
            nxt = state.next_state(ch)
            if nxt is None:
                # Token contains an invalid character — should not happen
                # if our masking is correct, but handle gracefully.
                logger.warning(
                    "DFA state advancement failed: no transition for %r in state %d. "
                    "Token: %r (id=%d)",
                    ch, state.id, token_str, token_id,
                )
                return
            state = nxt
            if state.is_accept:
                self.has_accepted = True

        self.current_dfa_state = state

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def _is_token_valid(self, token_str: str) -> bool:
        """Check if *token_str* is a valid continuation from the current DFA state.

        A token is valid iff:
        1. Every character has a valid transition from the current state.
        2. The final state can reach an accept state (not a dead end).

        Results are cached by (current_state_id, token_str).
        """
        cache_key = (self.current_dfa_state.id, token_str)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            is_valid, _ = cached
            return is_valid

        self._cache_misses += 1

        state = self.current_dfa_state
        for ch in token_str:
            nxt = state.next_state(ch)
            if nxt is None:
                self._cache[cache_key] = (False, -1)
                return False
            state = nxt

        result = state.can_reach_accept
        self._cache[cache_key] = (result, state.id)
        return result

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss counters."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache),
        }

    def reset_stats(self) -> None:
        """Zero out cache counters (cache entries are preserved)."""
        self._cache_hits = 0
        self._cache_misses = 0
