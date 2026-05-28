"""
Character-level Finite State Machine engine.

Provides NFA (with epsilon transitions) for flexible construction,
and DFA for efficient runtime token validation during constrained decoding.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Tuple, Optional


# ---------------------------------------------------------------------------
# Global NFA registry — maps NFA state id string -> NFAState object
# Needed because DFA states reference NFA states by id string.
# ---------------------------------------------------------------------------
_nfa_registry: Dict[str, "NFAState"] = {}


def _register(state: "NFAState") -> "NFAState":
    _nfa_registry[state.id] = state
    return state


def _get_nfa_state(sid: str) -> "NFAState":
    return _nfa_registry[sid]


def _clear_registry() -> None:
    """Clear the global NFA registry. Useful in tests."""
    _nfa_registry.clear()


# ---------------------------------------------------------------------------
# Sentinel for "match any character" (catch-all transition)
# Used in NFA/DFA to represent a transition that fires on any single character
# that does NOT have an explicit transition from the same state.
# ---------------------------------------------------------------------------
CATCHALL: str = "\x00__CATCHALL__"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NFAState:
    """A state in a non-deterministic finite automaton.

    Transitions are keyed by single-character strings.
    Epsilon transitions are stored separately.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    transitions: Dict[str, List["NFAState"]] = field(default_factory=dict)
    epsilon: List["NFAState"] = field(default_factory=list)
    is_accept: bool = False

    def add_transition(self, char: str, target: "NFAState") -> None:
        self.transitions.setdefault(char, []).append(target)

    def add_epsilon(self, target: "NFAState") -> None:
        self.epsilon.append(target)


@dataclass
class DFAState:
    """A state in a deterministic finite automaton.

    - ``id``: unique integer id
    - ``nfa_states``: frozenset of NFA state ids that this DFA state represents
    - ``transitions``: dict char -> DFAState (may include CATCHALL key)
    - ``is_accept``: True if any constituent NFA state is accepting
    - ``can_reach_accept``: precomputed by compute_reachability()
    """
    id: int
    nfa_states: FrozenSet[str] = field(default_factory=frozenset)
    transitions: Dict[str, "DFAState"] = field(default_factory=dict)
    is_accept: bool = False
    can_reach_accept: bool = False

    def next_state(self, char: str) -> "DFAState | None":
        """Return the next DFA state for ``char``, handling CATCHALL fallback."""
        if char in self.transitions:
            return self.transitions[char]
        if CATCHALL in self.transitions:
            return self.transitions[CATCHALL]
        return None


# ---------------------------------------------------------------------------
# Epsilon closure
# ---------------------------------------------------------------------------

def epsilon_closure(states: List[NFAState]) -> FrozenSet[str]:
    """Compute the epsilon-closure of a set of NFA states.

    Returns a frozenset of NFA state id strings reachable via zero or more
    epsilon transitions.
    """
    stack = list(states)
    result: Set[str] = {s.id for s in states}

    while stack:
        state = stack.pop()
        for next_s in state.epsilon:
            if next_s.id not in result:
                result.add(next_s.id)
                stack.append(next_s)

    return frozenset(result)


# ---------------------------------------------------------------------------
# NFA -> DFA subset construction
# ---------------------------------------------------------------------------

def nfa_to_dfa(start_nfa: NFAState) -> Tuple[DFAState, List[DFAState]]:
    """Convert an NFA to a DFA via subset construction.

    Args:
        start_nfa: The start state of the NFA.

    Returns:
        (start_dfa_state, list_of_all_dfa_states)
    """
    start_closure = epsilon_closure([start_nfa])

    def _closure_is_accept(closure: FrozenSet[str]) -> bool:
        return any(_get_nfa_state(sid).is_accept for sid in closure)

    start_dfa = DFAState(
        id=0,
        nfa_states=start_closure,
        is_accept=_closure_is_accept(start_closure),
    )

    dfa_states: List[DFAState] = [start_dfa]
    state_map: Dict[FrozenSet[str], DFAState] = {start_closure: start_dfa}
    worklist: List[DFAState] = [start_dfa]
    next_id = 1

    while worklist:
        current = worklist.pop()

        # Collect all explicit characters (excluding CATCHALL) with transitions
        all_chars: Set[str] = set()
        has_catchall = False
        for nfa_sid in current.nfa_states:
            nfa_s = _get_nfa_state(nfa_sid)
            for ch in nfa_s.transitions:
                if ch == CATCHALL:
                    has_catchall = True
                else:
                    all_chars.add(ch)

        for char in all_chars:
            # Target NFA states: explicit transition if present,
            # otherwise fall back to CATCHALL.  CATCHALL must not fire
            # for characters that have an explicit transition from the
            # same NFA state.
            target_nfa_list: List[NFAState] = []
            for nfa_sid in current.nfa_states:
                nfa_s = _get_nfa_state(nfa_sid)
                if char in nfa_s.transitions:
                    target_nfa_list.extend(nfa_s.transitions[char])
                elif CATCHALL in nfa_s.transitions:
                    target_nfa_list.extend(nfa_s.transitions[CATCHALL])

            if not target_nfa_list:
                continue

            target_closure = epsilon_closure(target_nfa_list)

            if target_closure not in state_map:
                new_dfa = DFAState(
                    id=next_id,
                    nfa_states=target_closure,
                    is_accept=_closure_is_accept(target_closure),
                )
                next_id += 1
                state_map[target_closure] = new_dfa
                dfa_states.append(new_dfa)
                worklist.append(new_dfa)

            current.transitions[char] = state_map[target_closure]

        # Handle catchall: build target for characters not explicitly handled
        # (only needed if some NFA state has a CATCHALL transition)
        if has_catchall:
            target_nfa_list_ca: List[NFAState] = []
            for nfa_sid in current.nfa_states:
                nfa_s = _get_nfa_state(nfa_sid)
                catchall_targets = nfa_s.transitions.get(CATCHALL, [])
                target_nfa_list_ca.extend(catchall_targets)

            if target_nfa_list_ca:
                ca_closure = epsilon_closure(target_nfa_list_ca)
                if ca_closure not in state_map:
                    new_dfa = DFAState(
                        id=next_id,
                        nfa_states=ca_closure,
                        is_accept=_closure_is_accept(ca_closure),
                    )
                    next_id += 1
                    state_map[ca_closure] = new_dfa
                    dfa_states.append(new_dfa)
                    worklist.append(new_dfa)

                current.transitions[CATCHALL] = state_map[ca_closure]

    # Precompute reachability for masked-token logic
    compute_reachability(dfa_states)

    return start_dfa, dfa_states


# ---------------------------------------------------------------------------
# Reachability — can each DFA state reach an accept state?
# ---------------------------------------------------------------------------

def compute_reachability(dfa_states: List[DFAState]) -> None:
    """Set ``can_reach_accept`` on every DFA state via reverse BFS from accept states.

    Considers both explicit transitions and catchall (CATCHALL key).
    """
    # Build reverse adjacency: target_id -> set of source_ids
    reverse: Dict[int, Set[int]] = {}
    for state in dfa_states:
        seen: Set[int] = set()
        for key, target in state.transitions.items():
            if target.id not in seen:
                seen.add(target.id)
                reverse.setdefault(target.id, set()).add(state.id)

    # BFS backwards from accept states
    can_reach: Set[int] = {s.id for s in dfa_states if s.is_accept}
    queue = list(can_reach)

    while queue:
        cur = queue.pop(0)
        for prev_id in reverse.get(cur, ()):
            if prev_id not in can_reach:
                can_reach.add(prev_id)
                queue.append(prev_id)

    for state in dfa_states:
        state.can_reach_accept = state.id in can_reach


# ---------------------------------------------------------------------------
# NFA composition utilities
# ---------------------------------------------------------------------------

def build_literal_nfa(s: str) -> Tuple[NFAState, NFAState]:
    """Build a linear NFA that matches the exact literal string ``s``.

    Returns ``(start_state, accept_state)``.
    """
    if not s:
        # Epsilon-only NFA (matches empty string)
        start = _register(NFAState())
        accept = _register(NFAState(is_accept=True))
        start.add_epsilon(accept)
        return start, accept

    states = [_register(NFAState()) for _ in range(len(s) + 1)]
    for i, ch in enumerate(s):
        states[i].add_transition(ch, states[i + 1])
    states[-1].is_accept = True
    return states[0], states[-1]


def nfa_concat(*nfas: Tuple[NFAState, NFAState]) -> Tuple[NFAState, NFAState]:
    """Concatenate multiple NFAs end-to-end.

    Returns ``(start, accept)`` for the concatenated NFA.
    """
    if not nfas:
        start = _register(NFAState())
        accept = _register(NFAState(is_accept=True))
        start.add_epsilon(accept)
        return start, accept

    cur_start, cur_accept = nfas[0]
    for next_start, next_accept in nfas[1:]:
        cur_accept.add_epsilon(next_start)
        cur_accept.is_accept = False
        cur_accept = next_accept
    return cur_start, cur_accept


def nfa_union(*nfas: Tuple[NFAState, NFAState]) -> Tuple[NFAState, NFAState]:
    """Build an NFA that matches any one of the input NFAs (alternation).

    Returns ``(start, accept)`` for the union NFA.
    """
    if not nfas:
        start = _register(NFAState())
        accept = _register(NFAState(is_accept=True))
        start.add_epsilon(accept)
        return start, accept

    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    for sub_start, sub_accept in nfas:
        start.add_epsilon(sub_start)
        sub_accept.add_epsilon(accept)
        sub_accept.is_accept = False

    return start, accept


def nfa_star(nfa: Tuple[NFAState, NFAState]) -> Tuple[NFAState, NFAState]:
    """Kleene star: zero or more repetitions of ``nfa``."""
    sub_start, sub_accept = nfa
    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    start.add_epsilon(sub_start)   # enter the loop
    start.add_epsilon(accept)      # zero repetitions

    sub_accept.add_epsilon(sub_start)  # loop back
    sub_accept.add_epsilon(accept)     # exit after one or more
    sub_accept.is_accept = False

    return start, accept


def nfa_optional(nfa: Tuple[NFAState, NFAState]) -> Tuple[NFAState, NFAState]:
    """Make ``nfa`` optional (zero or one occurrence)."""
    sub_start, sub_accept = nfa
    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    start.add_epsilon(sub_start)   # use it
    start.add_epsilon(accept)      # skip it

    sub_accept.add_epsilon(accept)
    sub_accept.is_accept = False

    return start, accept


def nfa_plus(nfa: Tuple[NFAState, NFAState]) -> Tuple[NFAState, NFAState]:
    """One or more repetitions of ``nfa``."""
    sub_start, sub_accept = nfa
    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    start.add_epsilon(sub_start)   # must go at least once

    sub_accept.add_epsilon(sub_start)  # loop back
    sub_accept.add_epsilon(accept)     # exit
    sub_accept.is_accept = False

    return start, accept


# ---------------------------------------------------------------------------
# DFA acceptance check (for testing)
# ---------------------------------------------------------------------------

def dfa_accepts(dfa_start: DFAState, text: str) -> bool:
    """Check whether the DFA accepts ``text`` (handles CATCHALL)."""
    state = dfa_start
    for ch in text:
        nxt = state.next_state(ch)
        if nxt is None:
            return False
        state = nxt
    return state.is_accept


def finalize_nfa(root_accept: NFAState) -> None:
    """Clear ``is_accept`` from all registered NFA states except *root_accept*.

    Call this after building a complete NFA to ensure only the top-level
    accept state counts.  Intermediate sub-NFA accept states would otherwise
    leak ``is_accept=True`` into the DFA and cause premature acceptance.
    """
    for sid, state in _nfa_registry.items():
        if state is not root_accept:
            state.is_accept = False


def dfa_next_state(state: DFAState, char: str) -> DFAState | None:
    """Convenience wrapper for state.next_state(char)."""
    return state.next_state(char)


def dfa_walk(state: DFAState, text: str) -> DFAState | None:
    """Walk the DFA with ``text``. Returns the resulting state or None if invalid."""
    cur = state
    for ch in text:
        cur = cur.next_state(ch)
        if cur is None:
            return None
    return cur
