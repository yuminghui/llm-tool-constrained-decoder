"""
Tool-Constrained Decoding for LLM Inference
============================================

A hot-swappable, training-free constrained decoder that forces an LLM to output
only a specific tool call during autoregressive generation.

Quick start::

    from constrained_decoding import ToolConstrainedDecoder

    decoder = ToolConstrainedDecoder(model, tokenizer)
    result = decoder.generate(
        prompt="<your prompt>",
        tool_name="get_weather",
        args_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    )
    print(result.tool_call)
"""

from .decoder import ToolConstrainedDecoder, DecoderResult
from .logits_processor import ToolConstraintLogitsProcessor
from .json_schema_to_fsm import build_tool_call_nfa, build_value_nfa
from .char_fsm import (
    NFAState,
    DFAState,
    CATCHALL,
    nfa_to_dfa,
    epsilon_closure,
    dfa_accepts,
    dfa_next_state,
    dfa_walk,
    _clear_registry,
)
from .token_index import TokenIndex, TokenEntry

__all__ = [
    # High-level API
    'ToolConstrainedDecoder',
    'DecoderResult',
    # Logits processor
    'ToolConstraintLogitsProcessor',
    # NFA builders
    'build_tool_call_nfa',
    'build_value_nfa',
    # FSM engine
    'NFAState',
    'DFAState',
    'CATCHALL',
    'nfa_to_dfa',
    'epsilon_closure',
    'dfa_accepts',
    'dfa_next_state',
    'dfa_walk',
    '_clear_registry',
    # Token index
    'TokenIndex',
    'TokenEntry',
]
