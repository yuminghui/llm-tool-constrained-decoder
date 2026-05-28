"""
Unit tests for the constrained decoding system.

Tests the FSM engine, JSON Schema -> NFA -> DFA pipeline,
token index construction, and the logits processor.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from constrained_decoding.char_fsm import (
    NFAState,
    DFAState,
    CATCHALL,
    _register,
    _get_nfa_state,
    _clear_registry,
    epsilon_closure,
    nfa_to_dfa,
    compute_reachability,
    dfa_accepts,
    dfa_walk,
    build_literal_nfa,
    nfa_concat,
    nfa_union,
    nfa_star,
    nfa_optional,
    nfa_plus,
)

from constrained_decoding.json_schema_to_fsm import (
    build_whitespace_nfa,
    build_string_value_nfa,
    build_number_nfa,
    build_integer_nfa,
    build_boolean_nfa,
    build_null_nfa,
    build_enum_nfa,
    build_const_nfa,
    build_object_nfa,
    build_array_nfa,
    build_value_nfa,
    build_tool_call_nfa,
)


# ======================================================================
# Helpers
# ======================================================================

def nfa_accepts(start: NFAState, text: str) -> bool:
    """Check if an NFA accepts *text* (for testing individual builders)."""
    dfa_start, _ = nfa_to_dfa(start)
    return dfa_accepts(dfa_start, text)


# ======================================================================
# char_fsm tests
# ======================================================================

def test_literal_nfa():
    _clear_registry()
    start, accept = build_literal_nfa("hello")
    assert nfa_accepts(start, "hello") is True
    assert nfa_accepts(start, "hell") is False
    assert nfa_accepts(start, "hello!") is False
    assert nfa_accepts(start, "hallo") is False
    print("  PASS test_literal_nfa")


def test_nfa_concat():
    _clear_registry()
    a_s, a_e = build_literal_nfa("abc")
    b_s, b_e = build_literal_nfa("def")
    c_s, c_e = nfa_concat((a_s, a_e), (b_s, b_e))
    assert nfa_accepts(c_s, "abcdef") is True
    assert nfa_accepts(c_s, "abc") is False
    assert nfa_accepts(c_s, "def") is False
    assert nfa_accepts(c_s, "abcde") is False
    print("  PASS test_nfa_concat")


def test_nfa_union():
    _clear_registry()
    a_s, a_e = build_literal_nfa("cat")
    b_s, b_e = build_literal_nfa("dog")
    u_s, u_e = nfa_union((a_s, a_e), (b_s, b_e))
    assert nfa_accepts(u_s, "cat") is True
    assert nfa_accepts(u_s, "dog") is True
    assert nfa_accepts(u_s, "bird") is False
    print("  PASS test_nfa_union")


def test_nfa_star():
    _clear_registry()
    a_s, a_e = build_literal_nfa("a")
    s_s, s_e = nfa_star((a_s, a_e))
    assert nfa_accepts(s_s, "") is True
    assert nfa_accepts(s_s, "a") is True
    assert nfa_accepts(s_s, "aa") is True
    assert nfa_accepts(s_s, "aaa") is True
    assert nfa_accepts(s_s, "b") is False
    print("  PASS test_nfa_star")


def test_nfa_optional():
    _clear_registry()
    a_s, a_e = build_literal_nfa("x")
    o_s, o_e = nfa_optional((a_s, a_e))
    assert nfa_accepts(o_s, "") is True
    assert nfa_accepts(o_s, "x") is True
    assert nfa_accepts(o_s, "xx") is False
    print("  PASS test_nfa_optional")


def test_catchall_in_dfa():
    """Test that CATCHALL transitions work correctly in the DFA."""
    _clear_registry()
    # Build NFA for: "a" + CATCHALL* + "z"
    s0 = _register(NFAState())
    s1 = _register(NFAState())
    s2 = _register(NFAState())

    s0.transitions.setdefault('a', []).append(s1)
    s1.transitions.setdefault(CATCHALL, []).append(s1)  # self-loop on any char
    s1.transitions.setdefault('z', []).append(s2)
    s2.is_accept = True

    dfa_start, all_dfa = nfa_to_dfa(s0)

    assert dfa_accepts(dfa_start, "az") is True
    assert dfa_accepts(dfa_start, "abz") is True
    assert dfa_accepts(dfa_start, "a你z") is True     # Unicode
    assert dfa_accepts(dfa_start, "a  z") is True
    assert dfa_accepts(dfa_start, "a") is False       # no closing z
    assert dfa_accepts(dfa_start, "bz") is False      # wrong start
    print(f"  PASS test_catchall_in_dfa (DFA has {len(all_dfa)} states)")


def test_dfa_reachability():
    """Test that can_reach_accept is correctly computed."""
    _clear_registry()
    # Build NFA: "a" -> dead_end | "b" -> accept
    s0 = _register(NFAState())
    s1 = _register(NFAState())
    s2 = _register(NFAState(is_accept=True))
    s3 = _register(NFAState())  # dead end

    s0.transitions.setdefault('a', []).append(s1)
    s1.transitions.setdefault('b', []).append(s2)
    s1.transitions.setdefault('c', []).append(s3)

    dfa_start, all_dfa = nfa_to_dfa(s0)

    # Find DFA state reached by "ac" — should NOT have can_reach_accept
    dead_state = dfa_walk(dfa_start, "ac")
    assert dead_state is not None
    assert dead_state.is_accept is False
    assert dead_state.can_reach_accept is False

    # Find DFA state reached by "ab" — should have can_reach_accept
    accept_state = dfa_walk(dfa_start, "ab")
    assert accept_state is not None
    assert accept_state.is_accept is True
    assert accept_state.can_reach_accept is True

    print("  PASS test_dfa_reachability")


# ======================================================================
# json_schema_to_fsm tests
# ======================================================================

def test_string_nfa():
    _clear_registry()
    start, _ = build_string_value_nfa()
    assert nfa_accepts(start, '""') is True
    assert nfa_accepts(start, '"hello"') is True
    assert nfa_accepts(start, '"hello world"') is True
    assert nfa_accepts(start, '"hello\\nworld"') is True
    assert nfa_accepts(start, '"hello\\u0041world"') is True
    assert nfa_accepts(start, '"hello"extra') is False
    assert nfa_accepts(start, 'hello') is False
    assert nfa_accepts(start, '"unclosed') is False
    print("  PASS test_string_nfa")


def test_number_nfa():
    _clear_registry()
    start, _ = build_number_nfa()
    valid = ["0", "42", "-17", "3.14", "-0.5", "1e10", "2.5e-3", "100"]
    invalid = ["01", "3.", ".5", "1e", "e10", "-", "--5", "1.2.3"]

    for s in valid:
        assert nfa_accepts(start, s) is True, f"Should accept: {s}"
    for s in invalid:
        assert nfa_accepts(start, s) is False, f"Should reject: {s}"
    print("  PASS test_number_nfa")


def test_boolean_null_nfa():
    _clear_registry()
    assert nfa_accepts(build_boolean_nfa()[0], "true") is True
    assert nfa_accepts(build_boolean_nfa()[0], "false") is True
    assert nfa_accepts(build_boolean_nfa()[0], "maybe") is False

    assert nfa_accepts(build_null_nfa()[0], "null") is True
    assert nfa_accepts(build_null_nfa()[0], "nil") is False
    print("  PASS test_boolean_null_nfa")


def test_enum_nfa():
    _clear_registry()
    start, _ = build_enum_nfa(["celsius", "fahrenheit"])
    assert nfa_accepts(start, '"celsius"') is True
    assert nfa_accepts(start, '"fahrenheit"') is True
    assert nfa_accepts(start, '"kelvin"') is False
    print("  PASS test_enum_nfa")


def test_object_nfa_simple():
    """Test object with one required property."""
    _clear_registry()
    start, _ = build_object_nfa(
        properties={
            "city": {"type": "string"},
        },
        required={"city"},
    )
    assert nfa_accepts(start, '{"city":"London"}') is True
    assert nfa_accepts(start, '{"city":""}') is True
    assert nfa_accepts(start, '{"city":"London","unit":"celsius"}') is False  # no such prop
    assert nfa_accepts(start, '{}') is False  # missing required
    print("  PASS test_object_nfa_simple")


def test_object_nfa_two_properties():
    """Test object with two required properties — both orderings."""
    _clear_registry()
    start, _ = build_object_nfa(
        properties={
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        required={"x", "y"},
    )
    assert nfa_accepts(start, '{"x":1,"y":2}') is True
    assert nfa_accepts(start, '{"y":2,"x":1}') is True
    assert nfa_accepts(start, '{"x":1}') is False  # missing y
    assert nfa_accepts(start, '{"y":2}') is False  # missing x
    print("  PASS test_object_nfa_two_properties")


def test_object_nfa_optional():
    """Test object with one required + one optional property."""
    _clear_registry()
    start, _ = build_object_nfa(
        properties={
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        required={"name"},
    )
    assert nfa_accepts(start, '{"name":"Alice"}') is True
    assert nfa_accepts(start, '{"name":"Alice","age":30}') is True
    assert nfa_accepts(start, '{"age":30,"name":"Alice"}') is True
    assert nfa_accepts(start, '{"age":30}') is False  # missing required name
    assert nfa_accepts(start, '{}') is False
    print("  PASS test_object_nfa_optional")


def test_object_nfa_empty():
    """Test empty object (no properties)."""
    _clear_registry()
    start, _ = build_object_nfa(properties={}, required=set())
    assert nfa_accepts(start, '{}') is True
    assert nfa_accepts(start, '{"extra":1}') is False
    print("  PASS test_object_nfa_empty")


def test_array_nfa():
    _clear_registry()
    start, _ = build_array_nfa({"type": "integer"})
    assert nfa_accepts(start, '[]') is True
    assert nfa_accepts(start, '[1]') is True
    assert nfa_accepts(start, '[1,2,3]') is True
    assert nfa_accepts(start, '[1,2,]') is False  # trailing comma
    assert nfa_accepts(start, '["a"]') is False  # wrong type
    print("  PASS test_array_nfa")


def test_tool_call_nfa():
    """Test the full tool call NFA."""
    _clear_registry()
    start = build_tool_call_nfa(
        tool_name="get_weather",
        args_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )

    dfa_start, all_dfa = nfa_to_dfa(start)
    print(f"  Tool call DFA has {len(all_dfa)} states")

    # Valid calls
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"city":"London"}}') is True
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"city":"北京"}}') is True
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"city":"London","unit":"celsius"}}') is True
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"unit":"fahrenheit","city":"Paris"}}') is True

    # Invalid calls
    assert dfa_accepts(dfa_start, '{"name":"other_tool","arguments":{}}') is False
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{}}') is False  # missing city
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"city":123}}') is False  # wrong type
    assert dfa_accepts(dfa_start, 'Hello') is False
    assert dfa_accepts(dfa_start, '{"name":"get_weather","arguments":{"city":"London"') is False  # unclosed

    print("  PASS test_tool_call_nfa")


def test_whitespace_handling():
    """Test that JSON whitespace is handled correctly."""
    _clear_registry()
    start, _ = build_object_nfa(
        properties={"a": {"type": "integer"}},
        required={"a"},
    )
    # With various whitespace
    assert nfa_accepts(start, '{"a":1}') is True
    assert nfa_accepts(start, '{ "a" : 1 }') is True
    assert nfa_accepts(start, '{\n"a":\t1\n}') is True
    print("  PASS test_whitespace_handling")


# ======================================================================
# Token index test
# ======================================================================

def test_token_index():
    """Test TokenIndex construction and queries."""
    from constrained_decoding.token_index import TokenIndex

    # Use a mock tokenizer
    class MockTokenizer:
        vocab_size = 10

        def decode(self, ids):
            mapping = {
                0: '<pad>', 1: '<eos>', 2: '{',
                3: '"', 4: 'name', 5: ':',
                6: 'get_weather', 7: '42',
                8: '}', 9: '',
            }
            results = []
            for i in ids:
                results.append(mapping.get(i, '?'))
            return ''.join(results)

        def get_vocab(self):
            return {}

        # Trigger the fast path in TokenIndex
        def convert_ids_to_tokens(self, ids):
            return [self.decode([i]) for i in ids]

    tokenizer = MockTokenizer()
    index = TokenIndex(tokenizer)

    assert len(index.id_to_entry) == 10
    assert index.id_to_entry[2].first_char == '{'
    assert index.id_to_entry[4].first_char == 'n'
    assert index.id_to_entry[9].first_char == ''  # empty string token

    candidates = index.get_candidates({'{'})
    assert len(candidates) == 1
    assert candidates[0].token_id == 2

    candidates = index.get_candidates({'{', '"', 'n'})
    assert len(candidates) == 3

    print("  PASS test_token_index")


# ======================================================================
# Logits processor test (mock DFA)
# ======================================================================

def test_logits_processor():
    """Test the LogitsProcessor with a synthetic DFA and tokenizer."""
    import torch
    from constrained_decoding.logits_processor import ToolConstraintLogitsProcessor
    from constrained_decoding.token_index import TokenIndex

    _clear_registry()

    # Build a simple DFA that accepts: {"a": <integer>}
    nfa_start, _ = build_object_nfa(
        properties={"a": {"type": "integer"}},
        required={"a"},
    )
    dfa_start, all_dfa = nfa_to_dfa(nfa_start)
    print(f"  Mock DFA has {len(all_dfa)} states")

    # Mock tokenizer with a tiny vocabulary
    class MockTokenizer:
        vocab_size = 20

        def decode(self, ids):
            mapping = {
                0: '<pad>', 1: '<eos>', 2: '{',
                3: '"', 4: 'a', 5: ':',
                6: '1', 7: '23', 8: '}',
                9: '999',
                10: '',   # empty token
                11: 'hello',  # invalid
                12: '{"b":1}',  # wrong key
                13: '42',
                14: ' ',    # whitespace
                15: '\n',  # newline
                16: '\t',  # tab
                17: 'x',   # invalid
                18: '9',
                19: '0',
            }
            results = []
            for i in ids:
                results.append(mapping.get(i, '?'))
            return ''.join(results)

        def get_vocab(self):
            return {}

        def convert_ids_to_tokens(self, ids):
            return [self.decode([i]) for i in ids]

    tokenizer = MockTokenizer()
    token_index = TokenIndex(tokenizer)

    # Create processor; simulate that the prompt has already been consumed.
    # We start with the DFA at its start state.
    # The start state expects '{' as the first character.
    processor = ToolConstraintLogitsProcessor(
        dfa_start=dfa_start,
        token_index=token_index,
        prompt_length=0,
        eos_token_id=1,
        pad_token_id=0,
    )

    # Simulate step 1: should only allow '{' (token 2)
    input_ids = torch.tensor([[0]])  # dummy input
    scores = torch.zeros((1, tokenizer.vocab_size))

    result = processor(input_ids, scores)
    # token 2 ('{') should be allowed (score = 0), rest should be -inf
    assert result[0, 2] == 0.0, f"Expected token 2 to be allowed, got {result[0, 2]}"
    assert result[0, 6] == -float('inf'), "Token 6 ('1') should be masked at step 1"

    # Advance state by '{' (token 2)
    input_ids = torch.tensor([[0, 2]])
    scores = torch.zeros((1, tokenizer.vocab_size))
    result = processor(input_ids, scores)

    # After '{', the DFA expects whitespace or '"'
    # Token 3 ('"') should be allowed
    assert result[0, 3] == 0.0, f"Expected token 3 to be allowed after '{{', got {result[0, 3]}"
    # Token 14 (' ') should also be allowed (whitespace)
    assert result[0, 14] == 0.0, f"Expected token 14 (space) to be allowed after '{{'"

    print("  PASS test_logits_processor")


# ======================================================================
# Integration test: full generation with real tokenizer (if available)
# ======================================================================

def test_full_integration_with_hf_tokenizer():
    """Test the full pipeline with a real HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  SKIP test_full_integration_with_hf_tokenizer (transformers not installed)")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            print("  SKIP test_full_integration_with_hf_tokenizer (no model available)")
            return

    from constrained_decoding.token_index import TokenIndex
    from constrained_decoding.logits_processor import ToolConstraintLogitsProcessor

    _clear_registry()

    # Build DFA for a simple tool call
    nfa_start = build_tool_call_nfa(
        tool_name="get_weather",
        args_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )
    dfa_start, all_dfa = nfa_to_dfa(nfa_start)
    print(f"  Tool call DFA has {len(all_dfa)} states")

    token_index = TokenIndex(tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Verify: for a prompt that already contains the tool call start,
    # the processor should mask appropriately.
    import torch

    prompt = '{"name":"get_weather","arguments":{"city":"'
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_length = prompt_ids.shape[1]

    processor = ToolConstraintLogitsProcessor(
        dfa_start=dfa_start,
        token_index=token_index,
        prompt_length=0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Simulate advancing through the prompt tokens
    input_ids = prompt_ids.clone()
    scores = torch.zeros((1, tokenizer.vocab_size))
    result = processor(input_ids, scores)

    # At this point, the DFA should be inside the string value for "city"
    # Only tokens that continue a valid JSON string should be allowed,
    # and the closing '"' should lead to an accept state eventually.
    allowed_count = (result[0] > -float('inf') / 2).sum().item()
    print(f"  After prompt, allowed tokens: {allowed_count} / {tokenizer.vocab_size}")

    assert allowed_count > 0, "Should allow at least some tokens"
    assert allowed_count < tokenizer.vocab_size, "Should mask some tokens"

    print("  PASS test_full_integration_with_hf_tokenizer")


# ======================================================================
# Run all tests
# ======================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing char_fsm.py ...")
    print("=" * 60)
    test_literal_nfa()
    test_nfa_concat()
    test_nfa_union()
    test_nfa_star()
    test_nfa_optional()
    test_catchall_in_dfa()
    test_dfa_reachability()

    print()
    print("=" * 60)
    print("Testing json_schema_to_fsm.py ...")
    print("=" * 60)
    test_string_nfa()
    test_number_nfa()
    test_boolean_null_nfa()
    test_enum_nfa()
    test_object_nfa_simple()
    test_object_nfa_two_properties()
    test_object_nfa_optional()
    test_object_nfa_empty()
    test_array_nfa()
    test_tool_call_nfa()
    test_whitespace_handling()

    print()
    print("=" * 60)
    print("Testing token_index.py ...")
    print("=" * 60)
    test_token_index()

    print()
    print("=" * 60)
    print("Testing logits_processor.py ...")
    print("=" * 60)
    test_logits_processor()

    print()
    print("=" * 60)
    print("Integration test ...")
    print("=" * 60)
    test_full_integration_with_hf_tokenizer()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
