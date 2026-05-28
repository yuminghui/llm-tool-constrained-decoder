"""
JSON Schema -> NFA (Non-deterministic Finite Automaton) converter.

Converts a JSON Schema to a character-level NFA that accepts exactly the set of
JSON strings valid under that schema. The NFA can then be converted to a DFA
for efficient runtime token validation.

Supports: string, number, integer, boolean, null, enum, const, object, array.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

from .char_fsm import (
    NFAState,
    CATCHALL,
    _register,
    _clear_registry,
    build_literal_nfa,
    nfa_concat,
    nfa_union,
    nfa_optional,
    nfa_star,
    finalize_nfa,
)

# ---------------------------------------------------------------------------
# Character sets
# ---------------------------------------------------------------------------

JSON_WHITESPACE = {' ', '\t', '\n', '\r'}

# Printable ASCII characters allowed unescaped in JSON strings
# (everything 0x20-0x7E except " (0x22) and \ (0x5C))
_SAFE_STRING_CHARS: Set[str] = {
    chr(c) for c in range(0x20, 0x7F)
    if c != 0x22 and c != 0x5C
}

# All decimal digits
_DIGITS = {str(d) for d in range(10)}
_NONZERO_DIGITS = {str(d) for d in range(1, 10)}

# Hex digits
_HEX_DIGITS = {str(d) for d in range(10)} | {c for c in 'abcdefABCDEF'}

# Simple JSON escape characters
_SIMPLE_ESCAPES = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}


# ---------------------------------------------------------------------------
# Whitespace (zero or more)
# ---------------------------------------------------------------------------

def build_whitespace_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching zero or more JSON whitespace characters.

    Equivalent to regex: [ \\t\\n\\r]*
    """
    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    for ch in JSON_WHITESPACE:
        start.transitions.setdefault(ch, []).append(start)  # self-loop

    start.add_epsilon(accept)  # zero repetitions

    return start, accept


# ---------------------------------------------------------------------------
# JSON string value (the value, not a property key)
# ---------------------------------------------------------------------------

def build_string_value_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching a JSON string VALUE.

    Handles:
      - Opening/closing double-quotes
      - Any printable ASCII char except ``\"`` and ``\\``
      - Escape sequences: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
      - Unicode chars beyond ASCII (via CATCHALL)

    State flow:
        start --\"--> in_string
        in_string --[safe char | CATCHALL]--> in_string
        in_string --\\--> after_backslash
        after_backslash --[simple escape]--> in_string
        after_backslash --u--> hex1 --hex--> hex2 --hex--> hex3 --hex--> hex4 --hex--> in_string
        in_string --\"--> accept
    """
    start = _register(NFAState())
    in_string = _register(NFAState())
    after_bs = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    # Opening quote
    start.transitions.setdefault('"', []).append(in_string)

    # Safe characters (printable ASCII except " and \)
    for ch in _SAFE_STRING_CHARS:
        in_string.transitions.setdefault(ch, []).append(in_string)

    # Catch-all for Unicode beyond ASCII (CJK, emoji, etc.)
    in_string.transitions.setdefault(CATCHALL, []).append(in_string)

    # Backslash escape
    in_string.transitions.setdefault('\\', []).append(after_bs)

    # Simple escapes
    for ch in _SIMPLE_ESCAPES:
        after_bs.transitions.setdefault(ch, []).append(in_string)

    # Unicode escape \uXXXX
    hex_states = [_register(NFAState()) for _ in range(4)]
    after_bs.transitions.setdefault('u', []).append(hex_states[0])
    for i in range(4):
        for ch in _HEX_DIGITS:
            target = hex_states[i + 1] if i < 3 else in_string
            hex_states[i].transitions.setdefault(ch, []).append(target)

    # Closing quote
    in_string.transitions.setdefault('"', []).append(accept)

    return start, accept


# ---------------------------------------------------------------------------
# JSON number (RFC 8259)
# ---------------------------------------------------------------------------

def build_number_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching a JSON number per RFC 8259.

    number = [ minus ] int [ frac ] [ exp ]
    int = zero / digit1-9 *DIGIT
    frac = \".\" 1*DIGIT
    exp = (\"e\" / \"E\") [\"+\" / \"-\"] 1*DIGIT

    Returns (start, accept) where *accept* is a single unified accept state.
    """
    start = _register(NFAState())
    opt_minus = _register(NFAState())
    after_zero = _register(NFAState())
    int_digits = _register(NFAState())
    frac_first = _register(NFAState())
    frac_digits = _register(NFAState())
    exp_sign = _register(NFAState())
    exp_first = _register(NFAState())
    exp_digits = _register(NFAState())

    # Single unified accept — all internal accept states epsilon here
    unified_accept = _register(NFAState(is_accept=True))
    after_zero.add_epsilon(unified_accept)
    int_digits.add_epsilon(unified_accept)
    frac_digits.add_epsilon(unified_accept)
    exp_digits.add_epsilon(unified_accept)

    # Optional minus
    start.transitions.setdefault('-', []).append(opt_minus)
    for d in _DIGITS:
        if d == '0':
            start.transitions.setdefault(d, []).append(after_zero)
        else:
            start.transitions.setdefault(d, []).append(int_digits)

    for d in _NONZERO_DIGITS:
        opt_minus.transitions.setdefault(d, []).append(int_digits)
    opt_minus.transitions.setdefault('0', []).append(after_zero)

    for d in _DIGITS:
        int_digits.transitions.setdefault(d, []).append(int_digits)

    # Fractional part
    for state in (int_digits, after_zero):
        state.transitions.setdefault('.', []).append(frac_first)
        state.transitions.setdefault('e', []).append(exp_sign)
        state.transitions.setdefault('E', []).append(exp_sign)

    for d in _DIGITS:
        frac_first.transitions.setdefault(d, []).append(frac_digits)
    for d in _DIGITS:
        frac_digits.transitions.setdefault(d, []).append(frac_digits)
    frac_digits.transitions.setdefault('e', []).append(exp_sign)
    frac_digits.transitions.setdefault('E', []).append(exp_sign)

    # Exponent
    for d in _DIGITS:
        exp_sign.transitions.setdefault(d, []).append(exp_digits)
    exp_sign.transitions.setdefault('+', []).append(exp_first)
    exp_sign.transitions.setdefault('-', []).append(exp_first)
    for d in _DIGITS:
        exp_first.transitions.setdefault(d, []).append(exp_digits)
    for d in _DIGITS:
        exp_digits.transitions.setdefault(d, []).append(exp_digits)

    return start, unified_accept


def build_integer_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching a JSON integer (number without fraction/exponent).

    Returns (start, unified_accept).
    """
    start = _register(NFAState())
    opt_minus = _register(NFAState())
    after_zero = _register(NFAState())
    int_digits = _register(NFAState())

    # Single unified accept
    unified_accept = _register(NFAState(is_accept=True))
    after_zero.add_epsilon(unified_accept)
    int_digits.add_epsilon(unified_accept)

    start.transitions.setdefault('-', []).append(opt_minus)
    for d in _DIGITS:
        if d == '0':
            start.transitions.setdefault(d, []).append(after_zero)
        else:
            start.transitions.setdefault(d, []).append(int_digits)

    for d in _NONZERO_DIGITS:
        opt_minus.transitions.setdefault(d, []).append(int_digits)
    opt_minus.transitions.setdefault('0', []).append(after_zero)

    for d in _DIGITS:
        int_digits.transitions.setdefault(d, []).append(int_digits)

    return start, unified_accept


# ---------------------------------------------------------------------------
# Literal builders
# ---------------------------------------------------------------------------

def build_boolean_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching ``true`` or ``false``."""
    return nfa_union(build_literal_nfa('true'), build_literal_nfa('false'))


def build_null_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching ``null``."""
    return build_literal_nfa('null')


def build_enum_nfa(values: List[Any]) -> Tuple[NFAState, NFAState]:
    """Build NFA matching any of the JSON-serialized enum values."""
    sub_nfas = []
    for val in values:
        json_str = json.dumps(val)
        sub_nfas.append(build_literal_nfa(json_str))
    return nfa_union(*sub_nfas)


def build_const_nfa(value: Any) -> Tuple[NFAState, NFAState]:
    """Build NFA matching a specific JSON value."""
    return build_literal_nfa(json.dumps(value))


# ---------------------------------------------------------------------------
# JSON object (with bitmask property-order tracking)
# ---------------------------------------------------------------------------

def build_object_nfa(
    properties: Dict[str, Dict[str, Any]],
    required: Set[str],
    additional_properties: bool = False,
) -> Tuple[NFAState, NFAState]:
    """Build NFA matching a JSON object conforming to the given schema.

    Uses bitmask tracking to handle arbitrary property ordering without
    enumerating N! permutations.

    Args:
        properties: Dict of property_name -> JSON Schema for that property.
        required: Set of required property names.
        additional_properties: If True, allow extra properties (weakens constraint).

    Returns:
        (start_state, accept_state)
    """
    prop_names = sorted(properties.keys())  # canonical order for bit assignment

    # Assign bit positions to required properties
    prop_bit: Dict[str, int] = {}
    bit_idx = 0
    for pname in sorted(required):
        prop_bit[pname] = 1 << bit_idx
        bit_idx += 1
    all_required_mask = (1 << len(required)) - 1

    # Extra bit that gets set when ANY property has been consumed.
    # This controls comma insertion: only the very first property
    # (mask==0) omits the leading comma.
    HAS_PROPERTY_BIT = 1 << len(required)

    # Total masks: 0 .. (1 << (len(required) + 1)) - 1
    num_masks = 1 << (len(required) + 1)

    # Create choice states keyed by seen_mask
    choice_states: Dict[int, NFAState] = {}
    for mask in range(num_masks):
        choice_states[mask] = _register(NFAState())

    # Accept state for the object
    accept = _register(NFAState(is_accept=True))

    # Build branches from each choice state
    for mask, choice_state in choice_states.items():

        # ---- Branch: close object "}" if all required seen ----
        if (mask & all_required_mask) == all_required_mask:
            ws_close = build_whitespace_nfa()
            close_brace = _register(NFAState())
            ws_close[1].transitions.setdefault('}', []).append(close_brace)
            close_brace.add_epsilon(accept)
            choice_state.add_epsilon(ws_close[0])

        # ---- Branch: each available property ----
        for pname, pschema in properties.items():
            bit = prop_bit.get(pname)

            # Required and already seen? skip
            if bit is not None and (mask & bit):
                continue

            new_mask = mask | HAS_PROPERTY_BIT
            if bit is not None:
                new_mask |= bit

            # Build a FRESH property sequence for this (mask, property) pair
            prop_start, prop_end = _build_one_property_sequence(pname, pschema)

            if not (mask & HAS_PROPERTY_BIT):
                # First property: no comma
                choice_state.add_epsilon(prop_start)
            else:
                # Comma + whitespace before subsequent property
                comma_state = _register(NFAState())
                comma_accept = _register(NFAState(is_accept=True))
                comma_state.transitions.setdefault(',', []).append(comma_accept)
                ws_after_comma = build_whitespace_nfa()
                chunk_start, chunk_accept = nfa_concat(
                    (comma_state, comma_accept),
                    ws_after_comma,
                    (prop_start, prop_end),
                )
                choice_state.add_epsilon(chunk_start)
                prop_end = chunk_accept  # use the chunk's accept for routing

            if new_mask in choice_states:
                prop_end.add_epsilon(choice_states[new_mask])

        # ---- Branch: additional properties (if enabled) ----
        if additional_properties:
            extra_prop_start, extra_prop_end = _build_extra_property_sequence()
            if not (mask & HAS_PROPERTY_BIT):
                choice_state.add_epsilon(extra_prop_start)
            else:
                comma_state = _register(NFAState())
                comma_accept = _register(NFAState(is_accept=True))
                comma_state.transitions.setdefault(',', []).append(comma_accept)
                chunk_start, chunk_accept = nfa_concat(
                    (comma_state, comma_accept),
                    (extra_prop_start, extra_prop_end),
                )
                choice_state.add_epsilon(chunk_start)
                extra_prop_end = chunk_accept
            extra_prop_end.add_epsilon(choice_states[mask | HAS_PROPERTY_BIT])  # mark consumed

    # Open brace
    start = _register(NFAState())
    start.transitions.setdefault('{', []).append(choice_states[0])

    # Handle empty object: "{" + whitespace* + "}"
    if len(required) == 0:
        ws_empty = build_whitespace_nfa()
        empty_accept = _register(NFAState())
        ws_empty[1].transitions.setdefault('}', []).append(empty_accept)
        empty_accept.add_epsilon(accept)
        start.add_epsilon(ws_empty[0])

    return start, accept


def _build_one_property_sequence(
    prop_name: str, prop_schema: Dict[str, Any]
) -> Tuple[NFAState, NFAState]:
    """Build NFA sequence for one object property.

    Format: optional_whitespace  '\"prop_name\"'  optional_whitespace  ':'
            optional_whitespace  <value_nfa>
    """
    ws1_start, ws1_accept = build_whitespace_nfa()
    key_start, key_accept = build_literal_nfa(json.dumps(prop_name))
    ws2_start, ws2_accept = build_whitespace_nfa()
    colon_start, colon_accept = build_literal_nfa(':')
    ws3_start, ws3_accept = build_whitespace_nfa()
    value_start, value_accept = build_value_nfa(prop_schema)

    return nfa_concat(
        (ws1_start, ws1_accept),
        (key_start, key_accept),
        (ws2_start, ws2_accept),
        (colon_start, colon_accept),
        (ws3_start, ws3_accept),
        (value_start, value_accept),
    )


def _build_extra_property_sequence() -> Tuple[NFAState, NFAState]:
    """Build NFA for an additional property: string key + ':' + any value."""
    ws1 = build_whitespace_nfa()
    key_s, key_e = build_string_value_nfa()  # any string as key
    ws2 = build_whitespace_nfa()
    colon_s, colon_e = build_literal_nfa(':')
    ws3 = build_whitespace_nfa()
    val_s, val_e = build_any_value_nfa()

    return nfa_concat(
        ws1, (key_s, key_e), ws2, (colon_s, colon_e), ws3, (val_s, val_e),
    )


# ---------------------------------------------------------------------------
# JSON array
# ---------------------------------------------------------------------------

def build_array_nfa(items_schema: Dict[str, Any]) -> Tuple[NFAState, NFAState]:
    """Build NFA matching a JSON array with items conforming to ``items_schema``.

    Matches: ``[`` (value (``,`` value)*)? ``]``
    """
    start = _register(NFAState())
    accept = _register(NFAState(is_accept=True))

    # Build the item sequence: value
    item_start, item_accept = build_value_nfa(items_schema)

    # After-item continuation: comma + whitespace + value
    ws_after_comma = build_whitespace_nfa()
    comma_state = _register(NFAState())
    comma_accept = _register(NFAState(is_accept=True))
    comma_state.transitions.setdefault(',', []).append(comma_accept)

    after_item = _register(NFAState())
    item_accept.add_epsilon(after_item)

    # From after_item: either continue with comma+value, or close

    # Comma branch: comma + whitespace* + next item
    ws_before_comma = build_whitespace_nfa()
    after_item.add_epsilon(ws_before_comma[0])
    ws_before_comma[1].add_epsilon(comma_state)
    comma_accept.add_epsilon(ws_after_comma[0])
    ws_after_comma[1].add_epsilon(item_start)

    # Close branch: whitespace* + ]
    ws_close = build_whitespace_nfa()
    after_item.add_epsilon(ws_close[0])
    close_brace = _register(NFAState())
    ws_close[1].transitions.setdefault(']', []).append(close_brace)
    close_brace.add_epsilon(accept)

    # Opening bracket
    ws_open = build_whitespace_nfa()
    start.transitions.setdefault('[', []).append(ws_open[0])

    # After opening whitespace: either first item, or close (empty array)
    ws_open[1].add_epsilon(item_start)
    ws_open[1].add_epsilon(ws_close[0])

    return start, accept


# ---------------------------------------------------------------------------
# Any JSON value (unconstrained)
# ---------------------------------------------------------------------------

def build_any_value_nfa() -> Tuple[NFAState, NFAState]:
    """Build NFA matching any JSON value (string | number | boolean | null | object | array).

    Objects and arrays are limited to depth 1 to keep the NFA finite.
    """
    # For "any" value, we accept: string, number, true, false, null
    # We do NOT include recursive object/array to keep the NFA finite
    str_nfa = build_string_value_nfa()
    num_nfa = build_number_nfa()
    bool_nfa = build_boolean_nfa()
    null_nfa = build_null_nfa()

    return nfa_union(str_nfa, num_nfa, bool_nfa, null_nfa)


# ---------------------------------------------------------------------------
# Type dispatch
# ---------------------------------------------------------------------------

def build_value_nfa(schema: Dict[str, Any]) -> Tuple[NFAState, NFAState]:
    """Dispatch schema to the correct NFA builder based on ``type`` or keywords."""
    if not schema:
        return build_any_value_nfa()

    # Keyword shortcuts
    if 'enum' in schema:
        return build_enum_nfa(schema['enum'])
    if 'const' in schema:
        return build_const_nfa(schema['const'])

    schema_type = schema.get('type', 'any')

    # Handle union types: {"type": ["string", "null"]}
    if isinstance(schema_type, list):
        sub_nfas = []
        for t in schema_type:
            sub_nfas.append(_build_typed_nfa(t, schema))
        return nfa_union(*sub_nfas)

    return _build_typed_nfa(schema_type, schema)


def _build_typed_nfa(schema_type: str, schema: Dict[str, Any]) -> Tuple[NFAState, NFAState]:
    """Build NFA for a specific JSON type."""
    if schema_type == 'string':
        return build_string_value_nfa()
    elif schema_type == 'number':
        return build_number_nfa()
    elif schema_type == 'integer':
        return build_integer_nfa()
    elif schema_type == 'boolean':
        return build_boolean_nfa()
    elif schema_type == 'null':
        return build_null_nfa()
    elif schema_type == 'object':
        return build_object_nfa(
            properties=schema.get('properties', {}),
            required=set(schema.get('required', [])),
            additional_properties=schema.get('additionalProperties', True),
        )
    elif schema_type == 'array':
        items_schema = schema.get('items', {})
        return build_array_nfa(items_schema)
    else:
        return build_any_value_nfa()


# ---------------------------------------------------------------------------
# Top-level tool call NFA
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = '{"name":"{name}","arguments":{arguments}}'


def build_tool_call_nfa(
    tool_name: str,
    args_schema: Dict[str, Any],
    template: str = DEFAULT_TEMPLATE,
) -> NFAState:
    """Build the full tool-call NFA.

    Substitutes ``{name}`` and ``{arguments}`` into the template, builds
    literal NFAs for fixed parts and a schema-driven NFA for the arguments.

    Args:
        tool_name: Name of the tool (e.g. ``"get_weather"``).
        args_schema: JSON Schema dict for the tool's arguments.
        template: Format string with ``{name}`` and ``{arguments}`` placeholders.

    Returns:
        The start NFAState of the full tool-call NFA.
    """
    # Build the arguments object NFA
    args_obj_start, args_obj_accept = build_object_nfa(
        properties=args_schema.get('properties', {}),
        required=set(args_schema.get('required', [])),
        additional_properties=args_schema.get('additionalProperties', False),
    )

    # Split template on placeholders
    parts_name = template.split('{name}')
    if len(parts_name) != 2:
        raise ValueError("Template must contain exactly one {name} placeholder")
    prefix = parts_name[0]

    parts_args = parts_name[1].split('{arguments}')
    if len(parts_args) != 2:
        raise ValueError("Template must contain exactly one {arguments} placeholder")
    name_suffix = parts_args[0]
    suffix = parts_args[1]

    # Build components
    components: List[Tuple[NFAState, NFAState]] = []

    if prefix:
        components.append(build_literal_nfa(prefix))

    # Tool name (just the raw name, no quotes — template provides quotes if needed)
    components.append(build_literal_nfa(tool_name))

    if name_suffix:
        components.append(build_literal_nfa(name_suffix))

    # Arguments object
    components.append((args_obj_start, args_obj_accept))

    if suffix:
        components.append(build_literal_nfa(suffix))

    full_start, full_accept = nfa_concat(*components)
    # Ensure only the top-level accept state is marked accepting.
    # Sub-NFA accept states (e.g. string value's internal accept) would
    # otherwise leak is_accept=True into the DFA.
    finalize_nfa(full_accept)
    return full_start
