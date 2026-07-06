# TC-Decoder (Tool Constrained Decoder)

A plug-and-play, training-free constrained decoder. During autoregressive generation, it constrains the LLM's output via a finite-state machine (FSM), ensuring that the generated text is **structurally guaranteed** to be a valid invocation of a specified tool.

## Core Idea

```
JSON Schema ──▶ NFA ──▶ DFA ──▶ LogitsProcessor ──▶ model.generate()
 (param defs)  (char-lvl) (determinized) (per-token mask)   (constrained output)
```

1. **Compile** the tool's JSON Schema into a character-level NFA (Nondeterministic Finite Automaton)
2. **Determinize** the NFA into a DFA via subset construction
3. At each decoding step, the LogitsProcessor iterates over all tokens in the vocabulary and sets the logit of any token the DFA cannot accept to `-inf`
4. The model can only sample along "legal paths," so the output is **guaranteed** to be a valid tool call

## Module Architecture

```
constrained_decoding/
├── char_fsm.py              # FSM engine: NFA/DFA data structures, NFA→DFA conversion, combinators
├── json_schema_to_fsm.py    # JSON Schema → NFA compiler
├── token_index.py           # Vocabulary first-character index (accelerates candidate filtering)
├── logits_processor.py      # HuggingFace LogitsProcessor (runtime constraint)
├── decoder.py               # High-level API: ToolConstrainedDecoder
└── __init__.py              # Public exports
```

### Data Flow

```
                    ┌──────────────────────┐
                    │  json_schema_to_fsm   │
                    │  build_tool_call_nfa()│
                    └──────────┬───────────┘
                               │ NFAState (start)
                               ▼
                    ┌──────────────────────┐
                    │      char_fsm         │
                    │    nfa_to_dfa()       │
                    └──────────┬───────────┘
                               │ DFAState (start)
                               ▼
            ┌──────────────────────────────────────┐
            │        ToolConstrainedDecoder         │
            │  ┌─────────────────────────────────┐ │
            │  │ prompt ──▶ tokenizer.encode()    │ │
            │  │                                  │ │
            │  │ DFA + TokenIndex ──▶             │ │
            │  │ ToolConstraintLogitsProcessor    │ │
            │  │                                  │ │
            │  │ model.generate(                  │ │
            │  │   logits_processor=[processor])  │ │
            │  └─────────────────────────────────┘ │
            └──────────────────────────────────────┘
                               │
                               ▼
                        DecoderResult
                        ├── text: str
                        ├── tool_call: dict
                        ├── token_ids: list
                        └── finish_reason: str
```

## Module Details

### 1. `char_fsm.py` — FSM Engine

The core character-level finite-state machine, providing two abstractions — NFA and DFA:

| Component | Description |
|------|------|
| `NFAState` | NFA state. Supports character transitions + ε transitions. Each state has a unique ID, globally registered |
| `DFAState` | DFA state. Transition table is `dict[char, DFAState]`. Supports `CATCHALL` (matches any character not explicitly listed) |
| `epsilon_closure()` | Computes the ε-closure of a set of NFA states |
| `nfa_to_dfa()` | Subset construction: NFA → DFA |
| `compute_reachability()` | Reverse BFS precomputes whether each DFA state can reach an accept state (used for token validity checking) |
| `CATCHALL` | Sentinel character `\x00__CATCHALL__`, meaning "any character not explicitly matched" |

**Combinators (NFA Algebra)**:

| Function | Regex Equivalent | Purpose |
|------|:--:|------|
| `build_literal_nfa(s)` | `"s"` | Match a fixed string |
| `nfa_concat(a, b)` | `ab` | Concatenation |
| `nfa_union(a, b)` | `a\|b` | Union (choose one) |
| `nfa_star(a)` | `a*` | Zero or more repetitions |
| `nfa_optional(a)` | `a?` | Zero or one |
| `nfa_plus(a)` | `a+` | One or more repetitions |

### 2. `json_schema_to_fsm.py` — Schema → NFA Compiler

Compiles JSON Schema into a character-level NFA. Supported Schema types:

| Type | Implementation | Description |
|------|------|------|
| `string` | `build_string_value_nfa()` | `"..."` including escapes, Unicode, CATCHALL |
| `number` | `build_number_nfa()` | Full RFC 8259 number grammar |
| `integer` | `build_integer_nfa()` | Integer (no decimal/exponent) |
| `boolean` | `build_boolean_nfa()` | `true` / `false` |
| `null` | `build_null_nfa()` | `null` |
| `enum` | `build_enum_nfa()` | Union of enum values |
| `const` | `build_const_nfa()` | Constant match |
| `object` | `build_object_nfa()` | **Bitmask** tracking of property appearance. Supports required fields, arbitrary order |
| `array` | `build_array_nfa()` | `[item, ...]` |

**Top-level entry point `build_tool_call_nfa()`**: Substitutes `tool_name` and `args_schema` into the template (default `{"name":"{name}","arguments":{arguments}}`), concatenating the prefix, tool name, argument NFA, and suffix.

**Bitmask Object Encoding**: For an object with N required properties, uses `2^(N+1)` states (instead of N! permutations). Each state corresponds to "the set of properties seen so far." An additional `HAS_PROPERTY_BIT` controls comma insertion (no comma before the first property).

### 3. `token_index.py` — Vocabulary Index

| Class | Description |
|------|------|
| `TokenEntry` | Metadata for a single token: `token_id`, `token_str`, `first_char` |
| `TokenIndex` | Builds an inverted index by first character `{first_char: [TokenEntry, ...]}`, plus an `id → entry` mapping |

In the LogitsProcessor, tokens are first rapidly rejected by first character, and only the remaining candidates undergo a full DFA walk.

### 4. `logits_processor.py` — Runtime Constraint

`ToolConstraintLogitsProcessor` implements the HuggingFace `LogitsProcessor` interface, inserted into the `model.generate()` pipeline:

**Per-step decoding flow**:

```
1. Advance the DFA state (consume the token generated in the previous step)
2. If the DFA has reached an accept state → force output EOS
3. Otherwise, iterate over every token in the vocabulary:
   a. Unknown/empty token → mask (-inf)
   b. First-character fast rejection → mask
   c. Full DFA walk → mask if no accept state is reachable
4. EOS is only allowed after accept; PAD is permanently masked
```

**Cache**: `(dfa_state_id, token_str) → (is_valid, next_state_id)`. When the same DFA state encounters the same token, the result is looked up directly, avoiding repeated character-level walks.

### 5. `decoder.py` — High-Level API

`ToolConstrainedDecoder` is the single user-facing entry point:

```python
decoder = ToolConstrainedDecoder(model, tokenizer)
result = decoder.generate(
    prompt="<formatted chat prompt>",
    tool_name="get_weather",
    args_schema={"type": "object", "properties": {...}},
    max_new_tokens=256,
    temperature=0.7,
)
# result.tool_call → {"name": "get_weather", "arguments": {...}}
# result.text       → '<tool_call>{"name":"get_weather",...}</tool_call>'
# result.token_ids  → [1234, 5678, ...]
```

**`generate()` internal steps**:

1. `build_tool_call_nfa()` — Schema → NFA
2. `nfa_to_dfa()` — NFA → DFA
3. `tokenizer.encode(prompt)` — obtain prompt token IDs
4. Create `ToolConstraintLogitsProcessor(dfa, token_index, ...)`
5. Call `model.generate(logits_processor=[processor], ...)`
6. Parse the generated text into a `DecoderResult`

**Template-aware parsing**: `_parse_with_template()` first attempts direct JSON parsing; on failure, strips non-JSON wrapping portions based on the template and retries.

## Design Highlights

### Why character-level FSM?

A token-level FSM would need to enforce constraints across tokens, but token boundaries are determined by the tokenizer and do not align with JSON's character structure. A character-level DFA validates whether an entire token string can be accepted by the DFA, naturally handling cases where token boundaries cross JSON syntactic structures.

### CATCHALL Mechanism

JSON strings allow arbitrary Unicode characters (via `\uXXXX` or direct encoding). The CATCHALL transition avoids enumerating all possible Unicode characters in the NFA, dramatically reducing the DFA state count.

### Bitmask Object Encoding

Instead of enumerating property permutations (N! variants), a bitmask tracks "the set of properties seen so far." Each `(mask, property)` combination generates one transition path. For an object with N required properties, the state count is `2^(N+1)` rather than N!.

### Caching Strategy

Cache granularity is `(dfa_state_id, token_str)`. Most tokens are evaluated repeatedly across multiple generation steps (same DFA state + same token). Cache hit rates are typically >90%, significantly reducing character-level walk overhead.

### Training-Free

The entire process requires no weight modification, no additional training, and no prompt engineering. Constraints are implemented entirely at the decoding stage via logit masking, with zero intrusion into the model itself.

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from constrained_decoding import ToolConstrainedDecoder

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

decoder = ToolConstrainedDecoder(model, tokenizer)

# Constrain generation to a plan tool call
result = decoder.generate(
    prompt="<|im_start|>system\nYou are a task planner.<|im_end|>\n<|im_start|>assistant\n",
    tool_name="plan",
    args_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer"},
                        "step_name": {"type": "string"},
                        "description": {"type": "string"},
                        "expected_tools": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["step_number", "step_name", "description", "expected_tools"]
                }
            }
        },
        "required": ["title", "steps"]
    },
    max_new_tokens=256,
    temperature=0.7,
)

print(result.tool_call["arguments"]["title"])
for step in result.tool_call["arguments"]["steps"]:
    print(f"  {step['step_number']}. {step['step_name']}")
```

## Guarantees

TC-Decoder provides the following **constructive guarantees** (by construction, not probabilistic):

- The output **is always** valid JSON (no unclosed quotes, illegal escapes, etc.)
- The output **always contains** the specified `tool_name`
- The output **always contains** an `arguments` object
- All `required` properties in `arguments` **are always present**
- Property values **always conform** to their declared types (string/number/boolean/array/object)
- No properties outside the Schema are emitted (when `additionalProperties=false`)
