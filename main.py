"""
Tool-Constrained Decoding Demo
===============================
Demonstrates constrained decoding with google/gemma-4-E2B-it:
  Step 1 forces a plan tool call (tool name locked, arguments generated freely by the LLM)
  Subsequent steps may call other tools freely.

Tool set:
  - plan:           Create an execution plan (mandatory first step)
  - get_weather:    Query weather information
  - search_web:     Search the web
  - calculator:     Perform mathematical calculations
"""

import json
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constrained_decoding import ToolConstrainedDecoder, DecoderResult

# ---------------------------------------------------------------------------
# Gemma 4 tool-call format template
# Gemma wraps JSON with <tool_call|> ... <|tool_call>
# ---------------------------------------------------------------------------
GEMMA_TOOL_CALL_TEMPLATE = '<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>'


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": "Create a plan before taking any action. This is the mandatory first-step tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of execution steps",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Rationale for the plan",
                    },
                },
                "required": ["steps", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Query the weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, e.g. '2 + 3 * 4'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# Tool schema mapping (used by the constrained decoder)
TOOL_SCHEMAS = {
    tool["function"]["name"]: tool["function"]["parameters"]
    for tool in TOOLS
}


# ---------------------------------------------------------------------------
# Scenario prompt
# ---------------------------------------------------------------------------
USER_QUERY = (
    "I need to research the latest AI industry trends for 2026, "
    "check the weather in San Francisco for an outdoor event, "
    "and calculate the total addressable market size ($50B growing at 15% for 3 years). "
    # "Please plan your approach first before doing anything."
)


def load_model_and_tokenizer(model_id: str = "google/gemma-4-E2B-it"):
    """Load the model and tokenizer."""
    print(f"[1/5] Loading model: {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Gemma 4 requires PyTorch compute capability 12.0+ for RTX 50 series
    # Fall back to CPU if CUDA is incompatible
    use_cuda = False
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            supported = torch.cuda.get_arch_list()
            arch_str = f"sm_{major}{minor}"
            use_cuda = any(arch_str in s for s in supported)
        except Exception:
            use_cuda = False

    device = "cuda" if use_cuda else "cpu"
    print(f"       device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model = model.to(device)

    print(f"       model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer, device


def build_prompt(tokenizer, tools: list, user_query: str) -> str:
    """Build a prompt with tool definitions using the Gemma chat template."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to external tools. "
                # "Always plan first using the 'plan' tool before calling any other tool. "
                # "After planning, execute tools one at a time."
                "Execute tools one at a time."
                "Please respond in Chinese."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def parse_tool_call(text: str) -> dict | None:
    """Parse JSON from a Gemma tool-call output."""
    # Strip Gemma special-token wrappers
    for prefix in ["<tool_call|>", "<|tool_call|>"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):]
            break

    for suffix in ["<|tool_call>", "<|tool_call|>", "<|eot|>", "<eos>"]:
        if text.rstrip().endswith(suffix):
            text = text.rstrip()[:-len(suffix)]
            break

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Tool-Constrained Decoding Demo")
    print("  Model: google/gemma-4-E2B-it")
    print("  Strategy: force plan tool on the first step")
    print("=" * 70)

    # ---- Load model ----
    model, tokenizer, device = load_model_and_tokenizer()

    # ---- Build prompt ----
    print("\n[2/5] Building prompt (with 4 tool definitions)...")
    prompt = build_prompt(tokenizer, TOOLS, USER_QUERY)
    print(f"       Prompt length: {len(prompt)} chars / {len(tokenizer.encode(prompt))} tokens")
    print(f"       Registered tools: {', '.join(TOOL_SCHEMAS.keys())}")

    # ---- Create constrained decoder ----
    print("\n[3/5] Initializing constrained decoder ...")
    decoder = ToolConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        template=GEMMA_TOOL_CALL_TEMPLATE,
    )
    print("       Decoder ready (hot-plug mode)")

    # ---- Step 1: Force plan tool call ----
    print("\n[4/5] Step 1: constrained decoding → forcing 'plan' tool call ...")
    print(f"       Constraint: tool_name='plan' (locked)")
    print(f"       Arguments: steps (array), reasoning (string) — generated freely by LLM")
    print("       [busy] Generating ...")

    plan_schema = TOOL_SCHEMAS["plan"]
    get_weather_schema = TOOL_SCHEMAS["get_weather"]

    result: DecoderResult = decoder.generate(
        prompt=prompt,
        tool_name="plan",
        args_schema=plan_schema,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
    )

    print(f"\n   Done (finish_reason={result.finish_reason})")
    print(f"   Generated tokens: {len(result.token_ids)}")
    print(f"   Initial test: {result.text}")

    # ---- Parse plan ----
    tool_call = result.tool_call
    if tool_call.get("_parse_error"):
        # Manual parse
        print(f"   JSON parse failed on generated text; retrying manually...")
        tool_call = parse_tool_call(result.text)
        if tool_call is None:
            print(f"   Manual parse also failed. Raw text:")
            print(f"   {result.text[:500]}")
        else:
            print(f"   Manual parse succeeded.")
    if tool_call is None:
        print("\n   *** Failed to parse tool call ***")
    else:
        print(f"\n   Parsed result:")
        print(f"     Tool name: {tool_call.get('name', 'N/A')}")
        args = tool_call.get("arguments", {})
        steps = args.get("steps", [])
        reasoning = args.get("reasoning", "N/A")
        print(f"     Reasoning: {reasoning}")
        print(f"     Plan steps:")
        for i, step in enumerate(steps, 1):
            print(f"       {i}. {step}")

    # ---- Simulate subsequent flow ----
    print(f"\n[5/5] Subsequent steps (in actual application):")
    print(f"      1. Execute the steps listed in the plan (call corresponding tools)")
    print(f"      2. Each step may optionally use constrained decoding")
    print(f"      3. e.g. step 2 constrained to get_weather, step 3 constrained to search_web ...")

    print(f"\n{'=' * 70}")
    print(f"  Demo complete! plan tool call was successfully constrained.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
