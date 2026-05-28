"""
Tool-Constrained Decoding Demo
===============================
使用 google/gemma-4-E2B-it 演示约束解码：
  第一步强制调用 plan 工具（工具名锁死，参数由 LLM 自由生成）
  之后可以自由调用其他工具

工具集:
  - plan:           制定执行计划（第一步必须调用）
  - get_weather:    查询天气
  - search_web:     搜索网络
  - calculator:     数学计算
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
# Gemma 4 tool-call 格式模板
# Gemma 用 <tool_call|> 开头、<|tool_call> 结尾包裹 JSON
# ---------------------------------------------------------------------------
GEMMA_TOOL_CALL_TEMPLATE = '<tool_call|>{"name":"{name}","arguments":{arguments}}<|tool_call>'


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": "在执行任何操作之前先制定计划。这是第一步必须调用的工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "有序的执行步骤列表",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "制定该计划的理由",
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
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
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
            "description": "在网络上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大返回结果数",
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
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# 工具 schema 映射表（给约束解码器用）
TOOL_SCHEMAS = {
    tool["function"]["name"]: tool["function"]["parameters"]
    for tool in TOOLS
}


# ---------------------------------------------------------------------------
# 场景提示词
# ---------------------------------------------------------------------------
USER_QUERY = (
    "I need to research the latest AI industry trends for 2026, "
    "check the weather in San Francisco for an outdoor event, "
    "and calculate the total addressable market size ($50B growing at 15% for 3 years). "
    # "Please plan your approach first before doing anything."
)


def load_model_and_tokenizer(model_id: str = "google/gemma-4-E2B-it"):
    """加载模型和分词器。"""
    print(f"[1/5] 加载模型: {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Gemma 4 需要 PyTorch 支持 compute capability 12.0+ for RTX 50 系列
    # 如果 CUDA 不兼容则退回 CPU
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

    print(f"       模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer, device


def build_prompt(tokenizer, tools: list, user_query: str) -> str:
    """使用 Gemma chat template 构建包含工具定义的 prompt。"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to external tools. "
                # "Always plan first using the 'plan' tool before calling any other tool. "
                # "After planning, execute tools one at a time."
                "Execute tools one at a time."
                "Please response in Chinese."
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
    """从 Gemma 的 tool call 输出中解析 JSON。"""
    # 去除 Gemma 的特殊 token 包裹
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
# 主演示
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Tool-Constrained Decoding 演示")
    print("  模型: google/gemma-4-E2B-it")
    print("  策略: 第一步强制调用 plan 工具")
    print("=" * 70)

    # ---- 加载模型 ----
    model, tokenizer, device = load_model_and_tokenizer()

    # ---- 构建 prompt ----
    print("\n[2/5] 构建 Prompt（含 4 个工具定义）...")
    prompt = build_prompt(tokenizer, TOOLS, USER_QUERY)
    print(f"       Prompt 长度: {len(prompt)} 字符 / {len(tokenizer.encode(prompt))} tokens")
    print(f"       已注册工具: {', '.join(TOOL_SCHEMAS.keys())}")

    # ---- 创建约束解码器 ----
    print("\n[3/5] 初始化约束解码器 ...")
    decoder = ToolConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        template=GEMMA_TOOL_CALL_TEMPLATE,
    )
    print("       解码器就绪（热插拔模式）")

    # ---- 第一步：强制调用 plan 工具 ----
    print("\n[4/5] 第一步：约束解码 → 强制调用 'plan' 工具 ...")
    print(f"       约束: tool_name='plan'（锁死）")
    print(f"       参数: steps (array), reasoning (string) — 由 LLM 自由生成")
    print("       [busy] 生成中 ...")

    plan_schema = TOOL_SCHEMAS["plan"]
    get_weather_schema = TOOL_SCHEMAS["get_weather"]

    result: DecoderResult = decoder.generate(
        prompt=prompt,
        tool_name="plan",
        args_schema=plan_schema,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )

    print(f"\n   Done (finish_reason={result.finish_reason})")
    print(f"   Generated tokens: {len(result.token_ids)}")
    print(f"   Initial test: {result.text}")

    # ---- 解析 plan ----
    tool_call = result.tool_call
    if tool_call.get("_parse_error"):
        # 手动解析
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

    # ---- 模拟后续流程 ----
    print(f"\n[5/5] 后续步骤模拟（实际应用中）：")
    print(f"      1. 执行 plan 中列出的步骤（调用对应工具）")
    print(f"      2. 每一步都可以选择是否使用约束解码")
    print(f"      3. 例如第2步约束到 get_weather, 第3步约束到 search_web ...")

    print(f"\n{'=' * 70}")
    print(f"  演示完成！plan 工具调用成功被约束。")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
