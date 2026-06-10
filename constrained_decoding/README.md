# TC-Decoder (Tool Constrained Decoder)

即插即用、无需训练的约束解码器。在自回归生成过程中，通过有限状态机（FSM）约束 LLM 的输出，确保生成的文本**在结构上保证**是指定工具的有效调用。

## 核心思路

```
JSON Schema ──▶ NFA ──▶ DFA ──▶ LogitsProcessor ──▶ model.generate()
   (参数定义)    (字符级)  (确定化)   (逐 token 掩码)     (约束后的输出)
```

1. 将工具的 JSON Schema **编译**为字符级 NFA（非确定性有限自动机）
2. 通过子集构造法将 NFA **确定化**为 DFA
3. 在每一步解码时，LogitsProcessor 遍历词表中所有 token，将 DFA 无法接受的 token 的 logit 设为 `-inf`
4. 模型只能在"合法路径"上采样，输出**保证**是合法的工具调用

## 模块架构

```
constrained_decoding/
├── char_fsm.py              # FSM 引擎：NFA/DFA 数据结构、NFA→DFA 转换、组合子
├── json_schema_to_fsm.py    # JSON Schema → NFA 编译器
├── token_index.py           # 词表首字符索引（加速候选过滤）
├── logits_processor.py      # HuggingFace LogitsProcessor（运行时约束）
├── decoder.py               # 高层 API：ToolConstrainedDecoder
└── __init__.py              # 公共导出
```

### 数据流

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

## 模块详解

### 1. `char_fsm.py` — FSM 引擎

字符级有限状态机核心，提供 NFA 和 DFA 两种抽象：

| 组件 | 说明 |
|------|------|
| `NFAState` | NFA 状态，支持字符转移 + ε 转移，每个状态有唯一 ID，全局注册 |
| `DFAState` | DFA 状态，转移表为 `dict[char, DFAState]`，支持 `CATCHALL`（匹配未显式列出的任意字符） |
| `epsilon_closure()` | 计算 NFA 状态集的 ε 闭包 |
| `nfa_to_dfa()` | 子集构造法：NFA → DFA |
| `compute_reachability()` | 反向 BFS 预计算每个 DFA 状态能否到达 accept（用于 token 合法性判断） |
| `CATCHALL` | 哨兵字符 `\x00__CATCHALL__`，表示"任意未显式匹配的字符" |

**组合子（NFA 代数）**：

| 函数 | 等价正则 | 用途 |
|------|:--:|------|
| `build_literal_nfa(s)` | `"s"` | 匹配固定字符串 |
| `nfa_concat(a, b)` | `ab` | 串联 |
| `nfa_union(a, b)` | `a\|b` | 并（多选一） |
| `nfa_star(a)` | `a*` | 零或多次重复 |
| `nfa_optional(a)` | `a?` | 零或一次 |
| `nfa_plus(a)` | `a+` | 一或多次重复 |

### 2. `json_schema_to_fsm.py` — Schema → NFA 编译器

将 JSON Schema 编译为字符级 NFA，支持的 Schema 类型：

| 类型 | 实现 | 说明 |
|------|------|------|
| `string` | `build_string_value_nfa()` | `"..."`含转义、Unicode、CATCHALL |
| `number` | `build_number_nfa()` | RFC 8259 完整数字语法 |
| `integer` | `build_integer_nfa()` | 整数（无小数/指数） |
| `boolean` | `build_boolean_nfa()` | `true` / `false` |
| `null` | `build_null_nfa()` | `null` |
| `enum` | `build_enum_nfa()` | 枚举值并 |
| `const` | `build_const_nfa()` | 常量匹配 |
| `object` | `build_object_nfa()` | **位掩码**跟踪属性出现情况，支持 required、任意顺序 |
| `array` | `build_array_nfa()` | `[item, ...]` |

**顶层入口 `build_tool_call_nfa()`**：将 `tool_name` 和 `args_schema` 代入模板（默认 `{"name":"{name}","arguments":{arguments}}`），串联前缀、工具名、参数 NFA、后缀。

**位掩码对象编码**：对于有 N 个 required 属性的 object，使用 `2^(N+1)` 个状态（而非 N! 个排列），每个状态对应"已见过的属性集合"。额外使用一个 `HAS_PROPERTY_BIT` 控制逗号插入（首属性前不加逗号）。

### 3. `token_index.py` — 词表索引

| 类 | 说明 |
|------|------|
| `TokenEntry` | 单个 token 的元数据：`token_id`、`token_str`、`first_char` |
| `TokenIndex` | 按首字符建立倒排索引 `{first_char: [TokenEntry, ...]}`，同时维护 `id → entry` 映射 |

在 LogitsProcessor 中，先按首字符快速拒绝明显不匹配的 token，再对剩余候选做完整 DFA 行走。

### 4. `logits_processor.py` — 运行时约束

`ToolConstraintLogitsProcessor` 实现 HuggingFace `LogitsProcessor` 接口，插入 `model.generate()` 流水线：

**每步解码流程**：

```
1. 推进 DFA 状态（消费上一步新生成的 token）
2. 若 DFA 已到达 accept 状态 → 强制输出 EOS
3. 否则，遍历词表每个 token:
   a. 未知/空 token → 掩码（-inf）
   b. 首字符快速拒绝 → 掩码
   c. 完整 DFA walk → 若无法到达 accept 则掩码
4. EOS 仅在 accept 后允许；PAD 永久掩码
```

**缓存**：`(dfa_state_id, token_str) → (is_valid, next_state_id)`。同一 DFA 状态面对相同 token 时直接查表，避免重复的字符级行走。

### 5. `decoder.py` — 高层 API

`ToolConstrainedDecoder` 面向用户的唯一入口：

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

**`generate()` 内部步骤**：

1. `build_tool_call_nfa()` — Schema → NFA
2. `nfa_to_dfa()` — NFA → DFA
3. `tokenizer.encode(prompt)` — 获取 prompt token IDs
4. 创建 `ToolConstraintLogitsProcessor(dfa, token_index, ...)`
5. 调用 `model.generate(logits_processor=[processor], ...)`
6. 解析生成的文本为 `DecoderResult`

**模板感知解析**：`_parse_with_template()` 先尝试直接 JSON 解析，失败则根据模板剥离非 JSON 包裹部分再解析。

## 设计要点

### 为什么是字符级 FSM？

token 级 FSM 需要在 token 之间做约束，但 token 边界由 tokenizer 决定，与 JSON 的字符结构不对齐。字符级 DFA 每次验证整个 token 字符串能否被 DFA 接受，自然处理了 token 边界穿越 JSON 语法结构的情况。

### CATCHALL 机制

JSON string 内部允许任意 Unicode 字符（通过 `\uXXXX` 或直接编码）。CATCHALL 转移避免在 NFA 中枚举所有可能的 Unicode 字符，大幅减小 DFA 状态数。

### 位掩码对象编码

不做属性排列枚举（N! 种），而是用位掩码跟踪"已见属性集合"。每个 `(mask, property)` 组合生成一条转移路径。对于 N 个 required 属性的对象，状态数为 `2^(N+1)` 而非 N!。

### 缓存策略

`(dfa_state_id, token_str)` 粒度的缓存。大多数 token 在多个生成步骤中会被反复评估（相同 DFA 状态 + 相同 token），缓存命中率通常 >90%，显著减少字符级行走开销。

### 训练无关

整个过程不需要修改模型权重、不需要额外训练、不需要提示词调整。约束完全在解码阶段通过 logits 掩码实现，对模型本身零侵入。

## 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from constrained_decoding import ToolConstrainedDecoder

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

decoder = ToolConstrainedDecoder(model, tokenizer)

# 约束生成 plan 工具调用
result = decoder.generate(
    prompt="<|im_start|>system\n你是任务规划器。<|im_end|>\n<|im_start|>assistant\n",
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

## 约束保证

TC-Decoder 提供以下**构造性保证**（by construction，非概率性）：

- 输出 **一定是** 合法 JSON（不会出现未闭合引号、非法转义等）
- 输出 **一定包含** 指定的 `tool_name`
- 输出 **一定包含** `arguments` 对象
- `arguments` 中所有 `required` 属性 **一定存在**
- 属性值 **一定符合** 声明的类型（string/number/boolean/array/object）
- 不会出现 Schema 未定义的属性（当 `additionalProperties=false` 时）
