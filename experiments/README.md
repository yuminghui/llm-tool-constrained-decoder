# Experiments

约束解码器实验套件。6 个独立实验，每个只做一种配置，产出全套指标供事后对比。

## 目录结构

```
experiments/
├── README.md
├── config.py                              # 集中配置（模型、量化、超参、路径）
├── prompts.py                             # 共享 prompt 模板
├── evaluate.json                          # Benchmark（~60 个遥感任务）
├── trajectory_utils.py                    # 轨迹保存、评测、错误分类等共享工具
├── experiment_baseline.py                 # Exp A：空白对照
├── experiment_separate_plan.py            # Exp B：分离规划 + 约束执行
├── experiment_constrained_plan.py         # Exp C：仅约束 plan
├── experiment_full_pipeline.py            # Exp D：约束全流程
├── experiment_prompt_plan.py              # Exp E：提示词约束 plan
├── experiment_prompt_constrained_plan.py  # Exp F：专用规划提示词 + 约束 plan + 自由执行
├── evaluate/
│   ├── __init__.py
│   └── llm_judge.py                       # LLM-as-Judge 评估脚本
└── trajectories/
    ├── exp_a_baseline/                    # Exp A 输出
    ├── exp_b_separate_plan/               # Exp B 输出
    ├── exp_c_constrained_plan/            # Exp C 输出
    ├── exp_d_full_pipeline/               # Exp D 输出
    ├── exp_e_prompt_plan/                 # Exp E 输出
    └── exp_f_prompt_constrained_plan/     # Exp F 输出
```

## 实验设计

| 实验 | Plan 生成提示词 | Plan 生成解码 | 执行阶段解码 | 文件 |
|------|:--:|:--:|:--:|------|
| **A. 空白对照** | —（不生成 plan） | — | 自由生成 | `experiment_baseline.py` |
| **B. 分离规划** | `PLANNER_SYSTEM_PROMPT`（专用规划提示词） | 约束解码 | 约束全部工具 | `experiment_separate_plan.py` |
| **C. 仅约束 plan** | `SYSTEM_PROMPT`（通用提示词） | 约束解码 | 自由生成 | `experiment_constrained_plan.py` |
| **D. 约束全流程** | `SYSTEM_PROMPT`（通用提示词） | 约束解码 | 约束全部工具 | `experiment_full_pipeline.py` |
| **E. 提示词约束** | `SYSTEM_PROMPT_E`（"MUST call plan"） | 自由生成 | 自由生成 | `experiment_prompt_plan.py` |
| **F. 提示词增强约束 plan** | `PLANNER_SYSTEM_PROMPT`（专用规划提示词） | 约束解码 | 自由生成 | `experiment_prompt_constrained_plan.py` |

### 各实验说明

- **A (空白对照)**：纯自由生成 agent，系统提示词不提及 `plan`，不启用约束解码。作为所有实验的参照基准，衡量模型在无任何引导下的自发行为。

- **B (分离规划+全流程约束)**：两阶段架构。阶段 1 用专用规划提示词 + 约束解码在独立上下文中生成详细 plan；阶段 2 执行 agent 按 plan 的 `expected_tools` 逐步约束执行所有工具调用。测试"独立上下文 + 专用提示词 + 全约束执行"组合效果。

- **C (流程内规划+仅约束 plan)**：单阶段 agent，第一步用约束解码强制 `plan`（通用提示词，不提及 plan），后续全部自由生成。测试"在通用提示词下仅强制第一步规划，能否改善后续自主行为"。

- **D (流程内规划+全流程约束)**：单阶段 agent，约束解码强制 plan 和所有后续工具调用，与 C 使用相同通用提示词。测试"全程硬约束 vs 仅强制第一步"的差异。

- **E (流程内规划+无约束)**：无约束解码，纯靠提示词说"MUST call plan first"诱导模型自愿调用 plan。与 A 对比测试提示词能否替代解码器强制，与 C 对比测试"提示词诱导 vs 解码器强制"的 plan 调用率和质量。

- **F (分离规划+仅约束plan)**：两阶段架构，与 B 共用专用规划提示词，但执行阶段不约束工具调用（自由生成）。与 C 对比测试 plan 提示词质量的影响，与 B 对比测试执行约束的必要性。

### 实验对照矩阵

C/D 对比执行策略，B/F 对比执行策略；C/F 对比 plan 提示词，D/B 对比 plan 提示词。

```
               │  执行: 自由生成   │  执行: 约束全部工具
───────────────┼──────────────────┼─────────────────────
Plan 提示词:   │                  │
  SYSTEM_PROMPT │     Exp C        │      Exp D
  (通用,不提plan)│                  │
───────────────┼──────────────────┼─────────────────────
Plan 提示词:   │                  │
  PLANNER_     │     Exp F        │      Exp B
  SYSTEM_PROMPT │                  │
  (专用规划提示) │                  │
───────────────┴──────────────────┴─────────────────────
```

外围对照：

| 对比 | 变量 | 说明 |
|------|------|------|
| A vs E | 提示词是否提及 plan | 测试纯自然语言指令能否诱导 plan 调用 |
| A vs C | 是否用约束解码强制 plan | 测试解码器强制的效果 |
| E vs C | 提示词诱导 vs 解码器强制 | 两种 plan 强制执行方式对比 |
| C vs D | 仅约束 plan vs 约束全部 | 测试"先规划"是否足够，还是需要全程约束 |
| C vs F | plan 提示词（通用 vs 专用） | 测试提示词质量对 plan 及下游任务的影响 |
| B vs D | plan 提示词（专用 vs 通用）+ 上下文隔离 | 同上，但在全约束条件下 |
| B vs F | 执行策略（约束 vs 自由） | 在专用 plan 提示词下，测试执行约束的必要性 |

## LLM-as-Judge 评估 (`evaluate/llm_judge.py`)

使用外部 LLM（OpenAI 兼容 API）对实验轨迹进行四维评分（0--5 整数），产出分维度分数、总分及中文理由。

### 评分维度

| 维度 | 说明 |
|------|------|
| `step_completeness` | 步骤完整性：required/optional 步骤覆盖程度 |
| `result_accuracy` | 结果准确性：final answer / task_summary 与预期答案的匹配度 |
| `flow_reasonableness` | 流程合理性：步骤顺序是否正确、有无冗余/重复调用 |
| `robustness` | 鲁棒性：错误处理和恢复能力 |

**overall = round(mean of 4 dimensions)**，整数 0--5。

### 输出格式

```json
{
  "experiment": "exp_a_baseline",
  "model_id": "Qwen/Qwen3-0.6B",
  "avg_scores": {
    "step_completeness": 3.45,
    "result_accuracy": 3.12,
    "flow_reasonableness": 3.67,
    "robustness": 4.01,
    "overall": 3.56
  },
  "tasks": [
    {"task_id": "task_000", "overall": 4, "reasoning": "所有必须步骤均已完成...", ...}
  ]
}
```

### 使用

```bash
export OPENAI_API_KEY=sk-...

# 评估单个轨迹文件
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/Qwen_Qwen3-0.6B.json

# 评估整个目录
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/

# 自定义模型和端点
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/ \
  -o results/judge_scores/ \
  --model gpt-4.1-mini \
  --base-url https://your-proxy/v1
```

## 每个实验产出的指标

| 指标 | 说明 |
|------|------|
| `prompt_tokens` / `generated_tokens` / `total_tokens` | 每条任务的 token 消耗 |
| `completion_score` | 基于 evaluate.json ground truth 的完成评分 |
| `completion_rate` | 评分达阈值的任务占比 |
| `error_type` | 失败原因分类：success / max_turns / exception / tool_parse_error / tool_error |
| `execution trajectory` | 完整执行轨迹 JSON（含每步 action/args/result/is_constrained） |
| `plan_call_rate` | plan 工具被调用的任务占比 |
| `num_constrained_calls` / `num_free_calls` | 约束 vs 自由调用步数 |
| `tool_call_match_rate` | plan 中 expected_tools 与实际调用工具的匹配率 |
| `total_tokens_all` | 全部任务的总 token 消耗 |
| `total_success_tokens` | 成功完成任务的总 token 消耗 |
| `avg_steps` | 平均步数 |

## 配置说明 (`config.py`)

| 配置项 | 说明 |
|--------|------|
| `ALL_MODELS` | 所有可用 SLM（≤2B），格式 `(model_id, quantize_bool)` |
| `QUANTIZATION_MODE` | 量化：`"4bit"` / `"8bit"` |
| `TEMPERATURE` | 采样温度 |
| `MAX_TURNS` | 最大交互轮数 |
| `EXP_A_MODELS` | Exp A 使用的模型列表 |
| `EXP_B_MODELS` ~ `EXP_F_MODELS` | 各实验使用的模型列表 |

## 快速运行

```bash
# 实验 A：空白对照
python -m experiments.experiment_baseline --quick --max-tasks 5

# 实验 B：分离规划 + 约束执行
python -m experiments.experiment_separate_plan --quick --max-tasks 5

# 实验 C：仅约束 plan
python -m experiments.experiment_constrained_plan --quick --max-tasks 5

# 实验 D：约束全流程
python -m experiments.experiment_full_pipeline --quick --max-tasks 5

# 实验 E：提示词约束 plan
python -m experiments.experiment_prompt_plan --quick --max-tasks 5

# 实验 F：专用规划提示词 + 约束 plan + 自由执行
python -m experiments.experiment_prompt_constrained_plan --quick --max-tasks 5

# 指定单模型
python -m experiments.experiment_baseline --model Qwen/Qwen3-1.7B
```
