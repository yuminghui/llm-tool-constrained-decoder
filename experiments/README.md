# Experiments

约束解码器（constrained decoder）相关的实验套件，用于评估 SLM Agent 在不同配置下的性能。

## 目录结构

```
experiments/
├── README.md                        # 本文件
├── __init__.py
├── config.py                        # 集中配置（模型、量化、超参、路径）
├── evaluate.json                    # Benchmark 任务集（~60 个遥感+变化检测任务）
├── trajectory_utils.py              # 轨迹保存/追加工具函数
├── experiment_context_tokens.py     # 实验1+2
├── experiment_plan.py               # 实验3
├── experiment_ablation.py           # 实验4
├── experiment_prompt_plan.py        # 实验5
└── trajectories/                    # 实验输出
    ├── exp1_cross_model/            # 实验1 按模型分文件
    ├── exp2/                        # 实验2 按模型分文件
    ├── exp3/                        # 实验3 按模型分文件
    ├── exp4/                        # 实验4 按模型分文件
    └── exp5/                        # 实验5 按模型分文件
```

## 配置说明 (`config.py`)

所有实验的超参数集中管理：

| 配置项 | 说明 |
|--------|------|
| `ALL_MODELS` | 所有可用 SLM（≤2B），格式 `(model_id, quantize_bool)` |
| `QUANTIZATION_MODE` | 量化模式：`"4bit"` / `"8bit"` |
| `TEMPERATURE` | 采样温度（0.7） |
| `MAX_TURNS` | 单次 agent 最大交互轮数（10） |
| `PLAN_MAX_NEW_TOKENS` | 约束解码 plan 的最大 token（256） |
| `FREE_MAX_NEW_TOKENS` | 自由生成的最大 token（512） |
| `COMPLETION_THRESHOLD` | 任务完成判定阈值（0.8） |

模型列表以 `(model_id, quantize)` 元组配置，按模型粒度控制是否启用量化：

```python
ALL_MODELS = [
    ("Qwen/Qwen3-0.6B",          False),
    ("google/gemma-4-E2B-it",    True),   # 2B MoE, 需 4bit
    ...
]
```

修改 `config.py` 即可调整所有实验的行为，无需改动实验代码。

---

## 实验 1 — 跨模型 Token 消耗对比

**文件**: `experiment_context_tokens.py`  
**运行**: `python -m experiments.experiment_context_tokens --exp 1`

**目的**: 对比不同 SLM 在相同 benchmark 上完成任务时的 token 消耗量。

**方法**: 对 `evaluate.json` 中的每个任务，用每个候选模型跑 plan-then-act agent（约束解码器启用），记录 prompt + generated tokens。

**输出**: `trajectories/exp1_cross_model/{model_name}.json`

---

## 实验 2 — 约束内联规划 vs 分离规划

**文件**: `experiment_context_tokens.py`  
**运行**: `python -m experiments.experiment_context_tokens --exp 2`

**目的**: 对比两种 plan-then-act 策略的 token 消耗和完成率：

- **A（内联规划）**: 同一个 session 中 step 0 约束解码器生成 plan → 继续执行
- **B（分离规划）**: 先独立运行 plan-only 子任务 → 再把 plan 注入新 session 执行

**关键指标**: 总 token = A 全程 vs B（plan 子任务 + execution agent）

**输出**: `trajectories/exp2/{model}_a_inline.json` + `{model}_b_separate.json`

---

## 实验 3 — Plan-first vs No-plan（LLM-as-Judge）

**文件**: `experiment_plan.py`  
**运行**: `python -m experiments.experiment_plan`

**目的**: 测试"先规划再行动"是否对任务完成质量有影响。

| 模式 | 说明 |
|------|------|
| **P+ (with plan)** | 约束解码器强制 step 0 调用 `plan`，按计划逐步执行 |
| **P- (without plan)** | 不规划，模型直接调用工具 |

**输出**: `trajectories/exp3/{model}_plan_yes.json` + `{model}_plan_no.json`

---

## 实验 4 — 消融实验：约束解码器对 Plan 步骤的影响 (LLM-as-Judge)

**文件**: `experiment_ablation.py`  
**运行**: `python -m experiments.experiment_ablation`

**目的**: 在严格控制变量的条件下，唯一变量是第一步 `plan` 是否使用约束解码器。

| 控制变量 | 值 |
|----------|-----|
| 模型 | `EXP4_MODELS` 中的模型 |
| System prompt | 同一段（均鼓励先 plan 再行动） |
| 温度 / 任务集 / 工具集 | 完全相同 |

**唯一变量**:

|  | A（约束解码器 on） | B（全自由生成） |
|------|------|------|
| **Plan 步骤** | 约束解码器强制 | 自由生成（模型自决） |
| **后续步骤** | 自由生成 | 自由生成 |

**输出**: `trajectories/exp4/{model}_with_constraint.json` + `{model}_without_constraint.json`

---

## 实验 5 — 提示词能否替代约束解码器 (LLM-as-Judge)

**文件**: `experiment_prompt_plan.py`  
**运行**: `python -m experiments.experiment_prompt_plan`

**目的**: 测试仅靠提示词（"MUST call plan first"）能否让模型真的先用 plan。
全部自由生成，无约束解码器介入。

**核心指标**:

| 指标 | 含义 |
|------|------|
| `plan_call_rate` | 调用 `plan` 的比例（任意位置） |
| `plan_first_rate` | `plan` 作为第一个工具调用的比例 |
| `completion_rate` | 标准任务完成率 |

**输出**: `trajectories/exp5/{model_name}.json`

**对比价值**: 与 Exp3（约束解码器强制 plan）、Exp4（消融实验）形成三角对照：
- Exp3: plan 100% 强制 → 上限
- Exp4: plan 约束 vs 自由 → 隔离约束解码器的效果
- Exp5: 提示词 only → 模型"自觉性"测试

---

## 实验全景

```
              ┌─────────────────────────────────────┐
              │         Plan 是否被调用？             │
              ├──────────┬──────────┬───────────────┤
              │ 强制调用  │ 提示词引导 │   不提 plan   │
    ┌─────────┼──────────┼──────────┼───────────────┤
    │约束解码器│ Exp1     │          │               │
    │   on    │ Exp2A    │   Exp4A  │               │
    │         │ Exp3 P+  │          │               │
    ├─────────┼──────────┼──────────┼───────────────┤
    │约束解码器│          │          │               │
    │  off    │    —     │   Exp5   │  Exp3 P-      │
    │         │          │   Exp4B  │               │
    └─────────┴──────────┴──────────┴───────────────┘
```

---

## 输出文件格式

每个轨迹 JSON 文件包含：

```json
[
  {
    "question": "任务描述...",
    "model_id": "Qwen/Qwen3-0.6B",
    "config": "with_plan",
    "prompt_tokens": 200,
    "generated_tokens": 300,
    "total_tokens": 500,
    "success": true,
    "trajectory": {
      "steps": [
        {"action": "plan", "is_constrained": true, "args": {...}},
        {"action": "gf_pms_preprocess_cli", "is_constrained": false, "args": {...}}
      ]
    }
  },
  {
    "type": "_summary",
    "total_tasks": 10,
    "total_tokens_all": 5000,
    "avg_tokens_per_task": 500.0,
    ...
  }
]
```

轨迹记录 `trajectory.steps[].action` 与 `evaluate.json` 中 `trajectory_ground_truth.steps[].action` 格式一致，可直接对比评测。

---

## 快速运行

```bash
# 实验1：跨模型 token 对比
python -m experiments.experiment_context_tokens --exp 1 --quick --max-tasks 3

# 实验2：内联 vs 分离规划
python -m experiments.experiment_context_tokens --exp 2 --max-tasks 5

# 实验1+2 全部
python -m experiments.experiment_context_tokens --exp all

# 实验3：plan vs no-plan
python -m experiments.experiment_plan --quick --max-tasks 5

# 实验4：消融实验
python -m experiments.experiment_ablation --quick --max-tasks 5

# 实验5：提示词能否替代约束解码器
python -m experiments.experiment_prompt_plan --quick --max-tasks 5

# 指定单模型
python -m experiments.experiment_plan --model Qwen/Qwen3-1.7B
```
