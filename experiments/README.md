# Experiments

约束解码器（constrained decoder）相关的实验套件，用于评估 SLM Agent 在不同配置下的性能。

## 目录结构

```
experiments/
├── README.md                        # 本文件
├── __init__.py
├── config.py                        # 所有实验的集中配置（模型、超参、路径）
├── evaluate.json                    # Benchmark 任务集（~60 个遥感+变化检测任务）
├── trajectory_utils.py              # 轨迹保存/追加工具函数
├── experiment_context_tokens.py     # 实验1+2：token 消耗对比
├── experiment_plan.py               # 实验3：plan-first vs no-plan
└── trajectories/                    # 实验输出的轨迹 JSON 文件
    ├── plan_yes.json
    └── plan_no.json
```

## 配置说明 (`config.py`)

所有实验的超参数集中管理：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ALL_MODELS` | Qwen2.5-0.5B/1.5B/3B | 可用的 SLM 模型列表 |
| `TEMPERATURE` | 0.7 | 采样温度 |
| `MAX_TURNS` | 10 | 单次 agent 最大交互轮数 |
| `PLAN_MAX_NEW_TOKENS` | 256 | 约束解码 plan 生成的最大 token |
| `FREE_MAX_NEW_TOKENS` | 512 | 自由生成的最大 token |
| `COMPLETION_THRESHOLD` | 0.8 | 任务完成判定阈值 |

修改 `config.py` 即可调整所有实验的行为，无需改动实验代码。

---

## 实验 1 — 跨模型 Token 消耗对比

**文件**: `experiment_context_tokens.py`  
**运行**: `python -m experiments.experiment_context_tokens --exp 1`

**目的**: 对比不同 SLM 在相同 benchmark 上完成任务时的 token 消耗量。

**方法**: 对 `evaluate.json` 中的每个任务，用每个候选模型跑一遍 plan-then-act agent（约束解码器启用），记录 prompt tokens + generated tokens。

**输出**:

```
Model                                     Avg Tokens  Avg Gen  Completion   Success
--------------------------------------------------------------------------------
Qwen/Qwen2.5-0.5B-Instruct                    1234      567       85.0%     90.0%
Qwen/Qwen2.5-1.5B-Instruct                    1100      450       90.0%     95.0%
Qwen/Qwen2.5-3B-Instruct                      1300      600       88.0%     92.0%
```

---

## 实验 2 — 约束内联规划 vs 分离规划

**文件**: `experiment_context_tokens.py`  
**运行**: `python -m experiments.experiment_context_tokens --exp 2`

**目的**: 对比两种 plan-then-act 策略的 token 消耗和完成率：

- **Approach A（内联规划）**: 在同一个 agent session 中，step 0 用约束解码器强制生成 plan，然后继续执行
- **Approach B（分离规划）**: 先独立运行一个 plan-only 子任务生成规划，再把规划注入到新的 agent session 中执行

**关键指标**: 总 token = A 的全程 token vs B 的（plan 子任务 token + execution agent token）

**输出**:

```
Metric                           A-Inline      B-Separate          Delta
----------------------------------------------------------------------------------
Avg Total Tokens                      1234            1456           -222
Completion Rate                      85.0%           82.0%          +3.0%
```

---

## 实验 3 — Plan-first vs No-plan（LLM-as-Judge）

**文件**: `experiment_plan.py`  
**运行**: `python -m experiments.experiment_plan`

**目的**: 测试"先规划再行动"是否对 SLM Agent 的任务完成质量有影响。

**方法**: 同一模型、同一批任务，跑两种模式：

| 模式 | 说明 |
|------|------|
| **P+ (with plan)** | 约束解码器强制 step 0 调用 `plan`，再按计划逐步执行 |
| **P- (without plan)** | 不规划，模型看了任务后直接调用工具 |

**输出文件**:

- `trajectories/plan_yes.json` — P+ 模式的所有任务执行轨迹
- `trajectories/plan_no.json` — P- 模式的所有任务执行轨迹

轨迹格式与 `evaluate.json` 兼容，后续用 LLM-as-Judge 对比评测。

---

## 工具函数 (`trajectory_utils.py`)

| 函数 | 用途 |
|------|------|
| `save_trajectory(result, path, ...)` | 单个 agent 结果 → 新 JSON 文件（覆盖写） |
| `append_trajectory(result, path, ...)` | 单个 agent 结果 → 追加到已有 JSON 数组 |
| `save_trajectories_batch(results, path, ...)` | 批量结果 → 新 JSON 文件 |
| `agent_result_to_record(result, ...)` | AgentResult → evaluate.json 兼容的 dict |

轨迹记录的 `trajectory.steps[].action` 字段与 `evaluate.json` 中 `trajectory_ground_truth.steps[].action` 格式一致，可直接用于对比评测。

---

## 快速运行

```bash
# 实验1：快速测试（单模型、少量任务）
python -m experiments.experiment_context_tokens --exp 1 --quick --max-tasks 3

# 实验2：完整运行
python -m experiments.experiment_context_tokens --exp 2

# 实验1+2 全部
python -m experiments.experiment_context_tokens --exp all

# 实验3：plan vs no-plan
python -m experiments.experiment_plan --quick --max-tasks 5

# 实验3：指定模型
python -m experiments.experiment_plan --model Qwen/Qwen2.5-1.5B-Instruct
```
