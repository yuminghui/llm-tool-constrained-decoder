# Experiments

约束解码器实验套件。4 个独立实验，每个只做一种配置，产出全套指标供事后对比。

## 目录结构

```
experiments/
├── README.md
├── config.py                        # 集中配置（模型、量化、超参、路径）
├── evaluate.json                    # Benchmark（~60 个遥感+变化检测任务）
├── trajectory_utils.py              # 轨迹保存、评测、错误分类等共享工具
├── experiment_baseline.py           # Exp A：空白对照
├── experiment_separate_plan.py      # Exp B：分离规划
├── experiment_constrained_plan.py   # Exp C：仅约束 plan
├── experiment_full_pipeline.py      # Exp D：约束全流程
└── trajectories/
    ├── exp_a_baseline/              # Exp A 输出
    ├── exp_b_separate_plan/         # Exp B 输出
    ├── exp_c_constrained_plan/      # Exp C 输出
    └── exp_d_full_pipeline/         # Exp D 输出
```

## 实验设计

| 实验 | 约束解码器 | Plan 调用 | 文件 |
|------|------|------|------|
| **A. 空白对照** | 无 | 无 plan | `experiment_baseline.py` |
| **B. 分离规划** | plan 子任务 + 执行全约束 | 独立子任务生成 | `experiment_separate_plan.py` |
| **C. 仅约束 plan** | 仅 step 0（plan） | 约束解码器强制 | `experiment_constrained_plan.py` |
| **D. 约束全流程** | plan + 所有后续工具 | 约束解码器强制 | `experiment_full_pipeline.py` |

```
              ┌──────────────┬──────────────┬──────────────┐
              │   无 plan     │  仅约束 plan  │  约束全流程   │
    ┌─────────┼──────────────┼──────────────┼──────────────┤
    │   无     │   Exp A      │      —       │      —       │
    │约束解码器│   (空白对照)  │              │              │
    ├─────────┼──────────────┼──────────────┼──────────────┤
    │   有     │   Exp B      │   Exp C      │   Exp D      │
    │约束解码器│   (分离规划)  │  (仅约束plan) │  (全流程)    │
    └─────────┴──────────────┴──────────────┴──────────────┘
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
| `EXP_B_MODELS` 等 | 各实验使用的模型列表 |

## 快速运行

```bash
# 实验 A：空白对照
python -m experiments.experiment_baseline --quick --max-tasks 5

# 实验 B：分离规划
python -m experiments.experiment_separate_plan --quick --max-tasks 5

# 实验 C：仅约束 plan
python -m experiments.experiment_constrained_plan --quick --max-tasks 5

# 实验 D：约束全流程
python -m experiments.experiment_full_pipeline --quick --max-tasks 5

# 指定单模型
python -m experiments.experiment_baseline --model Qwen/Qwen3-1.7B
```
