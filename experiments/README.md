# Experiments

Constrained decoder experiment suite for PTC-Decoder. Four experiments (A, B, E, F) produce metrics for post-hoc comparison; Experiments C and D have been deprecated.

## Directory Structure

```
experiments/
├── README.md
├── config.py                              # Centralized config (models, quantization, hyperparams, paths)
├── prompts.py                             # Shared prompt templates
├── evaluate.json                          # Benchmark (200 remote sensing tasks)
├── trajectory_utils.py                    # Shared utilities: trajectory saving, evaluation, error classification
├── experiment_baseline.py                 # Exp A: Baseline (blank control)
├── experiment_separate_plan.py            # Exp B: PTC-Decoder (separate plan + constrained execution)
├── experiment_prompt_plan.py              # Exp E: Prompt-only plan enforcement
├── experiment_prompt_constrained_plan.py  # Exp F: Dedicated planner + constrained plan + free execution
├── evaluate/
│   ├── __init__.py
│   └── llm_judge.py                       # LLM-as-Judge evaluation script
└── trajectories/                          # Experiment outputs (git-ignored)
    ├── exp_a_baseline/
    ├── exp_b_separate_plan/
    ├── exp_e_prompt_plan/
    └── exp_f_prompt_constrained_plan/
```

## Experiment Design

| Experiment | Plan Prompt | Plan Decoding | Execution Decoding | File |
|------|:--:|:--:|:--:|------|
| **A. Baseline** | — (no plan) | — | Free | `experiment_baseline.py` |
| **B. PTC-Decoder (Ours)** | `PLANNER_SYSTEM_PROMPT` | Constrained | Constrained | `experiment_separate_plan.py` |
| **E. Prompt-Only Plan** | "MUST call plan first" | Free | Free | `experiment_prompt_plan.py` |
| **F. Plan w/o TC-Decoder** | `PLANNER_SYSTEM_PROMPT` | Constrained | Free | `experiment_prompt_constrained_plan.py` |

> **Deprecated**: Experiments C (constrained plan only) and D (full constrained pipeline with generic prompt) have been removed in favor of the cleaner A/B/E/F design matrix above.

### Experiment Descriptions

- **A (Baseline)**: Pure free-generation agent. System prompt does not mention `plan`, constrained decoding disabled. Serves as the reference for all experiments.

- **B (PTC-Decoder / Ours)**: Two-phase architecture. Phase 1 uses a dedicated planner prompt + constrained decoding to generate a detailed plan in an isolated context. Phase 2 executes the agent by constraining all tool calls step by step according to the plan's `expected_tools`.

- **E (Prompt-Only Plan)**: No constrained decoding. Relies purely on a system prompt saying "MUST call plan first". Tests whether natural-language instructions alone can induce plan-following behavior in SLMs.

- **F (Plan w/o TC-Decoder)**: Two-phase architecture sharing the dedicated planner prompt with B, but execution uses free generation (no tool constraints). Serves as the ablation baseline isolating TC-Decoder's contribution.

### Experiment Comparison Matrix

```
               │  Execution: Free       │  Execution: Constrained
───────────────┼────────────────────────┼────────────────────────
Plan prompt:   │                        │
  "MUST call   │        Exp E           │          —
  plan first"  │                        │
───────────────┼────────────────────────┼────────────────────────
Plan prompt:   │                        │
  PLANNER_     │        Exp F           │         Exp B
  SYSTEM_PROMPT │  (Plan w/o TC-Decoder) │    (PTC-Decoder)
────────────────┴────────────────────────┴────────────────────────
```

Key comparisons:

| Comparison | Variable | Description |
|------|------|------|
| A vs B | Full PTC-Decoder vs. nothing | Overall effectiveness of our method |
| A vs E | Prompt wording only | Can SLMs follow plan instructions without constraints? |
| E vs B | Prompt-induced vs. decoder-enforced | Do we need constraint decoding, or is prompting enough? |
| F vs B | Execution constraints (TC-Decoder) | Ablation: the marginal contribution of TC-Decoder |

## LLM-as-Judge Evaluation (`evaluate/llm_judge.py`)

Uses an external LLM (OpenAI-compatible API) to score agent trajectories.

### Scoring Dimensions (0--5 integer)

| Dimension | Description |
|------|------|
| `step_completeness` | Coverage of required and optional benchmark steps |
| `result_accuracy` | Alignment of final answer / task summary with expected outcome |
| `flow_reasonableness` | Logical ordering, absence of redundant or nonsensical tool calls |
| `robustness` | Error handling and recovery |
| `overall` | Rounded mean of the four dimensions above |

### Output Format

```json
{
  "experiment": "exp_b_separate_plan",
  "model_id": "Qwen/Qwen3-0.6B",
  "avg_scores": {
    "step_completeness": 3.45,
    "result_accuracy": 3.12,
    "flow_reasonableness": 3.67,
    "robustness": 4.01,
    "overall": 3.56
  },
  "tasks": [ ... ]
}
```

### Usage

```bash
export OPENAI_API_KEY=sk-...

# Single trajectory file
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_b_separate_plan/Qwen_Qwen3-0.6B.json

# Entire directory
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_b_separate_plan/

# Custom model and endpoint
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_b_separate_plan/ \
  -o results/judge_scores/ \
  --model gpt-4.1-mini \
  --base-url https://your-proxy/v1
```

## Metrics Produced per Experiment

| Metric | Description |
|------|------|
| `prompt_tokens` / `generated_tokens` / `total_tokens` | Token consumption per task |
| `completion_score` | Rule-based completion against evaluate.json ground truth |
| `completion_rate` | Proportion of tasks reaching the completion threshold (0.8) |
| `success_rate` | Proportion of tasks ending without crash or timeout |
| `error_type` | Failure classification: success / max_turns / exception / tool_parse_error / tool_error |
| `trajectory` | Full execution trace (action/args/result/is_constrained per step) |
| `plan_call_rate` | Proportion of tasks where the `plan` tool was invoked |
| `planning_subtask` | Plan generation metadata (B and F only) |
| `num_constrained_calls` / `num_free_calls` | Constrained vs. free generation step counts |
| `tool_call_match_rate` | Recall / Precision / F1 of plan tools vs. executed tools |

## Configuration (`config.py`)

| Item | Description |
|------|------|
| `ALL_MODELS` | All available SLMs (≤2B) |
| `QUANTIZATION_MODE` | `"4bit"` or `"8bit"` |
| `TEMPERATURE` | Sampling temperature (default 0.7) |
| `MAX_TURNS` | Maximum agent interaction turns (default 10) |
| `PLAN_MAX_NEW_TOKENS` | Token budget for constrained plan generation |
| `FREE_MAX_NEW_TOKENS` | Token budget for free-generation steps |
| `COMPLETION_THRESHOLD` | Completion score threshold (default 0.8) |

## Quick Run

```bash
# Experiment A: Baseline
python -m experiments.experiment_baseline --quick --max-tasks 5

# Experiment B: PTC-Decoder (Ours)
python -m experiments.experiment_separate_plan --quick --max-tasks 5

# Experiment E: Prompt-Only Plan
python -m experiments.experiment_prompt_plan --quick --max-tasks 5

# Experiment F: Plan w/o TC-Decoder
python -m experiments.experiment_prompt_constrained_plan --quick --max-tasks 5

# Single model
python -m experiments.experiment_baseline --model Qwen/Qwen3-1.7B
```
