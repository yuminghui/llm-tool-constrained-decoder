# Experiments

Constrained decoder experiment suite. Six independent experiments, each running a single configuration to produce a full set of metrics for post-hoc comparison.

## Directory Structure

```
experiments/
├── README.md
├── config.py                              # Centralized config (models, quantization, hyperparams, paths)
├── prompts.py                             # Shared prompt templates
├── evaluate.json                          # Benchmark (~60 remote sensing tasks)
├── trajectory_utils.py                    # Shared utilities: trajectory saving, evaluation, error classification
├── experiment_baseline.py                 # Exp A: Blank control
├── experiment_separate_plan.py            # Exp B: Separate plan + constrained execution
├── experiment_constrained_plan.py         # Exp C: Constrained plan only
├── experiment_full_pipeline.py            # Exp D: Full constrained pipeline
├── experiment_prompt_plan.py              # Exp E: Prompt-constrained plan
├── experiment_prompt_constrained_plan.py  # Exp F: Dedicated planner prompt + constrained plan + free execution
├── evaluate/
│   ├── __init__.py
│   └── llm_judge.py                       # LLM-as-Judge evaluation script
└── trajectories/
    ├── exp_a_baseline/                    # Exp A output
    ├── exp_b_separate_plan/               # Exp B output
    ├── exp_c_constrained_plan/            # Exp C output
    ├── exp_d_full_pipeline/               # Exp D output
    ├── exp_e_prompt_plan/                 # Exp E output
    └── exp_f_prompt_constrained_plan/     # Exp F output
```

## Experiment Design

| Experiment | Plan Generation Prompt | Plan Generation Decoding | Execution Decoding | File |
|------|:--:|:--:|:--:|------|
| **A. Blank control** | — (no plan) | — | Free generation | `experiment_baseline.py` |
| **B. Separate plan** | `PLANNER_SYSTEM_PROMPT` (dedicated planner prompt) | Constrained | Constrain all tools | `experiment_separate_plan.py` |
| **C. Constrained plan only** | `SYSTEM_PROMPT` (generic prompt) | Constrained | Free generation | `experiment_constrained_plan.py` |
| **D. Full constrained pipeline** | `SYSTEM_PROMPT` (generic prompt) | Constrained | Constrain all tools | `experiment_full_pipeline.py` |
| **E. Prompt-constrained** | `SYSTEM_PROMPT_E` ("MUST call plan") | Free generation | Free generation | `experiment_prompt_plan.py` |
| **F. Prompt-enhanced constrained plan** | `PLANNER_SYSTEM_PROMPT` (dedicated planner prompt) | Constrained | Free generation | `experiment_prompt_constrained_plan.py` |

### Experiment Descriptions

- **A (Blank control)**: Pure free-generation agent. The system prompt does not mention `plan`, and constrained decoding is disabled. Serves as the baseline for all experiments, measuring the model's spontaneous behavior with no guidance.

- **B (Separate plan + full constrained execution)**: Two-phase architecture. Phase 1 uses a dedicated planner prompt + constrained decoding to generate a detailed plan in an isolated context. Phase 2 executes the agent by constraining all tool calls step by step according to the plan's `expected_tools`. Tests the combined effect of "isolated context + dedicated prompt + full constrained execution."

- **C (In-flow plan + constrained plan only)**: Single-phase agent. Step 1 uses constrained decoding to force a `plan` call (with a generic prompt that does not mention plan), and all subsequent steps use free generation. Tests "whether forcing only the first step to plan with a generic prompt improves subsequent autonomous behavior."

- **D (In-flow plan + full constrained pipeline)**: Single-phase agent. Constrained decoding forces the plan and all subsequent tool calls, using the same generic prompt as C. Tests the difference between "full hard constraints vs. forcing only the first step."

- **E (In-flow plan + no constraints)**: No constrained decoding. Relies purely on the prompt saying "MUST call plan first" to induce the model to voluntarily call plan. Compared with A, tests whether prompts can substitute for decoder enforcement. Compared with C, tests "prompt-induced vs. decoder-enforced" plan call rate and quality.

- **F (Separate plan + constrained plan only)**: Two-phase architecture. Shares the dedicated planner prompt with B, but does not constrain tool calls during execution (free generation). Compared with C, tests the impact of plan prompt quality. Compared with B, tests the necessity of execution constraints.

### Experiment Comparison Matrix

C/D compare execution strategy; B/F compare execution strategy. C/F compare plan prompt; D/B compare plan prompt.

```
               │  Execution: Free gen  │  Execution: Constrain all
───────────────┼──────────────────────┼─────────────────────────
Plan prompt:   │                      │
  SYSTEM_PROMPT │        Exp C         │         Exp D
  (generic)     │                      │
───────────────┼──────────────────────┼─────────────────────────
Plan prompt:   │                      │
  PLANNER_     │        Exp F         │         Exp B
  SYSTEM_PROMPT │                      │
  (dedicated)   │                      │
───────────────┴──────────────────────┴─────────────────────────
```

Peripheral comparisons:

| Comparison | Variable | Description |
|------|------|------|
| A vs E | Whether prompt mentions plan | Tests whether pure natural language instructions can induce plan calls |
| A vs C | Whether constrained decoding enforces plan | Tests the effect of decoder enforcement |
| E vs C | Prompt-induced vs. decoder-enforced | Compares two plan enforcement approaches |
| C vs D | Constrained plan only vs. constrain all | Tests whether "plan first" is sufficient or full constraints are needed |
| C vs F | Plan prompt (generic vs. dedicated) | Tests the impact of prompt quality on plan and downstream tasks |
| B vs D | Plan prompt (dedicated vs. generic) + context isolation | Same as above, but under full constraint conditions |
| B vs F | Execution strategy (constrained vs. free) | Under dedicated plan prompt, tests the necessity of execution constraints |

## LLM-as-Judge Evaluation (`evaluate/llm_judge.py`)

Uses an external LLM (OpenAI-compatible API) to score experiment trajectories across four dimensions (0--5 integer), producing per-dimension scores, an overall score, and Chinese-language reasoning.

### Scoring Dimensions

| Dimension | Description |
|------|------|
| `step_completeness` | Step completeness: coverage of required/optional steps |
| `result_accuracy` | Result accuracy: how well the final answer / task_summary matches the expected outcome |
| `flow_reasonableness` | Flow reasonableness: whether step order is correct, presence of redundant/duplicate calls |
| `robustness` | Robustness: error handling and recovery ability |

**overall = round(mean of 4 dimensions)**, integer 0--5.

### Output Format

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
    {"task_id": "task_000", "overall": 4, "reasoning": "All required steps completed...", ...}
  ]
}
```

### Usage

```bash
export OPENAI_API_KEY=sk-...

# Evaluate a single trajectory file
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/Qwen_Qwen3-0.6B.json

# Evaluate an entire directory
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/

# Custom model and endpoint
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_a_baseline/ \
  -o results/judge_scores/ \
  --model gpt-4.1-mini \
  --base-url https://your-proxy/v1
```

## Metrics Produced per Experiment

| Metric | Description |
|------|------|
| `prompt_tokens` / `generated_tokens` / `total_tokens` | Token consumption per task |
| `completion_score` | Completion score based on evaluate.json ground truth |
| `completion_rate` | Proportion of tasks whose score reaches the threshold |
| `error_type` | Failure reason classification: success / max_turns / exception / tool_parse_error / tool_error |
| `execution trajectory` | Full execution trajectory as JSON (including action/args/result/is_constrained per step) |
| `plan_call_rate` | Proportion of tasks where the plan tool was called |
| `num_constrained_calls` / `num_free_calls` | Constrained vs. free generation step counts |
| `tool_call_match_rate` | Match rate between plan expected_tools and actually called tools |
| `total_tokens_all` | Total token consumption across all tasks |
| `total_success_tokens` | Total token consumption for successfully completed tasks |
| `avg_steps` | Average number of steps |

## Configuration (`config.py`)

| Config Item | Description |
|--------|------|
| `ALL_MODELS` | All available SLMs (≤2B), format `(model_id, quantize_bool)` |
| `QUANTIZATION_MODE` | Quantization: `"4bit"` / `"8bit"` |
| `TEMPERATURE` | Sampling temperature |
| `MAX_TURNS` | Maximum interaction turns |
| `EXP_A_MODELS` | Model list for Exp A |
| `EXP_B_MODELS` ~ `EXP_F_MODELS` | Model lists for each experiment |

## Quick Run

```bash
# Experiment A: Blank control
python -m experiments.experiment_baseline --quick --max-tasks 5

# Experiment B: Separate plan + constrained execution
python -m experiments.experiment_separate_plan --quick --max-tasks 5

# Experiment C: Constrained plan only
python -m experiments.experiment_constrained_plan --quick --max-tasks 5

# Experiment D: Full constrained pipeline
python -m experiments.experiment_full_pipeline --quick --max-tasks 5

# Experiment E: Prompt-constrained plan
python -m experiments.experiment_prompt_plan --quick --max-tasks 5

# Experiment F: Dedicated planner prompt + constrained plan + free execution
python -m experiments.experiment_prompt_constrained_plan --quick --max-tasks 5

# Single model
python -m experiments.experiment_baseline --model Qwen/Qwen3-1.7B
```
