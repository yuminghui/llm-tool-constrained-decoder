# External-Dataset Generalization Experiments (`experiments/others`)

This suite reproduces the paper's **three comparison groups** on **three external,
open-source tool-calling agent benchmarks**, to show that PTC-Decoder generalizes
beyond the in-house remote-sensing benchmark (`experiments/evaluate.json`).

| Group | ≡ main-suite exp | Plan prompt | Plan decoding | Execution decoding |
|-------|:---:|:---:|:---:|:---:|
| **A. Baseline** | `experiment_baseline` | — (no plan) | — | free |
| **B. PTC-Decoder (ours)** | `experiment_separate_plan` | planner prompt | **constrained** | **constrained** |
| **F. Plan w/o TC-Decoder** | `experiment_prompt_constrained_plan` | planner prompt | **constrained** | free |

The comparison **B vs F** isolates the marginal contribution of the TC-Decoder
(execution-time tool-name constraints); **A vs B** measures the full method.

## Datasets

Each dataset is downloaded from its public source and converted into this repo's
existing contract — `evaluate.json` (tasks + ground-truth tool sequences) and
`tools.json` (tool schemas) — so the **same** agent loop, constrained decoder,
metrics and LLM judge run unchanged.

| Dataset | Source | License | What it tests |
|---------|--------|---------|---------------|
| **ToolAlpaca** | [tangqiaoyu/ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca) | Apache-2.0 | Simulated multi-step API use across ~20 toolsets; in-toolset distractors |
| **Seal-Tools** | [fairyshine/Seal-Tools](https://github.com/fairyshine/Seal-Tools) | see repo | Nested/multi-step calls over a large (~4k) tool pool; hard tool selection |
| **API-Bank** | [AlibabaResearch/DAMO-ConvAI/api-bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | MIT | Multi-turn dialogs flattened to tasks; auth-then-act tool chains |

Raw downloads (`data/_raw/`) and full converted datasets (`data/{toolalpaca,seal_tools,api_bank}/`)
are **git-ignored**. Only the tiny offline fixtures under **`data/sample/`** are committed.

### Conversion notes (faithful to the main benchmark)

- Ground-truth steps per task: optional leading `plan` → each domain tool
  (**required**, in ground-truth order) → `task_summary` (**required**) →
  `task_done` (**required**). This mirrors `experiments/evaluate.json` exactly, so
  scoring is apples-to-apples. `aggregate.py` additionally reports **domain-only
  recall** (excludes the `plan`/`task_summary`/`task_done` closure tools) to isolate
  the effect on real dataset-tool selection.
- Each task carries an `available_tools` list; the harness builds a **per-task tool
  registry** from it (GT tools ∪ sampled distractors for Seal-Tools/API-Bank; the full
  toolset for ToolAlpaca). This keeps SLM prompts bounded even for the ~4k Seal-Tools pool.
- Tools with no local implementation execute through a deterministic **simulated
  executor** (`generic_tools.GenericTool`). The rule-based tool-selection score only
  inspects *which* tools were called, so simulated return content does not affect it —
  it only keeps the loop progressing and gives the LLM judge readable traces.
- **Reference tool outputs** are injected where the source dataset provides them: each
  task may carry a `tool_responses` map (`tool_name → return string`) that the simulated
  executor echoes back instead of the generic acknowledgment. This gives *realistic*
  feedback for dependency-carrying tasks — e.g. API-Bank's `GetUserToken` returns its
  real `{"token": ...}` so a later `AddReminder` (free mode) can use it. API-Bank injects
  real `result.output`; Seal-Tools injects a response-field schema stub; ToolAlpaca has no
  reference outputs in its gold answers, so it keeps the generic executor.
- Parameter schemas: taken directly from the dataset (Seal-Tools) or **inferred** from
  reference argument dicts (API-Bank `param_dict`, ToolAlpaca `Action_Input`).

## Workflow

### 1. Prepare data (needs network, once)

```bash
python -m experiments.others.prepare_data --dataset all           # full sets
python -m experiments.others.prepare_data --dataset seal_tools --limit 200
python -m experiments.others.prepare_data --dataset sample --limit 5   # rebuild committed fixtures
```

### 2. Run the three groups

```bash
# Offline pipeline check — deterministic mock backend, no torch / no GPU.
python -m experiments.others.run_experiments --dataset sample --group all --backend mock

# Real run on a GPU box — reuses config.ALL_MODELS by default.
python -m experiments.others.run_experiments --dataset seal_tools --group all --backend hf

# Single model / quick smoke.
python -m experiments.others.run_experiments --dataset toolalpaca --group ptc \
    --backend hf --model Qwen/Qwen3-1.7B --quick
```

Flags: `--group {baseline,ptc,plan_wo_tc,all}`, `--backend {mock,hf}`, `--model`,
`--quantize`, `--max-tasks N`, `--quick` (5 tasks), `--no-resume`. Runs are
checkpointed per `(model, task)` and resume automatically.

Trajectories are saved to
`experiments/others/trajectories/<dataset>/exp_{a,b,f}_*/<model>.json` (git-ignored),
in the **same schema** the main suite uses — so `experiments/evaluate/llm_judge.py`
scores them without modification.

### 3. Aggregate

```bash
python -m experiments.others.aggregate --dataset all      # or: sample | seal_tools | ...
```

Prints, per dataset, the A/B/F **group-means** table and writes `report.md` +
`report.json` alongside the trajectories.

### 4. (Optional) LLM-as-Judge

```bash
export OPENAI_API_KEY=sk-...
python -m experiments.evaluate.llm_judge -i experiments/others/trajectories/seal_tools/exp_b_ptc_decoder/
```

## Metrics (per `(dataset, group, model)`)

| Metric | Meaning |
|--------|---------|
| `completion_rate` | fraction of tasks with completion score ≥ `COMPLETION_THRESHOLD` (0.8), full GT incl. closure tools |
| `avg_domain_recall` | mean recall over **domain** GT tools only (closure tools excluded) |
| `success_rate` | fraction of runs ending without crash/timeout |
| `avg_completion_score` | mean rule-based completion score |
| `plan_call_rate` | fraction of runs that invoked `plan` |
| `num_constrained` / `num_free` | constrained vs free decoding step counts |
| `avg_tokens` | mean total tokens/task |

## The `mock` backend

`torch` is not required to validate this suite. `--backend mock`
(`mock_backend.MockBackend`) is a deterministic, torch-free, oracle-ish stand-in
that walks `[GT domain tools] → task_summary → task_done`. It exercises every code
path (constrained plan, constrained/free execution, parsing, termination, save,
resume, aggregate) so the pipeline can be verified offline. **Its numbers are not a
scientific result** — real results come from `--backend hf` with the SLMs in
`experiments/config.py::ALL_MODELS`.

## Files

```
experiments/others/
├── README.md
├── common.py            # DatasetSpec, neutral prompts, planning subtask, 3 group runners, metrics
├── generic_tools.py     # GenericTool (simulated exec) + per-task registry builder
├── mock_backend.py      # deterministic torch-free backend for offline verification
├── prepare_data.py      # download raw + convert; build committed sample fixtures
├── run_experiments.py   # CLI: --dataset/--group/--backend/--model/--max-tasks/--quick
├── aggregate.py         # trajectories → A/B/F comparison table (report.md/json)
├── converters/
│   ├── __init__.py      # shared: schema inference, GT-step builder, distractor sampling, writer
│   ├── toolalpaca.py
│   ├── seal_tools.py
│   └── api_bank.py
└── data/
    └── sample/          # committed tiny fixtures (offline smoke tests)
```

Reuses (unchanged): `agentic/agent.py` (agent loop), `agentic/tools.py`
(`ToolRegistry`), `experiments/trajectory_utils.py` (scoring, save, resume),
`experiments/config.py` (models, hyperparameters), and the TC-Decoder in
`constrained_decoding/`.
