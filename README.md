# PTC-Decoder: Plan-Tool Constrained Decoder for SLM Agents

A training-free, plug-and-play decoder framework that enforces structured planning and tool invocation for small language models (SLMs, ≤2B) on offline resource-constrained edge devices.

## Overview

PTC-Decoder addresses the fundamental challenge of deploying SLM-based agents on devices such as remote sensing satellites: SLMs lack the reasoning capacity to reliably execute multi-step tasks. Our framework couples two components:

1. **Plan-to-Act** — elevates planning to an atomic tool and mandates its invocation at the first inference step.
2. **TC-Decoder** — a deterministic finite automaton (DFA) that applies token-level hard constraints on tool names during decoding, while preserving full model freedom over parameter generation.

The framework is training-free, requires no prompt modification, and operates as a hot-pluggable module within existing agent systems.

## Project Structure

```
├── README.md
├── requirements.txt
├── tools.json                            # Tool definitions for the agent
├── main.py                               # Entry point for interactive agent
├── agent_tools.py                        # Tool implementations
├── test_restrict_decode.py               # Unit test for constrained decoder
│
├── agentic/                              # Agent runtime
│   ├── __init__.py
│   ├── agent.py                          # Agent loop (free + constrained)
│   ├── llm_backend.py                    # HuggingFace model wrapper + tool-call parser
│   ├── tools.py                          # Tool registry and execution
│   └── experiment.py                     # Experiment runner utilities
│
├── constrained_decoding/                 # TC-Decoder implementation
│   ├── README.md
│   ├── __init__.py
│   ├── decoder.py                        # ToolConstrainedDecoder: DFA-based logit masking
│   ├── char_fsm.py                       # Character-level finite state machine
│   ├── json_schema_to_fsm.py             # JSON Schema → FSM compiler
│   ├── logits_processor.py               # HuggingFace LogitsProcessor integration
│   └── token_index.py                    # Token-to-index mapping for constraint ops
│
└── experiments/                          # Experiment suite (see experiments/README.md)
    ├── README.md
    ├── config.py                         # Centralized experiment configuration
    ├── prompts.py                        # Shared system / user prompt templates
    ├── evaluate.json                     # Benchmark (200 annotated remote sensing tasks)
    ├── trajectory_utils.py               # Trajectory I/O, completion scoring, error classification
    ├── experiment_baseline.py            # Exp A: Baseline (free generation)
    ├── experiment_separate_plan.py       # Exp B: PTC-Decoder (ours)
    ├── experiment_prompt_plan.py         # Exp E: Prompt-only plan enforcement
    ├── experiment_prompt_constrained_plan.py  # Exp F: Plan w/o TC-Decoder (ablation)
    └── evaluate/
        ├── __init__.py
        └── llm_judge.py                  # LLM-as-Judge trajectory evaluator
```

## Installation

```bash
# Python ≥ 3.10, CUDA ≥ 12.0 recommended
pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `bitsandbytes` (optional, for 4-bit quantization), `openai` (for LLM judge).

## Quick Start

### Interactive Agent (single query)

```bash
python main.py --model Qwen/Qwen3-1.7B
```

### Running Experiments

All experiments support `--quick` (5-task smoke test) and `--max-tasks N` (limit task count). Use `--model` to test a single model instead of the full model list.

```bash
# Experiment A: Baseline (free generation)
python -m experiments.experiment_baseline

# Experiment B: PTC-Decoder (ours)
python -m experiments.experiment_separate_plan

# Experiment E: Prompt-Only Plan
python -m experiments.experiment_prompt_plan

# Experiment F: Plan w/o TC-Decoder (ablation)
python -m experiments.experiment_prompt_constrained_plan

# Quick smoke test (5 tasks)
python -m experiments.experiment_baseline --quick --max-tasks 5

# Single model with 4-bit quantization
python -m experiments.experiment_separate_plan --model google/gemma-4-E2B-it --quantize
```

### LLM-as-Judge Evaluation

After running experiments, evaluate trajectories with an external LLM:

```bash
export OPENAI_API_KEY=sk-...

# Evaluate all trajectory files in a directory
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_b_separate_plan/

# Custom judge model and endpoint
python -m experiments.evaluate.llm_judge \
  -i experiments/trajectories/exp_b_separate_plan/ \
  -o results/judge_scores/ \
  --model deepseek-v4-flash \
  --base-url https://your-proxy/v1
```

### Unit Tests

```bash
python test_restrict_decode.py
```

## Models

Tested with 7 mainstream SLMs (≤2B parameters, as of May 2026):

| Model | Parameters | Quantization |
|-------|:--:|:--:|
| Qwen/Qwen3-1.7B | 1.7B | — |
| Qwen/Qwen3.5-2B | 2.0B | — |
| Qwen/Qwen3.5-0.8B | 0.8B | — |
| Qwen/Qwen3-0.6B | 0.6B | — |
| google/gemma-4-E2B-it | 2.0B (MoE) | 4-bit |
| google/gemma-3-1b-it | 1.0B | — |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | — |

Edit `experiments/config.py` → `ALL_MODELS` to add or remove models.

## Benchmark

`experiments/evaluate.json` contains 200 real remote sensing satellite tasks, manually annotated with:

- Ground-truth expected tool-call sequences (required / optional step labels)
- Expected final answers
- Task metadata

This is, to our knowledge, the first open benchmark for remote sensing Agent evaluation.

## Key Results

On the 7-model benchmark (200 tasks, LLM-as-Judge + rule-based metrics):

| Metric | Baseline → PTC-Decoder (Ours) |
|--------|:-----------------------------:|
| Overall (Ov.) LLM-Judge | +1.21 mean gain |
| Recall (tool coverage) | +0.126 mean gain |
| F1 (tool coverage) | +0.096 mean gain (peak 0.359) |
| Weakest-model relative gain | +350% |

**Ablation** (PTC-Decoder vs. Plan w/o TC-Decoder):

| Metric | Gain |
|--------|:----:|
| Overall (Ov.) | +0.99 |
| Result Accuracy (R.A.) | +1.42 |
| Flow Reasonableness (F.R.) | +1.00 |
| Recall (tool coverage) | +0.146 |
| F1 (tool coverage) | +0.110 |

TC-Decoder is the decisive component: removing it reduces F1 by 0.110 on average and degrades all quality metrics, yet offers no efficiency advantage.

## License

- **Code** ([`LICENSE`](LICENSE)): MIT License — freely usable, modifiable, and distributable for any purpose.
- **Data / Benchmark** (`experiments/evaluate.json` and other data files): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to share and adapt with attribution.
