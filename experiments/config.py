"""
Experiment configuration — centralizes all tunable parameters.

Add new models, adjust hyperparameters, or change paths here.
Individual experiment files import from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Project paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOOLS_JSON_PATH = os.path.join(PROJECT_ROOT, "tools.json")
EVALUATE_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "evaluate.json")
TRAJECTORY_DIR = os.path.join(PROJECT_ROOT, "experiments", "trajectories")
EXP3_TRAJECTORY_DIR = TRAJECTORY_DIR  # backward-compat alias

# ---------------------------------------------------------------------------
# Models — small language models (≤ 2B) for local inference
# ---------------------------------------------------------------------------

# fmt: off
ALL_MODELS = [
    # ---- Qwen (Alibaba) ----
    "Qwen/Qwen2.5-0.5B-Instruct",              # 0.5B  — Apache 2.0
    "Qwen/Qwen2.5-1.5B-Instruct",              # 1.5B  — Apache 2.0

    # ---- Google Gemma ----
    "google/gemma-2-2b-it",                    # 2.0B  — Gemma license, needs HF auth
    # "google/gemma-3-1b-it",                  # 1.0B  — uncomment when released on HF

    # ---- Meta Llama ----
    "meta-llama/Llama-3.2-1B-Instruct",        # 1.0B  — Llama Community License, needs HF auth

    # ---- Microsoft Phi ----
    "microsoft/phi-1_5",                       # 1.3B  — MIT

    # ---- HuggingFace SmolLM2 ----
    "HuggingFaceTB/SmolLM2-135M-Instruct",     # 0.14B — Apache 2.0
    "HuggingFaceTB/SmolLM2-360M-Instruct",     # 0.36B — Apache 2.0
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",     # 1.7B  — Apache 2.0

    # ---- DeepSeek (distilled) ----
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # 1.5B — MIT

    # ---- Stability AI ----
    "stabilityai/stablelm-2-1.6b-chat",        # 1.6B  — CC BY-NC-SA-4.0
]
# fmt: on

# Quick-test subset (fastest models for rapid iteration)
QUICK_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]

# ---------------------------------------------------------------------------
# Experiment 1 — cross-model token comparison
# ---------------------------------------------------------------------------

EXP1_MODELS = ALL_MODELS          # models to compare
EXP1_TASK_INDICES: Optional[List[int]] = None   # None = all tasks; or [0, 1, 2, ...]
EXP1_MAX_TASKS: int = 10          # cap number of tasks per model (0 = unlimited)
# Output dir for per-model trajectory files
EXP1_OUTPUT_DIR = os.path.join(TRAJECTORY_DIR, "exp1_cross_model")

# ---------------------------------------------------------------------------
# Experiment 2 — constrained plan vs separate plan
# ---------------------------------------------------------------------------

EXP2_MODELS = [ALL_MODELS[0]]     # models to compare (use [ALL_MODELS[0]] for single)
EXP2_TASK_INDICES: Optional[List[int]] = None
EXP2_MAX_TASKS: int = 0           # 0 = unlimited
EXP2_OUTPUT_DIR = os.path.join(TRAJECTORY_DIR, "exp2")

# ---------------------------------------------------------------------------
# Experiment 3 — plan-first vs no-plan (LLM-as-Judge)
# ---------------------------------------------------------------------------

EXP3_MODELS = [ALL_MODELS[0]]
EXP3_TASK_INDICES: Optional[List[int]] = None
EXP3_MAX_TASKS: int = 0           # 0 = unlimited
EXP3_OUTPUT_DIR = os.path.join(TRAJECTORY_DIR, "exp3")

# ---------------------------------------------------------------------------
# Experiment 4 — ablation: constraint decoder for plan step
# ---------------------------------------------------------------------------

EXP4_MODELS = [ALL_MODELS[0]]
EXP4_TASK_INDICES: Optional[List[int]] = None
EXP4_MAX_TASKS: int = 0           # 0 = unlimited
EXP4_OUTPUT_DIR = os.path.join(TRAJECTORY_DIR, "exp4")

# ---------------------------------------------------------------------------
# Agent hyperparameters (shared across experiments)
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
MAX_TURNS = 10

# Token budget for generation calls
PLAN_MAX_NEW_TOKENS = 256     # constrained plan generation
FREE_MAX_NEW_TOKENS = 512     # free generation steps
CONSTRAINED_MAX_NEW_TOKENS = 256  # constrained tool calls (non-plan)

# ---------------------------------------------------------------------------
# Task completion evaluation
# ---------------------------------------------------------------------------

# Weights for trajectory scoring
REQUIRED_STEP_WEIGHT = 1.0    # full credit for each required step
OPTIONAL_STEP_WEIGHT = 0.5    # bonus (clamped) for optional steps
TASK_DONE_WEIGHT = 0.2        # bonus for calling task_done

# Completion threshold: score >= this is considered "completed"
COMPLETION_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Dataclass form (optional — for programmatic use)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All experiment parameters in one object."""
    # Models
    models: List[str] = field(default_factory=lambda: ALL_MODELS)

    # Paths
    tools_json: str = TOOLS_JSON_PATH
    evaluate_json: str = EVALUATE_JSON_PATH

    # Agent
    temperature: float = TEMPERATURE
    max_turns: int = MAX_TURNS
    plan_max_new_tokens: int = PLAN_MAX_NEW_TOKENS
    free_max_new_tokens: int = FREE_MAX_NEW_TOKENS
    constrained_max_new_tokens: int = CONSTRAINED_MAX_NEW_TOKENS

    # Evaluation
    required_step_weight: float = REQUIRED_STEP_WEIGHT
    optional_step_weight: float = OPTIONAL_STEP_WEIGHT
    task_done_weight: float = TASK_DONE_WEIGHT
    completion_threshold: float = COMPLETION_THRESHOLD
