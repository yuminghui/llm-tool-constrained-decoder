"""
Context Token Experiment
=========================

Two sub-experiments:

**Experiment 1** — Cross-model token comparison
    Run the same benchmark tasks across different SLMs (small language models),
    measuring total tokens consumed to complete each task.

**Experiment 2** — Constrained-plan vs. separate-plan token efficiency
    Compare two plan-then-act strategies on token consumption and completion rate:

    A. **Constrained inline plan** — the agent forces ``plan`` as step 1 via the
       constraint decoder, then continues executing.  The plan lives inside the
       same conversation.

    B. **Separate plan + agent** — a *planning-only* sub-task generates the plan
       in its own context; then a *second* agent session executes it with the
       plan pre-seeded.  Tokens = plan-subtask tokens + execution tokens.

Usage::

    python -m experiments.experiment_context_tokens --exp 1
    python -m experiments.experiment_context_tokens --exp 2
    python -m experiments.experiment_context_tokens --exp all --quick
"""

from __future__ import annotations

import json
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.llm_backend import LLMBackend
from agentic.tools import load_tools_from_json
from agentic.agent import Agent, AgentConfig, AgentResult, AgentStep

from experiments.config import (
    TOOLS_JSON_PATH,
    EVALUATE_JSON_PATH,
    ALL_MODELS,
    QUICK_MODELS,
    EXP1_MODELS,
    EXP1_MAX_TASKS,
    EXP2_MODEL,
    EXP2_MAX_TASKS,
    TEMPERATURE,
    MAX_TURNS,
    PLAN_MAX_NEW_TOKENS,
    FREE_MAX_NEW_TOKENS,
    CONSTRAINED_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT,
    OPTIONAL_STEP_WEIGHT,
    TASK_DONE_WEIGHT,
    COMPLETION_THRESHOLD,
)


# ==========================================================================
# Task completion evaluator
# ==========================================================================

def evaluate_completion(
    agent_steps: List[AgentStep],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Score an agent trajectory against ground-truth expected steps.

    Args:
        agent_steps: The steps recorded during the agent run.
        ground_truth: Dict with ``steps`` key — a list of
            ``{"action": <tool_name>, "required": bool}`` entries.

    Returns:
        Dict with scoring breakdown.
    """
    called_tools = [s.tool_name for s in agent_steps if s.tool_name]

    gt_steps = ground_truth.get("steps", [])
    required_actions = [s["action"] for s in gt_steps if s.get("required")]
    optional_actions = [s["action"] for s in gt_steps if not s.get("required")]

    required_called = sum(1 for a in required_actions if a in called_tools)
    optional_called = sum(1 for a in optional_actions if a in called_tools)

    req_score = (required_called / len(required_actions)) if required_actions else 1.0
    opt_bonus = (optional_called / len(optional_actions) * OPTIONAL_STEP_WEIGHT) if optional_actions else 0.0

    task_done_bonus = TASK_DONE_WEIGHT if "task_done" in called_tools else 0.0

    total_score = min(req_score + opt_bonus + task_done_bonus, 1.0)

    return {
        "required_called": required_called,
        "required_total": len(required_actions),
        "optional_called": optional_called,
        "optional_total": len(optional_actions),
        "task_done_called": "task_done" in called_tools,
        "score": round(total_score, 3),
        "completed": total_score >= COMPLETION_THRESHOLD,
    }


# ==========================================================================
# Token counting helper
# ==========================================================================

def count_tokens(text: str, backend: LLMBackend) -> int:
    """Count the number of tokens in *text* using the backend's tokenizer."""
    return len(backend.tokenizer.encode(text))


# ==========================================================================
# Plan-only sub-task (for Experiment 2, Approach B)
# ==========================================================================

PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. Given a user's request, create a detailed step-by-step "
    "plan using the 'plan' tool. Break down the task into clear, executable steps, "
    "specifying which tools to use for each step. Do NOT execute any tools — only plan. "
    "Respond in Chinese."
)

EXECUTOR_SYSTEM_PROMPT = (
    "You are a task executor. A detailed plan has already been created. "
    "Your job is to execute the plan step by step using the available tools. "
    "Do NOT call the 'plan' tool — the plan is already done. "
    "Execute tools one at a time. When finished, use task_summary and task_done. "
    "Respond in Chinese."
)


def run_planning_subtask(
    backend: LLMBackend,
    tools_registry,
    user_query: str,
) -> Dict[str, Any]:
    """Run a planning-only sub-task.

    Returns:
        Dict with keys: tool_name, tool_args, generated_text, tool_result, tokens.
    """
    tool_defs = tools_registry.get_definitions()
    plan_schema = tools_registry.get_schema("plan")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    prompt = backend.build_prompt(messages, tool_defs)

    decoder_result = backend.generate_constrained(
        prompt=prompt,
        tool_name="plan",
        args_schema=plan_schema,
        max_new_tokens=PLAN_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    # Count tokens: prompt + generated
    prompt_tokens = len(backend.tokenizer.encode(prompt))
    generated_tokens = len(decoder_result.token_ids)
    total_tokens = prompt_tokens + generated_tokens

    # Execute plan to get tool_result
    plan_args = decoder_result.tool_call.get("arguments", {})
    tool_result = None
    if "_parse_error" not in decoder_result.tool_call:
        tool_result = tools_registry.execute("plan", plan_args)

    return {
        "tool_name": "plan",
        "tool_args": plan_args,
        "generated_text": decoder_result.text,
        "tool_result": tool_result,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
    }


# ==========================================================================
# Data classes for experiment results
# ==========================================================================

@dataclass
class TaskResult:
    """Result of a single task run."""
    task_index: int
    question: str
    approach: str           # "inline_constrained" | "separate_plan"
    model_id: str
    success: bool
    completion: Dict[str, Any]
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    num_steps: int
    total_time: float
    error: Optional[str] = None


@dataclass
class ExperimentReport:
    """Aggregated experiment results."""
    experiment_name: str
    results: List[TaskResult] = field(default_factory=list)

    def avg_tokens(self) -> float:
        return sum(r.total_tokens for r in self.results) / max(len(self.results), 1)

    def avg_generated_tokens(self) -> float:
        return sum(r.generated_tokens for r in self.results) / max(len(self.results), 1)

    def completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.completion["completed"]) / len(self.results)

    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)


# ==========================================================================
# Experiment 1 — Cross-model token comparison
# ==========================================================================

def run_experiment_1(
    models: List[str],
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> Dict[str, ExperimentReport]:
    """Compare token consumption across different SLMs on the same benchmark.

    Returns:
        Dict mapping model_id → ExperimentReport.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 — Cross-Model Token Comparison")
    print("=" * 70)

    tasks = _load_tasks(task_indices, max_tasks)
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    reports: Dict[str, ExperimentReport] = {}

    for model_id in models:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_id}")
        print(f"{'─' * 60}")

        backend = LLMBackend(model_id)
        agent_config = AgentConfig(
            max_turns=MAX_TURNS,
            temperature=TEMPERATURE,
            use_constrained_decoder=True,
            verbose=False,
        )
        report = ExperimentReport(experiment_name=f"Exp1 — {model_id}")

        for i, task in enumerate(tasks):
            question = task["question"]
            gt = task.get("trajectory_ground_truth", {})

            print(f"  [{i+1}/{len(tasks)}] {question[:70]}...", end=" ", flush=True)

            agent = Agent(backend, tools_registry, agent_config)

            # Measure prompt tokens
            messages = [
                {"role": "system", "content": agent._system_prompt},
                {"role": "user", "content": question},
            ]
            tool_defs = tools_registry.get_definitions()
            prompt = backend.build_prompt(messages, tool_defs)
            prompt_tokens = len(backend.tokenizer.encode(prompt))

            t0 = time.time()
            agent_result = agent.run(question)
            elapsed = time.time() - t0

            # Count generated tokens
            generated_tokens = 0
            for step in agent_result.steps:
                generated_tokens += len(backend.tokenizer.encode(step.generated_text))

            completion = evaluate_completion(agent_result.steps, gt)

            tr = TaskResult(
                task_index=i,
                question=question,
                approach="inline_constrained",
                model_id=model_id,
                success=agent_result.success,
                completion=completion,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
                num_steps=len(agent_result.steps),
                total_time=elapsed,
                error=agent_result.error,
            )
            report.results.append(tr)

            status = "OK" if completion["completed"] else "NOK"
            print(f"{status} | tokens={tr.total_tokens} | score={completion['score']:.2f} | "
                  f"{completion['required_called']}/{completion['required_total']} req")

        reports[model_id] = report
        _print_model_summary(model_id, report)

    return reports


# ==========================================================================
# Experiment 2 — Constrained plan vs separate plan
# ==========================================================================

def run_experiment_2(
    model_id: str,
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> Dict[str, ExperimentReport]:
    """Compare inline constrained plan (A) vs separate plan + agent (B).

    Returns:
        Dict with keys "inline_constrained" and "separate_plan" → ExperimentReport.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 — Constrained Plan vs. Separate Plan")
    print(f"  Model: {model_id}")
    print("=" * 70)

    tasks = _load_tasks(task_indices, max_tasks)
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    backend = LLMBackend(model_id)

    report_a = ExperimentReport(experiment_name="Exp2A — Inline Constrained Plan")
    report_b = ExperimentReport(experiment_name="Exp2B — Separate Plan + Agent")

    for i, task in enumerate(tasks):
        question = task["question"]
        gt = task.get("trajectory_ground_truth", {})

        print(f"\n  [{i+1}/{len(tasks)}] {question[:80]}")

        # ---- Approach A: Inline constrained plan ----
        print("    A (inline constrained plan)...", end=" ", flush=True)

        agent_config = AgentConfig(
            max_turns=MAX_TURNS,
            temperature=TEMPERATURE,
            use_constrained_decoder=True,
            verbose=False,
        )
        agent = Agent(backend, tools_registry, agent_config)

        messages = [
            {"role": "system", "content": agent._system_prompt},
            {"role": "user", "content": question},
        ]
        tool_defs = tools_registry.get_definitions()
        prompt = backend.build_prompt(messages, tool_defs)
        prompt_tokens = len(backend.tokenizer.encode(prompt))

        t0 = time.time()
        result_a = agent.run(question)
        elapsed_a = time.time() - t0

        gen_tokens_a = sum(
            len(backend.tokenizer.encode(s.generated_text))
            for s in result_a.steps
        )
        completion_a = evaluate_completion(result_a.steps, gt)

        tr_a = TaskResult(
            task_index=i,
            question=question,
            approach="inline_constrained",
            model_id=model_id,
            success=result_a.success,
            completion=completion_a,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens_a,
            total_tokens=prompt_tokens + gen_tokens_a,
            num_steps=len(result_a.steps),
            total_time=elapsed_a,
            error=result_a.error,
        )
        report_a.results.append(tr_a)

        print(f"tokens={tr_a.total_tokens} | score={completion_a['score']:.2f} | "
              f"{completion_a['required_called']}/{completion_a['required_total']} req")

        # ---- Approach B: Separate plan + agent ----
        print("    B (separate plan + agent)...", end=" ", flush=True)

        # Phase 1: Planning sub-task
        plan_result = run_planning_subtask(backend, tools_registry, question)

        # Phase 2: Execution with pre-seeded plan
        executor_config = AgentConfig(
            max_turns=MAX_TURNS,
            temperature=TEMPERATURE,
            use_constrained_decoder=True,
            verbose=False,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
        )
        executor = Agent(backend, tools_registry, executor_config)

        # Build prompt tokens for Phase 2
        exec_messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        exec_prompt = backend.build_prompt(exec_messages, tool_defs)
        exec_prompt_tokens = len(backend.tokenizer.encode(exec_prompt))

        t0 = time.time()
        result_b = executor.run(question, pre_seeded_plan=plan_result)
        elapsed_b = time.time() - t0

        gen_tokens_b = sum(
            len(backend.tokenizer.encode(s.generated_text))
            for s in result_b.steps
        )

        # Total tokens = Phase 1 (plan) + Phase 2 (execution)
        total_tokens_b = (
            plan_result["total_tokens"] +         # Phase 1: prompt + generated
            exec_prompt_tokens + gen_tokens_b      # Phase 2: prompt + generated
        )
        generated_tokens_b = plan_result["generated_tokens"] + gen_tokens_b

        completion_b = evaluate_completion(result_b.steps, gt)

        tr_b = TaskResult(
            task_index=i,
            question=question,
            approach="separate_plan",
            model_id=model_id,
            success=result_b.success,
            completion=completion_b,
            prompt_tokens=plan_result["prompt_tokens"] + exec_prompt_tokens,
            generated_tokens=generated_tokens_b,
            total_tokens=total_tokens_b,
            num_steps=len(result_b.steps),
            total_time=elapsed_b,
            error=result_b.error,
        )
        report_b.results.append(tr_b)

        print(f"tokens={tr_b.total_tokens} (plan={plan_result['total_tokens']} + "
              f"exec={exec_prompt_tokens + gen_tokens_b}) | "
              f"score={completion_b['score']:.2f} | "
              f"{completion_b['required_called']}/{completion_b['required_total']} req")

        # Quick comparison on this task
        diff = tr_a.total_tokens - tr_b.total_tokens
        sign = "less" if diff < 0 else "more"
        print(f"      A vs B: A uses {abs(diff)} tokens {sign} than B")

    return {"inline_constrained": report_a, "separate_plan": report_b}


# ==========================================================================
# Helpers
# ==========================================================================

def _load_tasks(
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> List[Dict[str, Any]]:
    """Load tasks from evaluate.json, optionally filtered by index or count."""
    with open(EVALUATE_JSON_PATH, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if task_indices is not None:
        all_tasks = [all_tasks[i] for i in task_indices if i < len(all_tasks)]

    if max_tasks and max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]

    return all_tasks


def _print_model_summary(model_id: str, report: ExperimentReport) -> None:
    """Print a per-model summary line."""
    print(f"\n  [{model_id}] avg_tokens={report.avg_tokens():.0f} | "
          f"avg_gen={report.avg_generated_tokens():.0f} | "
          f"completion={report.completion_rate():.1%} | "
          f"success={report.success_rate():.1%}")


# ==========================================================================
# Report printing
# ==========================================================================

def print_exp1_report(reports: Dict[str, ExperimentReport]) -> None:
    """Print final report for Experiment 1."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 — FINAL REPORT")
    print("=" * 70)

    print(f"\n{'Model':<40s} {'Avg Tokens':>10s} {'Avg Gen':>8s} "
          f"{'Completion':>11s} {'Success':>8s}")
    print("-" * 80)

    for model_id, report in reports.items():
        print(f"{model_id:<40s} {report.avg_tokens():>10.0f} "
              f"{report.avg_generated_tokens():>8.0f} "
              f"{report.completion_rate():>10.1%} "
              f"{report.success_rate():>8.1%}")

    print("-" * 80)

    if reports:
        best = min(reports.items(), key=lambda x: x[1].avg_tokens())
        print(f"\n  Most token-efficient: {best[0]} ({best[1].avg_tokens():.0f} avg tokens)")


def print_exp2_report(reports: Dict[str, ExperimentReport]) -> None:
    """Print final report for Experiment 2."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 — FINAL REPORT")
    print("=" * 70)

    ra = reports["inline_constrained"]
    rb = reports["separate_plan"]

    print(f"\n{'Metric':<35s} {'A-Inline':>15s} {'B-Separate':>15s} {'Delta':>15s}")
    print("-" * 82)

    print(f"{'Avg Total Tokens':<35s} {ra.avg_tokens():>15.0f} {rb.avg_tokens():>15.0f} "
          f"{ra.avg_tokens() - rb.avg_tokens():>+15.0f}")

    print(f"{'Avg Generated Tokens':<35s} {ra.avg_generated_tokens():>15.0f} "
          f"{rb.avg_generated_tokens():>15.0f} "
          f"{ra.avg_generated_tokens() - rb.avg_generated_tokens():>+15.0f}")

    print(f"{'Completion Rate':<35s} {ra.completion_rate():>15.1%} "
          f"{rb.completion_rate():>15.1%} "
          f"{ra.completion_rate() - rb.completion_rate():>+15.1%}")

    print(f"{'Success Rate':<35s} {ra.success_rate():>15.1%} "
          f"{rb.success_rate():>15.1%} "
          f"{ra.success_rate() - rb.success_rate():>+15.1%}")

    print("-" * 82)

    # Win/loss/tie by total tokens
    wins_a = 0
    wins_b = 0
    ties = 0
    for ta, tb in zip(ra.results, rb.results):
        if ta.total_tokens < tb.total_tokens:
            wins_a += 1
        elif tb.total_tokens < ta.total_tokens:
            wins_b += 1
        else:
            ties += 1

    print(f"\n  Token efficiency: A wins {wins_a}, B wins {wins_b}, ties {ties} "
          f"(out of {len(ra.results)} tasks)")

    if wins_a > wins_b:
        print(f"  Inline constrained plan uses fewer tokens on more tasks.")
    elif wins_b > wins_a:
        print(f"  Separate plan + agent uses fewer tokens on more tasks.")
    else:
        print(f"  Both approaches are tied on token efficiency.")


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Context Token Experiments"
    )
    parser.add_argument(
        "--exp", choices=["1", "2", "all"], default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use quick-test models (fewer models, fewer tasks)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=0,
        help="Max tasks per experiment (0 = use config default)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model for Experiment 2",
    )
    args = parser.parse_args()

    # --- Experiment 1 ---
    if args.exp in ("1", "all"):
        models = QUICK_MODELS if args.quick else EXP1_MODELS
        max_t = args.max_tasks or EXP1_MAX_TASKS
        reports = run_experiment_1(models, max_tasks=max_t)
        print_exp1_report(reports)

    # --- Experiment 2 ---
    if args.exp in ("2", "all"):
        model = args.model or EXP2_MODEL
        max_t = args.max_tasks or EXP2_MAX_TASKS
        reports = run_experiment_2(model, max_tasks=max_t)
        print_exp2_report(reports)


if __name__ == "__main__":
    main()
