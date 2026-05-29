"""
Plan-first vs No-plan Experiment
=================================

Tests whether forcing a ``plan`` step before action improves or degrades
SLM agent performance.

Two modes are compared head-to-head on the same benchmark tasks:

**P+  (with plan)** — Plan-to-Act
    Step 0: force ``plan`` via constraint decoder.
    Step 1+: execute planned tools (constrained), then final answer.

**P-  (without plan)** — Direct execution
    No planning step.  The model sees the user query and decides which
    tools to call directly, one turn at a time.

Both modes save their execution trajectories to separate JSON files for
later LLM-as-Judge evaluation::

    experiments/trajectories/plan_yes.json   ← P+ traces
    experiments/trajectories/plan_no.json    ← P- traces

Usage::

    python -m experiments.experiment_plan
    python -m experiments.experiment_plan --quick --max-tasks 5
"""

from __future__ import annotations

import json
import sys
import os
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.llm_backend import LLMBackend
from agentic.tools import load_tools_from_json
from agentic.agent import (
    Agent, AgentConfig, AgentResult, AgentStep,
)

from experiments.config import (
    TOOLS_JSON_PATH,
    EVALUATE_JSON_PATH,
    EXP3_MODEL,
    EXP3_TASK_INDICES,
    EXP3_MAX_TASKS,
    EXP3_TRAJECTORY_DIR,
    EXP3_TRAJECTORY_WITH_PLAN,
    EXP3_TRAJECTORY_WITHOUT_PLAN,
    TEMPERATURE,
    MAX_TURNS,
    PLAN_MAX_NEW_TOKENS,
    FREE_MAX_NEW_TOKENS,
)
from experiments.trajectory_utils import save_trajectories_batch


# ==========================================================================
# System prompts
# ==========================================================================

SYSTEM_PROMPT_WITH_PLAN = (
    "You are a helpful assistant with access to external tools. "
    "Always call the 'plan' tool first to create a step-by-step plan "
    "before taking any action. Execute tools one at a time. "
    "When you have gathered all needed information, use task_summary "
    "to summarize and task_done to finish. Respond in Chinese."
)

SYSTEM_PROMPT_WITHOUT_PLAN = (
    "You are a helpful assistant with access to external tools. "
    "Execute tools directly"
    "When the user asks you to do something, call the appropriate "
    "tools immediately to get the job done. "
    "When finished, use task_summary to summarize results and "
    "task_done to signal completion. Respond in Chinese."
)


# ==========================================================================
# Mode P- (without plan): direct execution agent
# ==========================================================================

def run_agent_no_plan(
    backend: LLMBackend,
    tools_registry,
    user_query: str,
    max_turns: int = MAX_TURNS,
    temperature: float = TEMPERATURE,
    verbose: bool = False,
) -> AgentResult:
    """Run agent in *no-plan* mode — go straight to tool execution.

    The model sees the query and decides what tools to call directly,
    without an explicit planning phase.
    """
    start_time = time.time()
    steps: List[AgentStep] = []
    total_tokens = 0

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_PLAN},
        {"role": "user", "content": user_query},
    ]
    tool_defs = tools_registry.get_definitions()

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"    [P- turn {turn}] generating...")

        prompt = backend.build_prompt(messages, tool_defs)
        raw_output = backend.generate_free(
            prompt=prompt,
            max_new_tokens=FREE_MAX_NEW_TOKENS,
            temperature=temperature,
        )
        total_tokens += len(backend.tokenizer.encode(raw_output))

        if verbose:
            print(f"      output: {raw_output[:150]}...")

        # Try to parse a tool call
        tool_call = backend.parse_tool_call(raw_output)

        if tool_call and "name" in tool_call:
            t_name = tool_call.get("name", "")
            t_args = tool_call.get("arguments", {})

            t0 = time.time()
            step = AgentStep(
                step_index=turn,
                tool_name=t_name,
                tool_args=t_args,
                tool_result=None,
                generated_text=raw_output,
                is_constrained=False,
                elapsed=0.0,
            )

            tool_entry = tools_registry.get(t_name)
            if tool_entry:
                step.tool_result = tools_registry.execute(t_name, t_args)
                if verbose:
                    print(f"      -> {t_name} executed")
            else:
                # Unknown tool — treat as final answer
                if verbose:
                    print(f"      -> unknown tool '{t_name}', stopping")
                step.elapsed = time.time() - t0
                steps.append(step)
                return AgentResult(
                    user_query=user_query,
                    steps=steps,
                    final_answer=raw_output,
                    total_time=time.time() - start_time,
                    total_tokens=total_tokens,
                    success=True,
                )

            step.elapsed = time.time() - t0
            steps.append(step)

            # Append tool call + result to conversation
            tool_call_json = json.dumps(
                {"name": t_name, "arguments": t_args},
                ensure_ascii=False,
            )
            assistant_msg = (
                f"{backend.tool_call_prefix}{tool_call_json}"
                f"{backend.tool_call_suffix}"
            )
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "tool", "content": step.tool_result or ""})

            # If task_done was called, stop
            if t_name == "task_done":
                return AgentResult(
                    user_query=user_query,
                    steps=steps,
                    final_answer=step.tool_result,
                    total_time=time.time() - start_time,
                    total_tokens=total_tokens,
                    success=True,
                )
        else:
            # No tool call — this is the final answer
            if verbose:
                print(f"      -> final answer (no tool call)")

            steps.append(AgentStep(
                step_index=turn,
                tool_name=None,
                tool_args=None,
                tool_result=None,
                generated_text=raw_output,
                is_constrained=False,
                elapsed=0.0,
            ))

            return AgentResult(
                user_query=user_query,
                steps=steps,
                final_answer=raw_output,
                total_time=time.time() - start_time,
                total_tokens=total_tokens,
                success=True,
            )

    # Max turns reached
    return AgentResult(
        user_query=user_query,
        steps=steps,
        final_answer=None,
        total_time=time.time() - start_time,
        total_tokens=total_tokens,
        success=False,
        error="Max turns reached",
    )


# ==========================================================================
# Main experiment
# ==========================================================================

def run_plan_experiment(
    model_id: str = EXP3_MODEL,
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> Dict[str, str]:
    """Run plan vs no-plan experiment on evaluate.json benchmark.

    Returns:
        Dict with paths to the two trajectory files:
        ``{"with_plan": path, "without_plan": path}``
    """
    # Load tasks
    with open(EVALUATE_JSON_PATH, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if task_indices is not None:
        all_tasks = [all_tasks[i] for i in task_indices if i < len(all_tasks)]
    if max_tasks and max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]

    print("\n" + "=" * 70)
    print("  PLAN EXPERIMENT — Plan-first vs. No-plan")
    print(f"  Model:  {model_id}")
    print(f"  Tasks:  {len(all_tasks)}")
    print("=" * 70)

    # Shared infrastructure
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    backend = LLMBackend(model_id)

    results_with_plan: List[AgentResult] = []
    results_without_plan: List[AgentResult] = []
    task_ids: List[str] = []

    # Agent config for P+ (with plan)
    agent_config = AgentConfig(
        max_turns=MAX_TURNS,
        temperature=TEMPERATURE,
        use_constrained_decoder=True,
        verbose=False,
        system_prompt=SYSTEM_PROMPT_WITH_PLAN,
    )

    for i, task in enumerate(all_tasks):
        question = task["question"]
        task_id = f"task_{i:03d}"
        task_ids.append(task_id)

        print(f"\n[{i+1}/{len(all_tasks)}] {question[:80]}")

        # ---- P+: with plan ----
        print("  P+ (with plan)...", end=" ", flush=True)
        t0 = time.time()
        agent = Agent(backend, tools_registry, agent_config)
        result_with = agent.run(question)
        elapsed_with = time.time() - t0

        # Count actual tokens
        gen_tokens = sum(
            len(backend.tokenizer.encode(s.generated_text))
            for s in result_with.steps
        )
        result_with.total_tokens = gen_tokens

        results_with_plan.append(result_with)

        n_steps = len(result_with.steps)
        tools_called = [s.tool_name for s in result_with.steps if s.tool_name]
        print(f"{n_steps} steps, {gen_tokens} tokens, {elapsed_with:.1f}s")
        print(f"         tools: {' -> '.join(tools_called) if tools_called else '(none)'}")

        # ---- P-: without plan ----
        print("  P- (no plan)...  ", end=" ", flush=True)
        t0 = time.time()
        result_without = run_agent_no_plan(
            backend, tools_registry, question,
            max_turns=MAX_TURNS,
            temperature=TEMPERATURE,
            verbose=False,
        )
        elapsed_without = time.time() - t0

        results_without_plan.append(result_without)

        n_steps = len(result_without.steps)
        tools_called = [s.tool_name for s in result_without.steps if s.tool_name]
        print(f"{n_steps} steps, {result_without.total_tokens} tokens, {elapsed_without:.1f}s")
        print(f"         tools: {' -> '.join(tools_called) if tools_called else '(none)'}")

    # ---- Save trajectories ----
    os.makedirs(EXP3_TRAJECTORY_DIR, exist_ok=True)

    path_with = save_trajectories_batch(
        results_with_plan,
        EXP3_TRAJECTORY_WITH_PLAN,
        model_id=model_id,
        config_name="with_plan",
        task_ids=task_ids,
        extra_meta={"experiment": "plan_vs_no_plan", "mode": "P+_with_plan"},
    )

    path_without = save_trajectories_batch(
        results_without_plan,
        EXP3_TRAJECTORY_WITHOUT_PLAN,
        model_id=model_id,
        config_name="without_plan",
        task_ids=task_ids,
        extra_meta={"experiment": "plan_vs_no_plan", "mode": "P-_without_plan"},
    )

    print(f"\nTrajectories saved:")
    print(f"  P+ (with plan):    {path_with}")
    print(f"  P- (without plan): {path_without}")

    # ---- Quick summary ----
    _print_comparison(results_with_plan, results_without_plan)

    return {"with_plan": path_with, "without_plan": path_without}


# ==========================================================================
# Quick summary
# ==========================================================================

def _print_comparison(
    with_plan: List[AgentResult],
    without_plan: List[AgentResult],
) -> None:
    """Print a quick side-by-side comparison."""
    n = len(with_plan)

    avg_steps_with = sum(len(r.steps) for r in with_plan) / max(n, 1)
    avg_steps_without = sum(len(r.steps) for r in without_plan) / max(n, 1)

    avg_tokens_with = sum(r.total_tokens for r in with_plan) / max(n, 1)
    avg_tokens_without = sum(r.total_tokens for r in without_plan) / max(n, 1)

    success_with = sum(1 for r in with_plan if r.success)
    success_without = sum(1 for r in without_plan if r.success)

    plan_calls_with = sum(
        1 for r in with_plan
        for s in r.steps if s.tool_name == "plan"
    )
    task_done_with = sum(
        1 for r in with_plan
        for s in r.steps if s.tool_name == "task_done"
    )
    task_done_without = sum(
        1 for r in without_plan
        for s in r.steps if s.tool_name == "task_done"
    )

    print(f"\n{'─' * 60}")
    print(f"  Quick Summary")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<30s} {'P+ (plan)':>12s} {'P- (no plan)':>14s}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 14}")
    print(f"  {'Avg steps/run':<30s} {avg_steps_with:>12.1f} {avg_steps_without:>14.1f}")
    print(f"  {'Avg tokens/run':<30s} {avg_tokens_with:>12.0f} {avg_tokens_without:>14.0f}")
    print(f"  {'Success rate':<30s} {success_with:>11d}/{n} {success_without:>13d}/{n}")
    print(f"  {'Plan calls (total)':<30s} {plan_calls_with:>12d} {'N/A':>14s}")
    print(f"  {'task_done calls':<30s} {task_done_with:>12d} {task_done_without:>14d}")
    print(f"{'─' * 60}")
    print(f"  Trajectories ready for LLM-as-Judge evaluation.")


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plan-first vs No-plan Experiment"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID (default from config)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on a small task subset only",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=0,
        help="Max tasks (0 = from config)",
    )
    args = parser.parse_args()

    model = args.model or EXP3_MODEL
    max_t = args.max_tasks or EXP3_MAX_TASKS
    if args.quick and max_t == 0:
        max_t = 5

    run_plan_experiment(model_id=model, max_tasks=max_t)


if __name__ == "__main__":
    main()
