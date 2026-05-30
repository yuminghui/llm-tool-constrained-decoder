"""
Ablation Experiment — Constraint Decoder for Plan Step
=======================================================

消融实验：在相同条件下（提示词、模型、温度、任务），**唯一变量**是第一步
``plan`` 工具是否使用约束解码器。

**Mode A  (with constraint)**:
    Step 0: 约束解码器强制 ``plan`` → 后续步骤自由生成
    = Agent(use_constrained_decoder=False) 的当前行为

**Mode B  (without constraint)**:
    全部步骤自由生成（包括 plan），模型自行决定何时调用 plan

控制变量：同一模型、同一 system prompt、同一温度、同一任务集。
两组轨迹分别写入两个 JSON 文件，供后续 LLM-as-Judge 评测。

Usage::

    python -m experiments.experiment_ablation
    python -m experiments.experiment_ablation --quick --max-tasks 5
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
from agentic.agent import Agent, AgentConfig, AgentResult, AgentStep

from experiments.config import (
    TOOLS_JSON_PATH,
    EVALUATE_JSON_PATH,
    EXP4_MODELS,
    EXP4_TASK_INDICES,
    EXP4_MAX_TASKS,
    EXP4_OUTPUT_DIR,
    TEMPERATURE,
    MAX_TURNS,
    FREE_MAX_NEW_TOKENS,
)
from experiments.trajectory_utils import save_trajectories_batch


# ==========================================================================
# System prompt — identical for both modes (control variable)
# ==========================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools. "
    "Always call the 'plan' tool first to create a step-by-step plan "
    "before taking any action. Execute tools one at a time. "
    "When you have gathered all needed information, use task_summary "
    "to summarize and task_done to finish. Respond in Chinese."
)


# ==========================================================================
# Mode B: all-free agent (no constraint decoder at all)
# ==========================================================================

def run_agent_all_free(
    backend: LLMBackend,
    tools_registry,
    user_query: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = MAX_TURNS,
    temperature: float = TEMPERATURE,
    verbose: bool = False,
) -> AgentResult:
    """Run agent in fully free mode — no constraint decoder for any step.

    The model receives the system prompt (which encourages planning) and
    decides on its own whether to call ``plan``, what tools to use, and
    in what order.  All generation is unconstrained.
    """
    start_time = time.time()
    steps: List[AgentStep] = []
    total_tokens = 0

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    tool_defs = tools_registry.get_definitions()

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"    [free turn {turn}] generating...")

        prompt = backend.build_prompt(messages, tool_defs)
        raw_output = backend.generate_free(
            prompt=prompt,
            max_new_tokens=FREE_MAX_NEW_TOKENS,
            temperature=temperature,
        )
        total_tokens += len(backend.tokenizer.encode(raw_output))

        if verbose:
            print(f"      output: {raw_output[:150]}...")

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
                if verbose:
                    print(f"      -> unknown tool '{t_name}', treating as final")
                step.elapsed = time.time() - t0
                steps.append(step)
                return AgentResult(
                    user_query=user_query, steps=steps,
                    final_answer=raw_output,
                    total_time=time.time() - start_time,
                    total_tokens=total_tokens, success=True,
                )

            step.elapsed = time.time() - t0
            steps.append(step)

            # Append to conversation
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

            if t_name == "task_done":
                return AgentResult(
                    user_query=user_query, steps=steps,
                    final_answer=step.tool_result,
                    total_time=time.time() - start_time,
                    total_tokens=total_tokens, success=True,
                )
        else:
            if verbose:
                print(f"      -> final answer")
            steps.append(AgentStep(
                step_index=turn, tool_name=None, tool_args=None,
                tool_result=None, generated_text=raw_output,
                is_constrained=False, elapsed=0.0,
            ))
            return AgentResult(
                user_query=user_query, steps=steps,
                final_answer=raw_output,
                total_time=time.time() - start_time,
                total_tokens=total_tokens, success=True,
            )

    return AgentResult(
        user_query=user_query, steps=steps, final_answer=None,
        total_time=time.time() - start_time, total_tokens=total_tokens,
        success=False, error="Max turns reached",
    )


# ==========================================================================
# Main experiment
# ==========================================================================

def run_ablation_experiment(
    models: List[str],
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> Dict[str, Dict[str, str]]:
    """Run ablation experiment for each model.

    Returns: ``{model_id: {"with_constraint": path, "without_constraint": path}}``
    """
    with open(EVALUATE_JSON_PATH, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    if task_indices is not None:
        all_tasks = [all_tasks[i] for i in task_indices if i < len(all_tasks)]
    if max_tasks and max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]

    print("\n" + "=" * 70)
    print("  ABLATION EXPERIMENT")
    print("  Constraint Decoder for Plan Step: ON vs OFF")
    print(f"  Models: {len(models)}  |  Tasks: {len(all_tasks)}")
    print("  Control: same prompt, same temp, same tasks")
    print("  Variable: constraint decoder for plan (step 0)")
    print("=" * 70)

    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    all_paths: Dict[str, Dict[str, str]] = {}

    for model_id in models:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_id}")
        print(f"{'─' * 60}")

        backend = LLMBackend(model_id)

        results_with: List[AgentResult] = []
        ptokens_with: List[int] = []
        gtokens_with: List[int] = []
        results_without: List[AgentResult] = []
        ptokens_without: List[int] = []
        gtokens_without: List[int] = []
        task_ids: List[str] = []

        config_a = AgentConfig(
            max_turns=MAX_TURNS, temperature=TEMPERATURE,
            use_constrained_decoder=False, verbose=False,
            system_prompt=SYSTEM_PROMPT,
        )
        tool_defs = tools_registry.get_definitions()

        for i, task in enumerate(all_tasks):
            question = task["question"]
            task_id = f"task_{i:03d}"
            task_ids.append(task_id)

            print(f"\n  [{i+1}/{len(all_tasks)}] {question[:80]}")

            # ---- Mode A: plan constrained ----
            print("    A (constraint ON for plan)...", end=" ", flush=True)

            msg_a = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_tokens_a = len(backend.tokenizer.encode(
                backend.build_prompt(msg_a, tool_defs)
            ))

            t0 = time.time()
            agent_a = Agent(backend, tools_registry, config_a)
            result_a = agent_a.run(question)
            elapsed_a = time.time() - t0

            gen_tokens_a = sum(
                len(backend.tokenizer.encode(s.generated_text))
                for s in result_a.steps
            )
            result_a.total_tokens = gen_tokens_a
            results_with.append(result_a)
            ptokens_with.append(prompt_tokens_a)
            gtokens_with.append(gen_tokens_a)

            tools_a = [s.tool_name for s in result_a.steps if s.tool_name]
            plan_called_a = "plan" in tools_a
            print(f"{len(result_a.steps)} steps, {gen_tokens_a}t, {elapsed_a:.1f}s")
            print(f"         plan={'Y' if plan_called_a else 'N'} | "
                  f"tools: {' -> '.join(tools_a) if tools_a else '(none)'}")

            # ---- Mode B: all free ----
            print("    B (constraint OFF entirely)...", end=" ", flush=True)

            msg_b = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_tokens_b = len(backend.tokenizer.encode(
                backend.build_prompt(msg_b, tool_defs)
            ))

            t0 = time.time()
            result_b = run_agent_all_free(
                backend, tools_registry, question,
                system_prompt=SYSTEM_PROMPT,
                max_turns=MAX_TURNS, temperature=TEMPERATURE,
            )
            elapsed_b = time.time() - t0
            results_without.append(result_b)
            ptokens_without.append(prompt_tokens_b)
            gtokens_without.append(result_b.total_tokens)

            tools_b = [s.tool_name for s in result_b.steps if s.tool_name]
            plan_called_b = "plan" in tools_b
            print(f"{len(result_b.steps)} steps, {result_b.total_tokens}t, {elapsed_b:.1f}s")
            print(f"         plan={'Y' if plan_called_b else 'N'} | "
                  f"tools: {' -> '.join(tools_b) if tools_b else '(none)'}")

        # Per-model save
        safe_name = model_id.replace("/", "_")
        os.makedirs(EXP4_OUTPUT_DIR, exist_ok=True)
        path_with = os.path.join(EXP4_OUTPUT_DIR, f"{safe_name}_with_constraint.json")
        path_without = os.path.join(EXP4_OUTPUT_DIR, f"{safe_name}_without_constraint.json")

        save_trajectories_batch(
            results_with, path_with, model_id=model_id,
            config_name="ablation_with_constraint", task_ids=task_ids,
            prompt_tokens_list=ptokens_with, generated_tokens_list=gtokens_with,
            extra_meta={"experiment": "ablation_constraint_decoder_for_plan",
                        "variable": "constraint decoder ON for plan step"},
        )
        save_trajectories_batch(
            results_without, path_without, model_id=model_id,
            config_name="ablation_without_constraint", task_ids=task_ids,
            prompt_tokens_list=ptokens_without, generated_tokens_list=gtokens_without,
            extra_meta={"experiment": "ablation_constraint_decoder_for_plan",
                        "variable": "constraint decoder OFF (all free)"},
        )
        print(f"\n  Trajectories saved:")
        print(f"    A (constraint ON):  {path_with}")
        print(f"    B (all free):       {path_without}")

        _print_summary(results_with, results_without)
        all_paths[model_id] = {"with_constraint": path_with, "without_constraint": path_without}

    return all_paths


# ==========================================================================
# Summary
# ==========================================================================

def _print_summary(
    with_constraint: List[AgentResult],
    without_constraint: List[AgentResult],
) -> None:
    """Print a quick side-by-side summary."""
    n = len(with_constraint)

    avg_steps_a = sum(len(r.steps) for r in with_constraint) / max(n, 1)
    avg_steps_b = sum(len(r.steps) for r in without_constraint) / max(n, 1)

    avg_tokens_a = sum(r.total_tokens for r in with_constraint) / max(n, 1)
    avg_tokens_b = sum(r.total_tokens for r in without_constraint) / max(n, 1)

    plan_calls_a = sum(
        1 for r in with_constraint
        for s in r.steps if s.tool_name == "plan"
    )
    plan_calls_b = sum(
        1 for r in without_constraint
        for s in r.steps if s.tool_name == "plan"
    )

    task_done_a = sum(
        1 for r in with_constraint
        for s in r.steps if s.tool_name == "task_done"
    )
    task_done_b = sum(
        1 for r in without_constraint
        for s in r.steps if s.tool_name == "task_done"
    )

    success_a = sum(1 for r in with_constraint if r.success)
    success_b = sum(1 for r in without_constraint if r.success)

    print(f"\n{'─' * 65}")
    print(f"  Quick Summary  (n={n} tasks)")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<35s} {'A (cstr ON)':>12s} {'B (all free)':>14s}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 14}")
    print(f"  {'Avg steps/run':<35s} {avg_steps_a:>12.1f} {avg_steps_b:>14.1f}")
    print(f"  {'Avg tokens/run':<35s} {avg_tokens_a:>12.0f} {avg_tokens_b:>14.0f}")
    print(f"  {'Success rate':<35s} {success_a:>11d}/{n} {success_b:>13d}/{n}")
    print(f"  {'Plan calls (total)':<35s} {plan_calls_a:>12d} {plan_calls_b:>14d}")
    print(f"  {'task_done calls':<35s} {task_done_a:>12d} {task_done_b:>14d}")
    print(f"{'─' * 65}")
    print(f"  Trajectories ready for LLM-as-Judge evaluation.")
    print(f"  Key question: does forced plan (A) produce better trajectories")
    print(f"  than model-decided plan (B)?")
    print(f"{'─' * 65}")


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation Experiment — Constraint Decoder for Plan Step"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID (default from config)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on a small task subset",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=0,
        help="Max tasks (0 = from config)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else EXP4_MODELS
    max_t = args.max_tasks or EXP4_MAX_TASKS
    if args.quick and max_t == 0:
        max_t = 5

    run_ablation_experiment(models=models, max_tasks=max_t)


if __name__ == "__main__":
    main()
