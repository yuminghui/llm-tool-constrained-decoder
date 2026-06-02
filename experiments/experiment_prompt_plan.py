"""
Prompt-only Plan Enforcement Experiment
=========================================

测试仅靠提示词约束"首先使用 plan 工具规划再行动"时：

1. 各模型实际调用 ``plan`` 工具的占比（plan_call_rate）
2. Agent 任务完成效果（标准 completion 评测）

全部使用自由生成（无约束解码器），**唯一变量是 system prompt** 是否强调
"必须先用 plan"。因此可以直接和 Exp3（约束解码器强制 plan）对比：
是谁在真正起作用 — 提示词还是约束解码器？

Usage::

    python -m experiments.experiment_prompt_plan
    python -m experiments.experiment_prompt_plan --quick --max-tasks 5
"""

from __future__ import annotations

import json
import sys
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.llm_backend import LLMBackend
from agentic.tools import load_tools_from_json
from agentic.agent import AgentResult, AgentStep

from experiments.config import (
    TOOLS_JSON_PATH,
    EVALUATE_JSON_PATH,
    EXP5_MODELS,
    EXP5_TASK_INDICES,
    EXP5_MAX_TASKS,
    EXP5_OUTPUT_DIR,
    QUANTIZATION_MODE,
    model_id as cfg_model_id,
    model_quantize,
    TEMPERATURE,
    MAX_TURNS,
    FREE_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT,
    OPTIONAL_STEP_WEIGHT,
    TASK_DONE_WEIGHT,
    COMPLETION_THRESHOLD,
)
from experiments.trajectory_utils import save_trajectories_batch


# ==========================================================================
# System prompt — strongly encourages plan-first (but no constraint decoder)
# ==========================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools. "
    "IMPORTANT: You MUST call the 'plan' tool FIRST before taking any other action. "
    "After creating the plan, execute the steps in order, one tool at a time. "
    "When finished, use task_summary to summarize and task_done to signal completion. "
    "Respond in Chinese."
)


# ==========================================================================
# Task completion evaluator (same as other experiments)
# ==========================================================================

def evaluate_completion(
    agent_steps: List[AgentStep],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
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
# Agent loop — all free generation (no constraint decoder)
# ==========================================================================

def run_agent_prompt_plan(
    backend: LLMBackend,
    tools_registry,
    user_query: str,
    max_turns: int = MAX_TURNS,
    temperature: float = TEMPERATURE,
) -> AgentResult:
    """Run agent in free mode with a plan-encouraging system prompt.

    No constraint decoder is used.  Whether the model actually calls
    ``plan`` depends entirely on how well it follows the prompt.
    """
    start_time = time.time()
    steps: List[AgentStep] = []
    total_tokens = 0

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    tool_defs = tools_registry.get_definitions()

    for turn in range(1, max_turns + 1):
        prompt = backend.build_prompt(messages, tool_defs)
        raw_output = backend.generate_free(
            prompt=prompt,
            max_new_tokens=FREE_MAX_NEW_TOKENS,
            temperature=temperature,
        )
        total_tokens += len(backend.tokenizer.encode(raw_output))

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
            else:
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

@dataclass
class PromptPlanMetrics:
    """Per-model aggregate metrics."""
    model_id: str
    num_tasks: int
    plan_calls: int            # how many tasks called plan at all
    plan_first: int             # how many called plan as the FIRST tool
    plan_call_rate: float
    plan_first_rate: float
    avg_tokens: float
    avg_steps: float
    completion_rate: float
    success_rate: float


def run_experiment_5(
    models: list,
    task_indices: Optional[List[int]] = None,
    max_tasks: int = 0,
) -> Dict[str, PromptPlanMetrics]:
    """Run prompt-only plan enforcement experiment.

    Returns:
        ``{model_id: PromptPlanMetrics}``
    """
    with open(EVALUATE_JSON_PATH, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    if task_indices is not None:
        all_tasks = [all_tasks[i] for i in task_indices if i < len(all_tasks)]
    if max_tasks and max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 5 — Prompt-only Plan Enforcement")
    print(f"  Models: {len(models)}  |  Tasks: {len(all_tasks)}")
    print("  Variable: prompt says 'MUST call plan first' (no constraint decoder)")
    print("=" * 70)

    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    all_metrics: Dict[str, PromptPlanMetrics] = {}

    for entry in models:
        model_id = cfg_model_id(entry)
        quantize = model_quantize(entry)

        print(f"\n{'─' * 60}")
        print(f"  Model: {model_id}")
        print(f"{'─' * 60}")

        backend = LLMBackend(
            model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE,
        )

        results: List[AgentResult] = []
        prompt_tokens_list: List[int] = []
        gen_tokens_list: List[int] = []
        task_ids: List[str] = []
        plan_calls = 0
        plan_first = 0
        completed = 0
        successful = 0
        total_tokens_sum = 0
        total_steps_sum = 0

        tool_defs = tools_registry.get_definitions()

        for i, task in enumerate(all_tasks):
            question = task["question"]
            gt = task.get("trajectory_ground_truth", {})
            tid = f"task_{i:03d}"
            task_ids.append(tid)

            print(f"  [{i+1}/{len(all_tasks)}] {question[:70]}...", end=" ", flush=True)

            # Count prompt tokens
            msg = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_tokens = len(backend.tokenizer.encode(
                backend.build_prompt(msg, tool_defs)
            ))

            t0 = time.time()
            agent_result = run_agent_prompt_plan(backend, tools_registry, question)
            elapsed = time.time() - t0

            gen_tokens = agent_result.total_tokens
            total_tokens = prompt_tokens + gen_tokens

            results.append(agent_result)
            prompt_tokens_list.append(prompt_tokens)
            gen_tokens_list.append(gen_tokens)

            # Plan metrics
            tool_names = [s.tool_name for s in agent_result.steps if s.tool_name]
            called_plan = "plan" in tool_names
            first_is_plan = tool_names and tool_names[0] == "plan"

            if called_plan:
                plan_calls += 1
            if first_is_plan:
                plan_first += 1

            # Completion
            comp = evaluate_completion(agent_result.steps, gt)
            if comp["completed"]:
                completed += 1
            if agent_result.success:
                successful += 1

            total_tokens_sum += total_tokens
            total_steps_sum += len(agent_result.steps)

            plan_tag = "P" if first_is_plan else ("p" if called_plan else "-")
            comp_tag = "OK" if comp["completed"] else "NOK"
            print(f"{plan_tag} {comp_tag} | {total_tokens}t | "
                  f"{len(agent_result.steps)} steps | {elapsed:.1f}s | "
                  f"tools: {' → '.join(tool_names) if tool_names else '(none)'}")

        # Per-model save
        safe_name = model_id.replace("/", "_")
        os.makedirs(EXP5_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(EXP5_OUTPUT_DIR, f"{safe_name}.json")
        save_trajectories_batch(
            results, out_path,
            model_id=model_id, config_name="prompt_only_plan",
            task_ids=task_ids,
            prompt_tokens_list=prompt_tokens_list,
            generated_tokens_list=gen_tokens_list,
            extra_meta={"experiment": "exp5_prompt_plan_enforcement"},
        )
        print(f"  Trajectories saved: {out_path}")

        n = len(all_tasks)
        metrics = PromptPlanMetrics(
            model_id=model_id, num_tasks=n,
            plan_calls=plan_calls, plan_first=plan_first,
            plan_call_rate=plan_calls / max(n, 1),
            plan_first_rate=plan_first / max(n, 1),
            avg_tokens=total_tokens_sum / max(n, 1),
            avg_steps=total_steps_sum / max(n, 1),
            completion_rate=completed / max(n, 1),
            success_rate=successful / max(n, 1),
        )
        all_metrics[model_id] = metrics
        _print_model_summary(metrics)

    return all_metrics


# ==========================================================================
# Summary
# ==========================================================================

def _print_model_summary(m: PromptPlanMetrics) -> None:
    print(f"\n  [{m.model_id}]")
    print(f"    plan_call={m.plan_calls}/{m.num_tasks} ({m.plan_call_rate:.0%})  "
          f"plan_first={m.plan_first}/{m.num_tasks} ({m.plan_first_rate:.0%})")
    print(f"    completion={m.completion_rate:.0%}  success={m.success_rate:.0%}  "
          f"avg_tokens={m.avg_tokens:.0f}  avg_steps={m.avg_steps:.1f}")


def print_exp5_report(all_metrics: Dict[str, PromptPlanMetrics]) -> None:
    """Print final report with plan_call_rate as the key metric."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT 5 — FINAL REPORT: Prompt-only Plan Enforcement")
    print("=" * 80)

    sorted_metrics = sorted(
        all_metrics.values(),
        key=lambda m: m.plan_call_rate, reverse=True,
    )

    print(f"\n{'Model':<42s} {'PlanCall':>9s} {'Plan1st':>8s} "
          f"{'Compl':>6s} {'Success':>8s} {'AvgTok':>7s} {'Steps':>6s}")
    print("-" * 90)

    for m in sorted_metrics:
        print(f"{m.model_id:<42s} {m.plan_call_rate:>8.0%} {m.plan_first_rate:>8.0%} "
              f"{m.completion_rate:>5.0%} {m.success_rate:>8.0%} "
              f"{m.avg_tokens:>7.0f} {m.avg_steps:>6.1f}")

    print("-" * 90)

    best_follower = sorted_metrics[0]
    worst_follower = sorted_metrics[-1]
    print(f"\n  Best plan follower:  {best_follower.model_id} "
          f"({best_follower.plan_call_rate:.0%} plan calls)")
    print(f"  Worst plan follower: {worst_follower.model_id} "
          f"({worst_follower.plan_call_rate:.0%} plan calls)")

    print(f"\n  Compare this table with Exp3 (constraint decoder forces plan):")
    print(f"  - If prompt-only achieves similar completion to Exp3 → prompt is enough")
    print(f"  - If prompt-only has lower plan_call_rate AND lower completion")
    print(f"    → constraint decoder adds real value beyond prompt engineering")


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prompt-only Plan Enforcement Experiment"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Single model override")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick-test models")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max tasks (0 = from config)")
    args = parser.parse_args()

    models = [(args.model, False)] if args.model else EXP5_MODELS
    max_t = args.max_tasks or EXP5_MAX_TASKS
    if args.quick and max_t == 0:
        max_t = 5

    metrics = run_experiment_5(models=models, max_tasks=max_t)
    print_exp5_report(metrics)


if __name__ == "__main__":
    main()
