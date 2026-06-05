"""
Experiment C — Constrained Plan Only
======================================

Constraint decoder forces ``plan`` at step 0.
All subsequent steps are free generation (model decides which tools to call).

Metrics: tokens, completion, trajectory, plan usage, constrained/free calls,
         error types, tool_call_match_rate (plan expected_tools vs actual).

Usage::

    python -m experiments.experiment_constrained_plan
    python -m experiments.experiment_constrained_plan --quick --max-tasks 5
"""

from __future__ import annotations

import json, sys, os, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.llm_backend import LLMBackend
from agentic.tools import load_tools_from_json, set_summary_backend
from agentic.agent import Agent, AgentConfig, AgentResult, AgentStep
from experiments.trajectory_utils import (
    save_trajectories_batch, load_benchmark_tasks,
    evaluate_completion, categorize_error,
)
from experiments.config import (
    TOOLS_JSON_PATH, EVALUATE_JSON_PATH,
    EXP_C_MODELS, EXP_C_TASK_INDICES, EXP_C_MAX_TASKS, EXP_C_OUTPUT_DIR,
    QUANTIZATION_MODE, model_id as _mid, model_quantize,
    TEMPERATURE, MAX_TURNS,
    REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD,
)
from experiments.prompts import SYSTEM_PROMPT, USER_TASK_PROMPT


@dataclass
class ExpMetrics:
    model_id: str; num_tasks: int
    total_prompt_tokens: int; total_generated_tokens: int; total_tokens_all: int
    total_success_tokens: int; num_success: int
    plan_calls: int; plan_first: int; num_constrained: int; num_free: int
    completed: int; error_counts: Dict[str, int]
    avg_tokens: float; avg_steps: float
    tool_match_total: int = 0; tool_match_hits: int = 0
    @property
    def plan_call_rate(self): return self.plan_calls / max(self.num_tasks, 1)
    @property
    def completion_rate(self): return self.completed / max(self.num_tasks, 1)
    @property
    def success_rate(self): return self.num_success / max(self.num_tasks, 1)
    @property
    def match_rate(self):
        return self.tool_match_hits / max(self.tool_match_total, 1) if self.tool_match_total else 1.0


def run_experiment(models: list, task_indices=None, max_tasks=0) -> Dict[str, ExpMetrics]:
    tasks = load_benchmark_tasks(EVALUATE_JSON_PATH, task_indices, max_tasks)
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    all_metrics: Dict[str, ExpMetrics] = {}
    tool_defs = tools_registry.get_definitions()

    print(f"\n{'='*70}\n  EXPERIMENT C — Constrained Plan Only\n  Models: {len(models)}  Tasks: {len(tasks)}\n{'='*70}")

    for entry in models:
        model_id = _mid(entry); quantize = model_quantize(entry)
        print(f"\n{'─'*60}\n  Model: {model_id}\n{'─'*60}")
        backend = LLMBackend(model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE)

        results, pt_list, gt_list, task_ids = [], [], [], []
        plan_calls, plan_first, num_cstr, num_free, completed, num_ok = 0, 0, 0, 0, 0, 0
        total_prompt, total_gen, total_all, total_ok_tokens, total_steps = 0, 0, 0, 0, 0
        match_total, match_hits = 0, 0
        error_counts: Dict[str, int] = {}

        # use_constrained_decoder=False: plan is constrained, execution is free
        agent_conf = AgentConfig(max_turns=MAX_TURNS, temperature=TEMPERATURE,
                                 use_constrained_decoder=False, verbose=False,
                                 system_prompt=SYSTEM_PROMPT)

        for i, task in enumerate(tasks):
            q = task["question"]; gt_truth = task.get("trajectory_ground_truth", {})
            tid = f"task_{i:03d}"; task_ids.append(tid)
            print(f"  [{i+1}/{len(tasks)}] {q[:70]}...", end=" ", flush=True)

            msg = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_TASK_PROMPT.format(query=q)}]
            prompt_tokens = len(backend.tokenizer.encode(backend.build_prompt(msg, tool_defs)))

            t0 = time.time()
            agent = Agent(backend, tools_registry, agent_conf)
            ar = agent.run(USER_TASK_PROMPT.format(query=q))
            ar.user_query = q  # record original question, not the augmented prompt
            elapsed = time.time()-t0

            gen_tokens = sum(len(backend.tokenizer.encode(s.generated_text)) for s in ar.steps)
            total_tok = prompt_tokens + gen_tokens

            results.append(ar); pt_list.append(prompt_tokens); gt_list.append(gen_tokens)
            total_prompt += prompt_tokens; total_gen += gen_tokens; total_all += total_tok
            total_steps += len(ar.steps)

            tool_names = [s.tool_name for s in ar.steps if s.tool_name]
            if "plan" in tool_names: plan_calls += 1
            if tool_names and tool_names[0] == "plan": plan_first += 1
            num_cstr += sum(1 for s in ar.steps if s.is_constrained)
            num_free += sum(1 for s in ar.steps if not s.is_constrained)

            # tool_call_match_rate: compare plan expected_tools with actual tools called
            plan_step = next((s for s in ar.steps if s.tool_name == "plan" and s.tool_args), None)
            if plan_step:
                expected = set()
                for s in (plan_step.tool_args.get("steps", []) or []):
                    if isinstance(s, dict) and s.get("expected_tools"):
                        for t in s["expected_tools"]:
                            if isinstance(t, str): expected.add(t)
                actual = set(t for t in tool_names if t and t not in ("plan", "task_done", "task_summary"))
                if expected:
                    match_total += len(expected); match_hits += len(expected & actual)

            comp = evaluate_completion(ar.steps, gt_truth, REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD)
            if comp["completed"]: completed += 1
            err_type = categorize_error(ar)
            error_counts[err_type] = error_counts.get(err_type, 0) + 1
            if ar.success: num_ok += 1; total_ok_tokens += total_tok

            print(f"{comp['score']:.2f} {err_type} | {total_tok}t {len(ar.steps)}s {elapsed:.1f}s | plan=Y cstr={sum(1 for s in ar.steps if s.is_constrained)} free={sum(1 for s in ar.steps if not s.is_constrained)}")

        safe = model_id.replace("/","_"); os.makedirs(EXP_C_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(EXP_C_OUTPUT_DIR, f"{safe}.json")
        save_trajectories_batch(results, path, model_id=model_id, config_name="constrained_plan",
            task_ids=task_ids, prompt_tokens_list=pt_list, generated_tokens_list=gt_list,
            extra_meta={"experiment": "exp_c_constrained_plan"})
        print(f"  Saved: {path}")

        m = ExpMetrics(model_id=model_id, num_tasks=len(tasks),
            total_prompt_tokens=total_prompt, total_generated_tokens=total_gen, total_tokens_all=total_all,
            total_success_tokens=total_ok_tokens, num_success=num_ok,
            plan_calls=plan_calls, plan_first=plan_first, num_constrained=num_cstr, num_free=num_free,
            completed=completed, error_counts=error_counts,
            avg_tokens=total_all/max(len(tasks),1), avg_steps=total_steps/max(len(tasks),1),
            tool_match_total=match_total, tool_match_hits=match_hits)
        all_metrics[model_id] = m
        _print_summary(m)

        # Release GPU memory before next model
        set_summary_backend(None)
        del backend
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_metrics


def _print_summary(m: ExpMetrics):
    print(f"\n  [{m.model_id}]")
    print(f"    completion={m.completion_rate:.0%}  success={m.success_rate:.0%}  plan_call={m.plan_call_rate:.0%}")
    print(f"    tokens: total={m.total_tokens_all}  success_tokens={m.total_success_tokens}")
    print(f"    constrained={m.num_constrained}  free={m.num_free}  match_rate={m.match_rate:.0%}  errors={m.error_counts}")


def print_report(metrics: Dict[str, ExpMetrics]):
    print(f"\n{'='*80}\n  EXPERIMENT C — FINAL REPORT: Constrained Plan\n{'='*80}")
    sorted_m = sorted(metrics.values(), key=lambda m: m.completion_rate, reverse=True)
    print(f"\n{'Model':<42s} {'Compl':>6s} {'Success':>8s} {'Match':>6s} {'AvgTok':>7s} {'Cstr':>5s} {'Free':>5s}")
    print("-"*85)
    for m in sorted_m:
        print(f"{m.model_id:<42s} {m.completion_rate:>5.0%} {m.success_rate:>8.0%} "
              f"{m.match_rate:>5.0%} {m.avg_tokens:>7.0f} {m.num_constrained:>5d} {m.num_free:>5d}")
    print("-"*85)
    for m in sorted_m:
        if m.error_counts:
            print(f"  {m.model_id}: errors={m.error_counts}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Experiment C — Constrained Plan")
    p.add_argument("--quick", action="store_true"); p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    models = [(args.model, False)] if args.model else EXP_C_MODELS
    max_t = args.max_tasks or EXP_C_MAX_TASKS
    if args.quick and max_t == 0: max_t = 5
    metrics = run_experiment(models, max_tasks=max_t)
    print_report(metrics)


if __name__ == "__main__":
    main()
