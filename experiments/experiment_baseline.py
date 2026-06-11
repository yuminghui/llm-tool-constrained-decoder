"""
Experiment A — Baseline (Blank Control)
=========================================

Bare agent: no constraint decoder, no plan step, pure free generation.
This is the baseline against which all other experiments are compared.

Metrics produced per task:
  - prompt / generated / total tokens
  - completion score (evaluate_completion)
  - error type (categorize_error)
  - execution trajectory (saved to JSON)
  - plan tool usage rate (expected ~0 since no plan is enforced)
  - num constrained / free calls

Usage::

    python -m experiments.experiment_baseline
    python -m experiments.experiment_baseline --quick --max-tasks 5
"""

from __future__ import annotations

import json, sys, os, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.llm_backend import LLMBackend
from agentic.tools import load_tools_from_json, set_summary_backend
from agentic.agent import AgentResult, AgentStep
from experiments.trajectory_utils import (
    save_trajectories_batch, load_benchmark_tasks,
    evaluate_completion, categorize_error,
)
from experiments.config import (
    TOOLS_JSON_PATH, EVALUATE_JSON_PATH,
    EXP_A_MODELS, EXP_A_TASK_INDICES, EXP_A_MAX_TASKS, EXP_A_OUTPUT_DIR,
    QUANTIZATION_MODE, model_id as _mid, model_quantize,
    TEMPERATURE, MAX_TURNS, FREE_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD,
)
from experiments.prompts import SYSTEM_PROMPT, USER_TASK_PROMPT


def run_agent_baseline(backend, tools_registry, user_query: str) -> AgentResult:
    """Bare agent loop — all free generation, no plan."""
    start = time.time()
    steps: List[AgentStep] = []
    total_tokens = 0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK_PROMPT.format(query=user_query)},
    ]
    tool_defs = tools_registry.get_definitions()

    for turn in range(1, MAX_TURNS + 1):
        prompt = backend.build_prompt(messages, tool_defs)
        raw = backend.generate_free(prompt, max_new_tokens=FREE_MAX_NEW_TOKENS, temperature=TEMPERATURE)
        total_tokens += len(backend.tokenizer.encode(raw))

        tc = backend.parse_tool_call(raw)
        if tc and "name" in tc:
            t_name = tc["name"]; t_args = tc.get("arguments", {})
            t0 = time.time()
            step = AgentStep(step_index=turn, tool_name=t_name, tool_args=t_args,
                             tool_result=None, generated_text=raw, is_constrained=False, elapsed=0)
            entry = tools_registry.get(t_name)
            if entry:
                step.tool_result = tools_registry.execute(t_name, t_args)
            else:
                step.elapsed = time.time() - t0; steps.append(step)
                return AgentResult(user_query=user_query, steps=steps, final_answer=raw,
                                   total_time=time.time()-start, total_tokens=total_tokens, success=True)
            step.elapsed = time.time() - t0; steps.append(step)
            # Append to messages
            tcj = json.dumps({"name": t_name, "arguments": t_args}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": f"{backend.tool_call_prefix}{tcj}{backend.tool_call_suffix}"})
            messages.append({"role": "tool", "content": step.tool_result or ""})
            if t_name == "task_done":
                return AgentResult(user_query=user_query, steps=steps, final_answer=step.tool_result,
                                   total_time=time.time()-start, total_tokens=total_tokens, success=True)
            if t_name == "wait_user_instruction":
                return AgentResult(user_query=user_query, steps=steps, final_answer=step.tool_result,
                                   total_time=time.time()-start, total_tokens=total_tokens,
                                   success=True, error="wait_user_instruction")
        else:
            steps.append(AgentStep(step_index=turn, tool_name=None, tool_args=None, tool_result=None,
                                   generated_text=raw, is_constrained=False,
                                   elapsed=time.time() - start))
            return AgentResult(user_query=user_query, steps=steps, final_answer=raw,
                               total_time=time.time()-start, total_tokens=total_tokens, success=True)

    return AgentResult(user_query=user_query, steps=steps, final_answer=None,
                       total_time=time.time()-start, total_tokens=total_tokens,
                       success=False, error="Max turns reached")


@dataclass
class ExpMetrics:
    model_id: str; num_tasks: int
    total_prompt_tokens: int; total_generated_tokens: int; total_tokens_all: int
    total_success_tokens: int; num_success: int
    plan_calls: int; plan_first: int; num_constrained: int; num_free: int
    completed: int; error_counts: Dict[str, int]
    avg_tokens: float; avg_steps: float
    @property
    def plan_call_rate(self): return self.plan_calls / max(self.num_tasks, 1)
    @property
    def completion_rate(self): return self.completed / max(self.num_tasks, 1)
    @property
    def success_rate(self): return self.num_success / max(self.num_tasks, 1)


def run_experiment(models: list, task_indices=None, max_tasks=0) -> Dict[str, ExpMetrics]:
    tasks = load_benchmark_tasks(EVALUATE_JSON_PATH, task_indices, max_tasks)
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH)
    all_metrics: Dict[str, ExpMetrics] = {}
    tool_defs = tools_registry.get_definitions()

    print(f"\n{'='*70}\n  EXPERIMENT A — Baseline (Blank Control)\n  Models: {len(models)}  Tasks: {len(tasks)}\n{'='*70}")

    for entry in models:
        model_id = _mid(entry); quantize = model_quantize(entry)
        print(f"\n{'─'*60}\n  Model: {model_id}\n{'─'*60}")
        backend = LLMBackend(model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE)
        set_summary_backend(backend)

        results, pt_list, gt_list, task_ids = [], [], [], []
        plan_calls, plan_first, num_cstr, num_free, completed, num_ok = 0, 0, 0, 0, 0, 0
        total_prompt, total_gen, total_all, total_ok_tokens, total_steps = 0, 0, 0, 0, 0
        error_counts: Dict[str, int] = {}

        for i, task in enumerate(tasks):
            q = task["question"]; gt = task.get("trajectory_ground_truth", {})
            tid = f"task_{i:03d}"; task_ids.append(tid)
            print(f"  [{i+1}/{len(tasks)}] {q[:70]}...", end=" ", flush=True)

            msg = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_TASK_PROMPT.format(query=q)}]
            prompt_tokens = len(backend.tokenizer.encode(backend.build_prompt(msg, tool_defs)))

            t0 = time.time(); ar = run_agent_baseline(backend, tools_registry, q); elapsed = time.time()-t0
            gen_tokens = ar.total_tokens; total_tok = prompt_tokens + gen_tokens

            results.append(ar); pt_list.append(prompt_tokens); gt_list.append(gen_tokens)
            total_prompt += prompt_tokens; total_gen += gen_tokens; total_all += total_tok
            total_steps += len(ar.steps)

            tool_names = [s.tool_name for s in ar.steps if s.tool_name]
            if "plan" in tool_names: plan_calls += 1
            if tool_names and tool_names[0] == "plan": plan_first += 1
            num_cstr += sum(1 for s in ar.steps if s.is_constrained)
            num_free += sum(1 for s in ar.steps if not s.is_constrained)

            comp = evaluate_completion(ar.steps, gt, REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD)
            if comp["completed"]: completed += 1
            err_type = categorize_error(ar)
            error_counts[err_type] = error_counts.get(err_type, 0) + 1
            if ar.success: num_ok += 1; total_ok_tokens += total_tok

            pt = "P" if plan_first else ("p" if plan_calls else "-")
            print(f"{pt} {comp['score']:.2f} {err_type} | {total_tok}t {len(ar.steps)}s {elapsed:.1f}s")

        safe = model_id.replace("/","_"); os.makedirs(EXP_A_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(EXP_A_OUTPUT_DIR, f"{safe}.json")
        save_trajectories_batch(results, path, model_id=model_id, config_name="baseline",
            task_ids=task_ids, prompt_tokens_list=pt_list, generated_tokens_list=gt_list,
            extra_meta={"experiment": "exp_a_baseline"})
        print(f"  Saved: {path}")

        m = ExpMetrics(model_id=model_id, num_tasks=len(tasks),
            total_prompt_tokens=total_prompt, total_generated_tokens=total_gen, total_tokens_all=total_all,
            total_success_tokens=total_ok_tokens, num_success=num_ok,
            plan_calls=plan_calls, plan_first=plan_first, num_constrained=num_cstr, num_free=num_free,
            completed=completed, error_counts=error_counts,
            avg_tokens=total_all/max(len(tasks),1), avg_steps=total_steps/max(len(tasks),1))
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
    print(f"    plan_call={m.plan_call_rate:.0%}  completion={m.completion_rate:.0%}  success={m.success_rate:.0%}")
    print(f"    tokens: total={m.total_tokens_all}  success_tokens={m.total_success_tokens}")
    print(f"    constrained={m.num_constrained}  free={m.num_free}  errors={m.error_counts}")


def print_report(metrics: Dict[str, ExpMetrics]):
    print(f"\n{'='*80}\n  EXPERIMENT A — FINAL REPORT: Baseline\n{'='*80}")
    sorted_m = sorted(metrics.values(), key=lambda m: m.completion_rate, reverse=True)
    print(f"\n{'Model':<42s} {'Compl':>6s} {'Success':>8s} {'Plan':>6s} {'AvgTok':>7s} {'Cstr':>5s} {'Free':>5s}")
    print("-"*85)
    for m in sorted_m:
        print(f"{m.model_id:<42s} {m.completion_rate:>5.0%} {m.success_rate:>8.0%} "
              f"{m.plan_call_rate:>5.0%} {m.avg_tokens:>7.0f} {m.num_constrained:>5d} {m.num_free:>5d}")
    print("-"*85)
    for m in sorted_m:
        if m.error_counts:
            print(f"  {m.model_id}: errors={m.error_counts}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Experiment A — Baseline")
    p.add_argument("--quick", action="store_true"); p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    models = [(args.model, False)] if args.model else EXP_A_MODELS
    max_t = args.max_tasks or EXP_A_MAX_TASKS
    if args.quick and max_t == 0: max_t = 5
    metrics = run_experiment(models, max_tasks=max_t)
    print_report(metrics)


if __name__ == "__main__":
    main()
