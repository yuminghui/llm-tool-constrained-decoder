"""
Experiment E — Prompt-only Plan Enforcement
=============================================

System prompt says "MUST call plan first". No constraint decoder.
Tests whether models voluntarily follow the plan-first instruction.

Metrics: tokens, completion, trajectory, plan_call_rate, plan_first_rate,
         constrained/free calls, error types, tool_call_match_rate.

Usage::

    python -m experiments.experiment_prompt_plan
    python -m experiments.experiment_prompt_plan --quick --max-tasks 5
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
    EXP_E_MODELS, EXP_E_TASK_INDICES, EXP_E_MAX_TASKS, EXP_E_OUTPUT_DIR,
    QUANTIZATION_MODE, model_id as _mid, model_quantize,
    TEMPERATURE, MAX_TURNS, FREE_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools. "
    "IMPORTANT: You MUST call the 'plan' tool FIRST before taking any other action. "
    "After creating the plan, execute the steps in order, one tool at a time. "
    "When finished, use task_summary to summarize and task_done to finish. "
    "Respond in Chinese."
)


def run_agent_prompt_plan(backend, tools_registry, user_query: str) -> AgentResult:
    """Free-generation agent loop with plan-encouraging system prompt."""
    start = time.time()
    steps: List[AgentStep] = []
    total_tokens = 0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
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
            tcj = json.dumps({"name": t_name, "arguments": t_args}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": f"{backend.tool_call_prefix}{tcj}{backend.tool_call_suffix}"})
            messages.append({"role": "tool", "content": step.tool_result or ""})
            if t_name == "task_done":
                return AgentResult(user_query=user_query, steps=steps, final_answer=step.tool_result,
                                   total_time=time.time()-start, total_tokens=total_tokens, success=True)
        else:
            steps.append(AgentStep(step_index=turn, tool_name=None, tool_args=None, tool_result=None,
                                   generated_text=raw, is_constrained=False, elapsed=0))
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
    tool_match_total: int = 0; tool_match_hits: int = 0
    @property
    def plan_call_rate(self): return self.plan_calls / max(self.num_tasks, 1)
    @property
    def plan_first_rate(self): return self.plan_first / max(self.num_tasks, 1)
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

    print(f"\n{'='*70}\n  EXPERIMENT E — Prompt-only Plan Enforcement\n  Models: {len(models)}  Tasks: {len(tasks)}\n{'='*70}")

    for entry in models:
        model_id = _mid(entry); quantize = model_quantize(entry)
        print(f"\n{'─'*60}\n  Model: {model_id}\n{'─'*60}")
        backend = LLMBackend(model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE)

        results, pt_list, gt_list, task_ids = [], [], [], []
        plan_calls, plan_first, num_cstr, num_free, completed, num_ok = 0, 0, 0, 0, 0, 0
        total_prompt, total_gen, total_all, total_ok_tokens, total_steps = 0, 0, 0, 0, 0
        match_total, match_hits = 0, 0
        error_counts: Dict[str, int] = {}

        for i, task in enumerate(tasks):
            q = task["question"]; gt_truth = task.get("trajectory_ground_truth", {})
            tid = f"task_{i:03d}"; task_ids.append(tid)
            print(f"  [{i+1}/{len(tasks)}] {q[:70]}...", end=" ", flush=True)

            msg = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
            prompt_tokens = len(backend.tokenizer.encode(backend.build_prompt(msg, tool_defs)))

            t0 = time.time(); ar = run_agent_prompt_plan(backend, tools_registry, q); elapsed = time.time()-t0
            gen_tokens = ar.total_tokens; total_tok = prompt_tokens + gen_tokens

            results.append(ar); pt_list.append(prompt_tokens); gt_list.append(gen_tokens)
            total_prompt += prompt_tokens; total_gen += gen_tokens; total_all += total_tok
            total_steps += len(ar.steps)

            tool_names = [s.tool_name for s in ar.steps if s.tool_name]
            if "plan" in tool_names: plan_calls += 1
            if tool_names and tool_names[0] == "plan": plan_first += 1
            num_cstr += sum(1 for s in ar.steps if s.is_constrained)
            num_free += sum(1 for s in ar.steps if not s.is_constrained)

            # tool_call_match_rate: compare plan expected_tools with actual tools
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

            pt = "P" if plan_first else ("p" if "plan" in tool_names else "-")
            print(f"{pt} {comp['score']:.2f} {err_type} | {total_tok}t {len(ar.steps)}s {elapsed:.1f}s")

        safe = model_id.replace("/","_"); os.makedirs(EXP_E_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(EXP_E_OUTPUT_DIR, f"{safe}.json")
        save_trajectories_batch(results, path, model_id=model_id, config_name="prompt_plan",
            task_ids=task_ids, prompt_tokens_list=pt_list, generated_tokens_list=gt_list,
            extra_meta={"experiment": "exp_e_prompt_plan"})
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

    return all_metrics


def _print_summary(m: ExpMetrics):
    print(f"\n  [{m.model_id}]")
    print(f"    plan_call={m.plan_call_rate:.0%}  plan_first={m.plan_first_rate:.0%}  completion={m.completion_rate:.0%}  success={m.success_rate:.0%}")
    print(f"    tokens: total={m.total_tokens_all}  success_tokens={m.total_success_tokens}")
    print(f"    constrained={m.num_constrained}  free={m.num_free}  match_rate={m.match_rate:.0%}  errors={m.error_counts}")


def print_report(metrics: Dict[str, ExpMetrics]):
    print(f"\n{'='*80}\n  EXPERIMENT E — FINAL REPORT: Prompt-only Plan\n{'='*80}")
    sorted_m = sorted(metrics.values(), key=lambda m: m.plan_call_rate, reverse=True)
    print(f"\n{'Model':<42s} {'Plan':>6s} {'1st':>5s} {'Compl':>6s} {'Match':>6s} {'AvgTok':>7s} {'Free':>5s}")
    print("-"*85)
    for m in sorted_m:
        print(f"{m.model_id:<42s} {m.plan_call_rate:>5.0%} {m.plan_first_rate:>5.0%} "
              f"{m.completion_rate:>5.0%} {m.match_rate:>5.0%} {m.avg_tokens:>7.0f} {m.num_free:>5d}")
    print("-"*85)
    print(f"  Key question: does the prompt make models actually call plan?")
    print(f"  Compare plan_call_rate here (prompt only) vs Exp C (constraint decoder).")
    for m in sorted_m:
        if m.error_counts:
            print(f"  {m.model_id}: errors={m.error_counts}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Experiment E — Prompt-only Plan")
    p.add_argument("--quick", action="store_true"); p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    models = [(args.model, False)] if args.model else EXP_E_MODELS
    max_t = args.max_tasks or EXP_E_MAX_TASKS
    if args.quick and max_t == 0: max_t = 5
    metrics = run_experiment(models, max_tasks=max_t)
    print_report(metrics)


if __name__ == "__main__":
    main()
