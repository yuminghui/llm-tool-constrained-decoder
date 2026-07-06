"""
Experiment B — Separate Plan + Agent
======================================

Phase 1: dedicated planning subtask with its own system prompt.
         Constraint decoder forces a ``plan`` tool call.
Phase 2: execution agent with pre-seeded plan (normal agent run).

Metrics: tokens, completion, trajectory, plan usage, constrained/free calls,
         error types, tool_call_match_rate (plan vs actual).

Usage::

    python -m experiments.experiment_separate_plan
    python -m experiments.experiment_separate_plan --quick --max-tasks 5
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
    load_benchmark_tasks,
    evaluate_completion, categorize_error,
    load_completed_task_ids, save_task_incremental,
    _result_is_complete,
)
from experiments.config import (
    TOOLS_JSON_PATH, EVALUATE_JSON_PATH,
    EXP_B_MODELS, EXP_B_TASK_INDICES, EXP_B_MAX_TASKS, EXP_B_OUTPUT_DIR,
    QUANTIZATION_MODE, model_id as _mid, model_quantize,
    TEMPERATURE, MAX_TURNS, PLAN_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD,
)
from experiments.prompts import SYSTEM_PROMPT, USER_TASK_PROMPT

PLANNER_SYSTEM_PROMPT = (
    "# Role\n"
    "You are a professional Task Planner. Your sole responsibility is to analyze the user's "
    "request, decompose the task, and produce a detailed, clear, and actionable step-by-step "
    "execution plan.\n"
    "You are **only responsible for planning**. You must never execute any tools or simulate "
    "tool execution results.\n"
    "\n"
    "# Plan Structure\n"
    "You must use the `plan` tool to output your plan. The plan must contain the following fields:\n"
    "- **title**: A brief one-line summary of the task objective\n"
    "- **steps**: A list of step objects, each containing:\n"
    "  - **step_number** (integer): Step index, starting from 1\n"
    "  - **step_name** (string): A short name for the step, starting with a verb "
    "(e.g. \"Retrieve data\", \"Execute preprocessing\")\n"
    "  - **description** (string): A detailed description of the step — what to do, why, "
    "and what result is expected\n"
    "  - **expected_tools** (array of strings): The list of tool names expected for this step. "
    "If a single tool is expected, use [\"tool_name\"]; if multiple tools are expected (called "
    "in order), use [\"tool_a\", \"tool_b\"]\n"
    "\n"
    "# Plan Rules\n"
    "1. Steps must be ordered by execution sequence. When the output of one step is the input "
    "to the next, describe the dependency in the description\n"
    "2. The expected_tools for each step must be actual tool names that exist in the tool list. "
    "Do not invent non-existent tools\n"
    "3. The last two steps **must** be:\n"
    "   - Penultimate step: use `task_summary` to summarize all completed work, "
    "including what was done, the results, and whether user requirements were met\n"
    "   - Final step: use `task_done` to end the task\n"
    "4. Do not include the `plan` tool in expected_tools — the planning phase is already complete\n"
    "5. Keep the number of steps reasonable: 2–4 steps for simple tasks, 5–8 steps for complex tasks\n"
    "\n"
    "# Example\n"
    "Suppose the user says \"process the image to the atmospheric correction stage\", "
    "you should output something like:\n"
    "```\n"
    "<tool_call>\n"
    "{\"name\":\"plan\",\"arguments\":{\n"
    "  \"title\": \"Gaofen Image Atmospheric Correction Processing\",\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"step_number\": 1,\n"
    "      \"step_name\": \"Execute atmospheric correction preprocessing\",\n"
    "      \"description\": \"Call the preprocessing tool to perform atmospheric correction "
    "on the input image. The case parameter selects the atmospheric correction option\",\n"
    "      \"expected_tools\": [\"gf_pms_preprocess_cli\"]\n"
    "    },\n"
    "    {\n"
    "      \"step_number\": 2,\n"
    "      \"step_name\": \"Summarize the work\",\n"
    "      \"description\": \"Compile the processing results: which images were processed, "
    "what parameters were used, and where the output is located\",\n"
    "      \"expected_tools\": [\"task_summary\"]\n"
    "    },\n"
    "    {\n"
    "      \"step_number\": 3,\n"
    "      \"step_name\": \"End the task\",\n"
    "      \"description\": \"Confirm task completion and call task_done to finish\",\n"
    "      \"expected_tools\": [\"task_done\"]\n"
    "    }\n"
    "  ]\n"
    "}}\n"
    "</tool_call>\n"
    "```\n"
    "\n"
    "# Important\n"
    "- Please respond in Chinese\n"
    "- **You must** use the `<tool_call>` tag format to output the plan tool call, "
    "with the format `<tool_call>{\"name\":\"plan\",\"arguments\":{...}}</tool_call>`\n"
    "- Do not output any text, explanations, or code block markers outside "
    "the `<tool_call>` tags\n"
    "- The expected_tools in the plan will be used directly for subsequent automated "
    "execution, so the tool names and ordering must be accurate"
)


def run_planning_subtask(backend, tools_registry, user_query: str) -> dict:
    """Dedicated planning subtask — constrained decoder forces plan tool call.

    Retries with more tokens if the plan JSON is truncated on the first attempt.
    """
    td = tools_registry.get_definitions()
    ps = tools_registry.get_schema("plan")
    msg = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    prompt = backend.build_prompt(msg, td)

    for attempt, max_tok in enumerate((PLAN_MAX_NEW_TOKENS, PLAN_MAX_NEW_TOKENS * 2), 1):
        dr = backend.generate_constrained(
            prompt=prompt, tool_name="plan", args_schema=ps,
            max_new_tokens=max_tok, temperature=TEMPERATURE,
        )
        if "_parse_error" not in dr.tool_call:
            break  # success
        if attempt == 1:
            print(f"    [plan subtask] truncated at {max_tok} tokens, retrying with {max_tok*2}...")

    pt = len(backend.tokenizer.encode(prompt))
    gt = len(dr.token_ids)
    pa = dr.tool_call.get("arguments", {})
    tr = tools_registry.execute("plan", pa) if "_parse_error" not in dr.tool_call else None
    return {
        "tool_name": "plan",
        "tool_args": pa,
        "generated_text": dr.text,
        "tool_result": tr,
        "prompt_tokens": pt,
        "generated_tokens": gt,
        "total_tokens": pt + gt,
        "is_constrained": True,
    }


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


def run_experiment(models: list, task_indices=None, max_tasks=0, skip_existing: bool = False) -> Dict[str, ExpMetrics]:
    tasks = load_benchmark_tasks(EVALUATE_JSON_PATH, task_indices, max_tasks)
    tools_registry = load_tools_from_json(TOOLS_JSON_PATH, benchmark_mode=True)
    all_metrics: Dict[str, ExpMetrics] = {}
    tool_defs = tools_registry.get_definitions()

    print(f"\n{'='*70}\n  EXPERIMENT B — Separate Plan + Agent\n  Models: {len(models)}  Tasks: {len(tasks)}\n{'='*70}")

    for entry in models:
        model_id = _mid(entry); quantize = model_quantize(entry)

        # ---- checkpoint / resume -------------------------------------------
        completed_ids = load_completed_task_ids(EXP_B_OUTPUT_DIR, model_id) if skip_existing else set()
        if skip_existing and _result_is_complete(EXP_B_OUTPUT_DIR, model_id, len(tasks)):
            print(f"\n  [{model_id}] All {len(tasks)} tasks complete — skip")
            continue
        resumed = len(completed_ids) > 0
        print(f"\n{'─'*60}\n  Model: {model_id}{' (resuming)' if resumed else ''}\n{'─'*60}")
        backend = LLMBackend(model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE)

        results, pt_list, gt_list, task_ids = [], [], [], []
        plan_calls, plan_first, num_cstr, num_free, completed, num_ok = 0, 0, 0, 0, 0, 0
        total_prompt, total_gen, total_all, total_ok_tokens, total_steps = 0, 0, 0, 0, 0
        match_total, match_hits = 0, 0
        error_counts: Dict[str, int] = {}
        econf = AgentConfig(max_turns=MAX_TURNS, temperature=TEMPERATURE, use_constrained_decoder=True, verbose=False, system_prompt=SYSTEM_PROMPT)

        for i, task in enumerate(tasks):
            q = task["question"]; gt_truth = task.get("trajectory_ground_truth", {})
            tid = f"task_{i:03d}"

            if tid in completed_ids:
                print(f"  [{i+1}/{len(tasks)}] (cached) {q[:70]}...")
                continue

            print(f"  [{i+1}/{len(tasks)}] {q[:70]}...", end=" ", flush=True)

            # Phase 1: planning subtask
            plan_res = run_planning_subtask(backend, tools_registry, q)
            planning_record = {
                "prompt_tokens": plan_res["prompt_tokens"],
                "generated_tokens": plan_res["generated_tokens"],
                "is_constrained": plan_res["is_constrained"],
                "parsed_successfully": plan_res.get("tool_result") is not None,
                "generated_text": plan_res["generated_text"],
            }

            # Phase 2: execution
            exec_msg = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_TASK_PROMPT.format(query=q)}]
            exec_pt = len(backend.tokenizer.encode(backend.build_prompt(exec_msg, tool_defs)))

            t0 = time.time()
            executor = Agent(backend, tools_registry, econf)
            ar = executor.run(USER_TASK_PROMPT.format(query=q), pre_seeded_plan=plan_res)
            ar.user_query = q  # record original question, not the augmented prompt
            elapsed = time.time()-t0

            gen_tokens = sum(len(backend.tokenizer.encode(s.generated_text)) for s in ar.steps)
            prompt_tokens = plan_res["prompt_tokens"] + exec_pt
            total_tok = prompt_tokens + gen_tokens

            results.append(ar); pt_list.append(prompt_tokens); gt_list.append(gen_tokens)
            total_prompt += prompt_tokens; total_gen += gen_tokens; total_all += total_tok
            total_steps += len(ar.steps)

            tool_names = [s.tool_name for s in ar.steps if s.tool_name]
            # Plan is in separate context, but we track the executor's plan calls
            if "plan" in tool_names: plan_calls += 1
            if tool_names and tool_names[0] == "plan": plan_first += 1
            num_cstr += sum(1 for s in ar.steps if s.is_constrained)
            num_free += sum(1 for s in ar.steps if not s.is_constrained)

            # tool_call_match_rate: compare plan expected_tools vs actual tools
            expected = set()
            for s in (plan_res.get("tool_args", {}).get("steps", []) or []):
                if isinstance(s, dict) and s.get("expected_tools"):
                    for t in s["expected_tools"]:
                        if isinstance(t, str) and t not in ("task_done", "task_summary"):
                            expected.add(t)
            actual = set(t for t in tool_names if t and t not in ("plan", "task_done", "task_summary"))
            if expected:
                match_total += len(expected); match_hits += len(expected & actual)

            comp = evaluate_completion(ar.steps, gt_truth, REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD)
            if comp["completed"]: completed += 1
            err_type = categorize_error(ar)
            error_counts[err_type] = error_counts.get(err_type, 0) + 1
            if ar.success: num_ok += 1; total_ok_tokens += total_tok

            print(f"{comp['score']:.2f} {err_type} | {total_tok}t ({plan_res['total_tokens']}+{exec_pt+gen_tokens}) | {len(ar.steps)}s {elapsed:.1f}s")

            # ---- incremental save (includes planning_subtask) --------------
            save_task_incremental(ar, EXP_B_OUTPUT_DIR, model_id, config_name="separate_plan",
                task_id=tid, prompt_tokens=prompt_tokens, generated_tokens=gen_tokens,
                extra_meta={"experiment": "exp_b_separate_plan"},
                extra_record_fields={"planning_subtask": planning_record})
            completed_ids.add(tid)

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
    print(f"\n{'='*80}\n  EXPERIMENT B — FINAL REPORT: Separate Plan\n{'='*80}")
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
    p = argparse.ArgumentParser(description="Experiment B — Separate Plan")
    p.add_argument("--quick", action="store_true"); p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--quantize", action="store_true", default=False,
                   help="Enable 4bit quantization for --model")
    args = p.parse_args()
    models = [(args.model, args.quantize)] if args.model else EXP_B_MODELS
    max_t = args.max_tasks or EXP_B_MAX_TASKS
    if args.quick and max_t == 0: max_t = 5
    skip_existing = args.model is None and args.max_tasks == 0 and not args.quick
    metrics = run_experiment(models, max_tasks=max_t, skip_existing=skip_existing)
    print_report(metrics)


if __name__ == "__main__":
    main()
