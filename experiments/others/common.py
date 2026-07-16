"""
Shared harness for the external-dataset generalization experiments.

This module parameterizes the paper's three comparison groups so a single loop
can run them on any dataset that follows the repo's ``evaluate.json`` contract:

    A. Baseline            — free generation, no plan  (== experiment_baseline)
    B. PTC-Decoder (ours)  — constrained plan + constrained execution (== exp B)
    F. Plan w/o TC-Decoder — constrained plan + free execution        (== exp F)

Everything scoring-related is reused verbatim from
``experiments.trajectory_utils`` so the metrics are identical to the main
benchmark.  The agent loop itself is reused from ``agentic.agent`` (which is
domain-agnostic).  The only per-dataset pieces are the tool pool and the
per-task ``available_tools`` subset, handled by
``experiments.others.generic_tools.registry_from_defs``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic.agent import Agent, AgentConfig, AgentResult, AgentStep
from agentic.tools import ToolRegistry, set_summary_backend
from experiments.trajectory_utils import (
    load_benchmark_tasks,
    evaluate_completion,
    categorize_error,
    load_completed_task_ids,
    save_task_incremental,
    _result_is_complete,
)
from experiments.config import (
    TEMPERATURE, MAX_TURNS, FREE_MAX_NEW_TOKENS, PLAN_MAX_NEW_TOKENS,
    REQUIRED_STEP_WEIGHT, OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD,
)
from experiments.others.generic_tools import registry_from_defs, META_TOOLS

# ---------------------------------------------------------------------------
# Prompts (domain-neutral variants of experiments/prompts.py + the Exp-B planner)
#
# The main benchmark is Chinese remote-sensing; these external benchmarks are
# English tool-calling tasks, so the language mandate is dropped and the role is
# generalized.  The *structure* (mandatory task_summary + task_done closure,
# one-tool-at-a-time) is kept identical so the methodology matches the paper.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "# Role\n"
    "You are an intelligent tool-using agent. Your task is to fulfill the user's request "
    "by invoking the tools available in the environment.\n"
    "\n"
    "# Tool Usage Guidelines\n"
    "The system provides a list of available tools through the function-calling mechanism. "
    "You should:\n"
    "1. Issue a function call to invoke the appropriate tool when an action is needed\n"
    "2. Tool execution results will be returned to you as tool-role messages\n"
    "3. Based on the returned results, continue calling other tools or provide a final answer\n"
    "\n"
    "# Limits\n"
    "- You may only call **one tool at a time**, and only interact with the outside world "
    "through function calls\n"
    "- Before completing a task, you **must call the `task_summary` tool to summarize your "
    "work** and respond to the user\n"
    "- When you believe the task is complete, call the **`task_done`** tool\n"
    "- `task_summary` and `task_done` are two functionally distinct tools, and both must be "
    "used in each task — do not confuse them\n"
    "- Strictly adhere to objective facts: do not fabricate non-existent tools or invent "
    "knowledge you do not possess"
)

USER_TASK_PROMPT = (
    "# Task\n"
    "Now, please fulfill the user's request by calling tools in the specified format. You must:\n"
    "1. Call tools to complete the user's request.\n"
    "2. If you need to output natural-language text, use `task_summary` instead.\n"
    "3. Use the `task_done` tool to end the current task when you deem it complete.\n"
    "4. Use the `task_summary` tool before ending the task to summarize the work.\n"
    "5. You can only call one tool at a time.\n"
    "\n"
    "# User query\n"
    "{query}"
)

PLANNER_SYSTEM_PROMPT = (
    "# Role\n"
    "You are a professional Task Planner. Your sole responsibility is to analyze the user's "
    "request, decompose the task, and produce a detailed, actionable step-by-step execution "
    "plan. You are only responsible for planning; never execute any tools.\n"
    "\n"
    "# Plan Structure\n"
    "Use the `plan` tool to output your plan. It must contain:\n"
    "- **title**: a brief one-line summary of the task objective\n"
    "- **steps**: a list of step objects, each with:\n"
    "  - **step_number** (integer): step index starting from 1\n"
    "  - **step_name** (string): a short name starting with a verb\n"
    "  - **description** (string): what to do, why, and the expected result\n"
    "  - **expected_tools** (array of strings): the tool names expected for this step\n"
    "\n"
    "# Plan Rules\n"
    "1. Steps must be ordered by execution sequence.\n"
    "2. expected_tools must be actual tool names that exist in the tool list. "
    "Do not invent non-existent tools.\n"
    "3. The last two steps **must** be: (penultimate) `task_summary` to summarize the work, "
    "then (final) `task_done` to end the task.\n"
    "4. Do not include the `plan` tool in expected_tools.\n"
    "5. Keep the number of steps reasonable: 2–4 for simple tasks, 5–8 for complex tasks.\n"
    "\n"
    "# Important\n"
    "- You **must** use the `<tool_call>` tag format to output the plan tool call: "
    "`<tool_call>{\"name\":\"plan\",\"arguments\":{...}}</tool_call>`\n"
    "- Do not output any text outside the `<tool_call>` tags.\n"
    "- The expected_tools will be used directly for automated execution, so tool names and "
    "ordering must be accurate."
)

# Experiment id → output subdir (mirrors the main suite's naming).
GROUP_DIRS = {
    "baseline": "exp_a_baseline",
    "ptc": "exp_b_ptc_decoder",
    "plan_wo_tc": "exp_f_plan_wo_tc_decoder",
}
GROUP_LABELS = {
    "baseline": "A. Baseline",
    "ptc": "B. PTC-Decoder (ours)",
    "plan_wo_tc": "F. Plan w/o TC-Decoder",
}
GROUP_CONFIG_NAMES = {
    "baseline": "baseline",
    "ptc": "ptc_decoder",
    "plan_wo_tc": "plan_wo_tc_decoder",
}


# ---------------------------------------------------------------------------
# Dataset spec + loading
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Locates a converted dataset on disk."""
    name: str
    data_dir: str

    @property
    def evaluate_json(self) -> str:
        return os.path.join(self.data_dir, "evaluate.json")

    @property
    def tools_json(self) -> str:
        return os.path.join(self.data_dir, "tools.json")

    def output_root(self, trajectories_root: str) -> str:
        return os.path.join(trajectories_root, self.name)


def load_tool_pool(tools_json: str) -> Dict[str, Dict[str, Any]]:
    """Load a dataset's tools.json into a ``name -> def`` mapping."""
    with open(tools_json, "r", encoding="utf-8") as f:
        defs = json.load(f)
    return {d["name"]: d for d in defs}


def domain_tools_from_gt(task: Dict[str, Any]) -> List[str]:
    """Ordered ground-truth domain tools for a task (meta-tools removed)."""
    steps = task.get("trajectory_ground_truth", {}).get("steps", [])
    out: List[str] = []
    for s in steps:
        a = s.get("action")
        if a and a not in META_TOOLS and a not in out:
            out.append(a)
    return out


def build_task_registry(
    pool: Dict[str, Dict[str, Any]],
    task: Dict[str, Any],
    *,
    benchmark_mode: bool = True,
) -> ToolRegistry:
    """Build the per-task tool registry from ``available_tools`` (or the whole pool).

    ``task['tool_responses']`` (if present) supplies dataset-provided reference
    return strings that the simulated executor echoes back, so dependency-carrying
    tasks (e.g. API-Bank auth-then-act) see realistic tool output.
    """
    available = task.get("available_tools")
    return registry_from_defs(
        pool, available, benchmark_mode=benchmark_mode,
        canned_responses=task.get("tool_responses"),
    )


# ---------------------------------------------------------------------------
# Planning subtask (neutral-prompt port of experiment_separate_plan)
# ---------------------------------------------------------------------------

def run_planning_subtask(backend, tools_registry: ToolRegistry, user_query: str) -> dict:
    """Dedicated planning subtask — constrained decoder forces a plan tool call.

    Mirrors ``experiments.experiment_separate_plan.run_planning_subtask`` but with
    the domain-neutral :data:`PLANNER_SYSTEM_PROMPT`.  Retries once with more
    tokens if the plan JSON is truncated.
    """
    td = tools_registry.get_definitions()
    ps = tools_registry.get_schema("plan")
    msg = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    prompt = backend.build_prompt(msg, td)

    dr = None
    for attempt, max_tok in enumerate((PLAN_MAX_NEW_TOKENS, PLAN_MAX_NEW_TOKENS * 2), 1):
        dr = backend.generate_constrained(
            prompt=prompt, tool_name="plan", args_schema=ps,
            max_new_tokens=max_tok, temperature=TEMPERATURE,
        )
        if "_parse_error" not in dr.tool_call:
            break

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


# ---------------------------------------------------------------------------
# Group runners — each returns (AgentResult, prompt_tokens, gen_tokens, extra)
# ---------------------------------------------------------------------------

def _run_baseline(backend, registry: ToolRegistry, query: str) -> Tuple[AgentResult, int, int, dict]:
    """Bare free-generation agent loop (no plan). Port of experiment_baseline."""
    start = time.time()
    steps: List[AgentStep] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK_PROMPT.format(query=query)},
    ]
    tool_defs = registry.get_definitions()
    prompt_tokens = len(backend.tokenizer.encode(backend.build_prompt(messages, tool_defs)))
    gen_tokens = 0

    result = AgentResult(user_query=query)
    for turn in range(1, MAX_TURNS + 1):
        prompt = backend.build_prompt(messages, tool_defs)
        raw = backend.generate_free(prompt, max_new_tokens=FREE_MAX_NEW_TOKENS, temperature=TEMPERATURE)
        gen_tokens += len(backend.tokenizer.encode(raw))

        tc = backend.parse_tool_call(raw)
        if tc and "name" in tc:
            t_name = tc["name"]; t_args = tc.get("arguments", {})
            step = AgentStep(step_index=turn, tool_name=t_name, tool_args=t_args,
                             tool_result=None, generated_text=raw, is_constrained=False,
                             elapsed=time.time() - start)
            if registry.get(t_name):
                step.tool_result = registry.execute(t_name, t_args)
            else:
                steps.append(step)
                result.steps = steps; result.final_answer = raw
                result.success = True; result.total_time = time.time() - start
                return result, prompt_tokens, gen_tokens, {}
            steps.append(step)
            tcj = json.dumps({"name": t_name, "arguments": t_args}, ensure_ascii=False)
            messages.append({"role": "assistant",
                             "content": f"{backend.tool_call_prefix}{tcj}{backend.tool_call_suffix}"})
            messages.append({"role": "tool", "content": step.tool_result or ""})
            if t_name in ("task_done", "wait_user_instruction"):
                result.steps = steps; result.final_answer = step.tool_result
                result.success = True
                result.error = "wait_user_instruction" if t_name == "wait_user_instruction" else None
                result.total_time = time.time() - start
                return result, prompt_tokens, gen_tokens, {}
        else:
            steps.append(AgentStep(step_index=turn, tool_name=None, tool_args=None,
                                   tool_result=None, generated_text=raw, is_constrained=False,
                                   elapsed=time.time() - start))
            result.steps = steps; result.final_answer = raw
            result.success = True; result.total_time = time.time() - start
            return result, prompt_tokens, gen_tokens, {}

    result.steps = steps; result.success = False; result.error = "Max turns reached"
    result.total_time = time.time() - start
    return result, prompt_tokens, gen_tokens, {}


def _run_plan_then_act(
    backend, registry: ToolRegistry, query: str, *, constrained_exec: bool,
) -> Tuple[AgentResult, int, int, dict]:
    """Planning subtask + Agent execution (constrained or free).

    ``constrained_exec=True``  → PTC-Decoder (Exp B).
    ``constrained_exec=False`` → Plan w/o TC-Decoder (Exp F).
    """
    tool_defs = registry.get_definitions()

    plan_res = run_planning_subtask(backend, registry, query)
    planning_record = {
        "prompt_tokens": plan_res["prompt_tokens"],
        "generated_tokens": plan_res["generated_tokens"],
        "is_constrained": plan_res["is_constrained"],
        "parsed_successfully": plan_res.get("tool_result") is not None,
        "generated_text": plan_res["generated_text"],
    }

    exec_msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK_PROMPT.format(query=query)},
    ]
    exec_pt = len(backend.tokenizer.encode(backend.build_prompt(exec_msg, tool_defs)))

    econf = AgentConfig(max_turns=MAX_TURNS, temperature=TEMPERATURE,
                        use_constrained_decoder=constrained_exec, verbose=False,
                        system_prompt=SYSTEM_PROMPT)
    executor = Agent(backend, registry, econf)
    ar = executor.run(USER_TASK_PROMPT.format(query=query), pre_seeded_plan=plan_res)
    ar.user_query = query

    gen_tokens = sum(len(backend.tokenizer.encode(s.generated_text)) for s in ar.steps)
    prompt_tokens = plan_res["prompt_tokens"] + exec_pt
    return ar, prompt_tokens, gen_tokens, {"planning_subtask": planning_record}


GROUP_RUNNERS: Dict[str, Callable] = {
    "baseline": lambda b, r, q: _run_baseline(b, r, q),
    "ptc": lambda b, r, q: _run_plan_then_act(b, r, q, constrained_exec=True),
    "plan_wo_tc": lambda b, r, q: _run_plan_then_act(b, r, q, constrained_exec=False),
}


# ---------------------------------------------------------------------------
# Per-task evaluation extras (domain-only completeness alongside the standard one)
# ---------------------------------------------------------------------------

def evaluate_domain_only(agent_steps: list, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Step completeness restricted to domain tools (excludes the meta-tools).

    Isolates the effect on *real* dataset-tool selection, since task_summary /
    task_done / plan are method-provided closure tools present in the GT.

    Additionally computes domain-level **Precision** and **F1** (Seal-Tools
    style) to measure whether the agent called irrelevant tools (FP).
    """
    called = [s.tool_name for s in agent_steps if s.tool_name]
    steps = ground_truth.get("steps", [])
    required = [s["action"] for s in steps
                if s.get("required") and s["action"] not in META_TOOLS]
    optional = [s["action"] for s in steps
                if not s.get("required") and s["action"] not in META_TOOLS]

    req_called = sum(1 for a in required if a in called)
    opt_called = sum(1 for a in optional if a in called)
    rec = round(req_called / len(required), 3) if required else 1.0

    # Precision/F1: domain tools agent called vs. all domain tools in GT
    gt_all = set(required) | set(optional)
    agent_domain = set(t for t in called if t not in META_TOOLS)
    tp = len(agent_domain & gt_all)
    fp = len(agent_domain - gt_all)
    prec = round(tp / max(tp + fp, 1), 3)
    f1 = round(2 * prec * rec / max(prec + rec, 1e-9), 3)

    return {
        "domain_required_called": req_called,
        "domain_required_total": len(required),
        "domain_optional_called": opt_called,
        "domain_optional_total": len(optional),
        "domain_recall": rec,
        "domain_precision": prec,
        "domain_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main group loop
# ---------------------------------------------------------------------------

def run_group(
    spec: DatasetSpec,
    group: str,
    models: List[Any],
    backend_factory: Callable[[str, bool], Any],
    trajectories_root: str,
    *,
    max_tasks: int = 0,
    task_indices: Optional[List[int]] = None,
    skip_existing: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run one comparison group over ``models`` on ``spec``.

    Args:
        spec: dataset locator.
        group: one of ``baseline`` / ``ptc`` / ``plan_wo_tc``.
        models: list of ``(model_id, quantize)`` entries (or bare ids).
        backend_factory: ``(model_id, quantize) -> backend`` (real or mock).
        trajectories_root: root dir for saved trajectories.
        max_tasks / task_indices: task subsetting.
        skip_existing: resume — skip tasks already saved for a model.

    Returns:
        ``{model_id: {completion_rate, success_rate, domain_recall, ...}}``.
    """
    if group not in GROUP_RUNNERS:
        raise ValueError(f"unknown group {group!r}; expected one of {list(GROUP_RUNNERS)}")

    tasks = load_benchmark_tasks(spec.evaluate_json, task_indices, max_tasks)
    pool = load_tool_pool(spec.tools_json)
    out_dir = os.path.join(spec.output_root(trajectories_root), GROUP_DIRS[group])
    runner = GROUP_RUNNERS[group]
    cfg_name = GROUP_CONFIG_NAMES[group]

    print(f"\n{'='*72}\n  {spec.name.upper()}  |  {GROUP_LABELS[group]}"
          f"  |  models={len(models)}  tasks={len(tasks)}\n{'='*72}")

    all_metrics: Dict[str, Dict[str, Any]] = {}

    for entry in models:
        model_id = entry[0] if isinstance(entry, (tuple, list)) else entry
        quantize = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else False

        completed_ids = load_completed_task_ids(out_dir, model_id) if skip_existing else set()
        if skip_existing and _result_is_complete(out_dir, model_id, len(tasks)):
            print(f"  [{model_id}] all {len(tasks)} tasks complete — skip")
            all_metrics[model_id] = _summarize_from_disk(out_dir, model_id, len(tasks))
            continue

        print(f"\n{'-'*60}\n  Model: {model_id}{' (resuming)' if completed_ids else ''}\n{'-'*60}")
        backend = backend_factory(model_id, quantize)

        agg = _MetricAcc(len(tasks))
        for i, task in enumerate(tasks):
            tid = f"task_{i:03d}"
            q = task["question"]
            gt_truth = task.get("trajectory_ground_truth", {})

            if tid in completed_ids:
                print(f"  [{i+1}/{len(tasks)}] (cached) {q[:60]}...")
                agg.merge_cached(out_dir, model_id, tid)
                continue

            print(f"  [{i+1}/{len(tasks)}] {q[:60]}...", end=" ", flush=True)

            registry = build_task_registry(pool, task)
            if hasattr(backend, "set_task_context"):
                backend.set_task_context(domain_tools_from_gt(task))

            t0 = time.time()
            ar, prompt_tokens, gen_tokens, extra = runner(backend, registry, q)
            elapsed = time.time() - t0

            comp = evaluate_completion(ar.steps, gt_truth, REQUIRED_STEP_WEIGHT,
                                       OPTIONAL_STEP_WEIGHT, TASK_DONE_WEIGHT, COMPLETION_THRESHOLD)
            dom = evaluate_domain_only(ar.steps, gt_truth)
            err_type = categorize_error(ar)
            tool_names = [s.tool_name for s in ar.steps if s.tool_name]

            eval_meta = {
                "completion_score": comp["score"],
                "completed": comp["completed"],
                "required_called": comp["required_called"],
                "required_total": comp["required_total"],
                "task_done_called": comp["task_done_called"],
                "error_type": err_type,
                "called_tools": tool_names,
                "num_constrained": sum(1 for s in ar.steps if s.is_constrained),
                "num_free": sum(1 for s in ar.steps if not s.is_constrained),
                **dom,
            }
            agg.add(comp, dom, ar, err_type, prompt_tokens + gen_tokens)

            extra_record = {"eval": eval_meta}
            extra_record.update(extra)
            save_task_incremental(
                ar, out_dir, model_id, config_name=cfg_name, task_id=tid,
                prompt_tokens=prompt_tokens, generated_tokens=gen_tokens,
                extra_meta={"experiment": GROUP_DIRS[group], "dataset": spec.name, "group": group},
                extra_record_fields=extra_record,
            )
            completed_ids.add(tid)
            print(f"score={comp['score']:.2f} dom={dom['domain_recall']:.2f} {err_type} "
                  f"| {len(ar.steps)}s {elapsed:.2f}s")

        metrics = agg.finalize(model_id)
        all_metrics[model_id] = metrics
        _print_model_summary(model_id, metrics)

        # ---- Release GPU memory before next model ---------------------------
        # Must be INLINE (not a helper function) so ``del backend`` drops the
        # same-scope variable.  Also needs ``gc.collect()`` to break potential
        # circular references (e.g. constrained_decoder → model) before
        # ``torch.cuda.empty_cache()`` can actually reclaim GPU memory.
        set_summary_backend(None)
        del backend
        backend = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch  # noqa
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # -----------------------------------------------------------------

    return all_metrics


# ---------------------------------------------------------------------------
# Metric accumulation
# ---------------------------------------------------------------------------

class _MetricAcc:
    def __init__(self, n_tasks: int) -> None:
        self.n = n_tasks
        self.completed = 0
        self.success = 0
        self.dom_recall_sum = 0.0
        self.score_sum = 0.0
        self.plan_calls = 0
        self.num_constrained = 0
        self.num_free = 0
        self.total_tokens = 0
        self.error_counts: Dict[str, int] = {}
        self.counted = 0

    def add(self, comp, dom, ar, err_type, tokens) -> None:
        self.counted += 1
        if comp["completed"]:
            self.completed += 1
        if ar.success:
            self.success += 1
        self.dom_recall_sum += dom["domain_recall"]
        self.score_sum += comp["score"]
        names = [s.tool_name for s in ar.steps if s.tool_name]
        if "plan" in names:
            self.plan_calls += 1
        self.num_constrained += sum(1 for s in ar.steps if s.is_constrained)
        self.num_free += sum(1 for s in ar.steps if not s.is_constrained)
        self.total_tokens += tokens
        self.error_counts[err_type] = self.error_counts.get(err_type, 0) + 1

    def merge_cached(self, out_dir, model_id, tid) -> None:
        """Fold a cached task's saved eval back into the accumulator."""
        rec = _read_task_record(out_dir, model_id, tid)
        if not rec:
            return
        ev = rec.get("eval", {})
        self.counted += 1
        if ev.get("completed"):
            self.completed += 1
        if rec.get("success"):
            self.success += 1
        self.dom_recall_sum += ev.get("domain_recall", 0.0)
        self.score_sum += ev.get("completion_score", 0.0)
        names = ev.get("called_tools", [])
        if "plan" in names:
            self.plan_calls += 1
        self.total_tokens += rec.get("total_tokens", 0)
        et = ev.get("error_type", "unknown")
        self.error_counts[et] = self.error_counts.get(et, 0) + 1

    def finalize(self, model_id) -> Dict[str, Any]:
        d = max(self.counted, 1)
        return {
            "model_id": model_id,
            "num_tasks": self.counted,
            "completion_rate": round(self.completed / d, 4),
            "success_rate": round(self.success / d, 4),
            "avg_completion_score": round(self.score_sum / d, 4),
            "avg_domain_recall": round(self.dom_recall_sum / d, 4),
            "plan_call_rate": round(self.plan_calls / d, 4),
            "num_constrained": self.num_constrained,
            "num_free": self.num_free,
            "avg_tokens": round(self.total_tokens / d, 1),
            "error_counts": self.error_counts,
        }


def _summarize_from_disk(out_dir, model_id, n_tasks) -> Dict[str, Any]:
    acc = _MetricAcc(n_tasks)
    for i in range(n_tasks):
        acc.merge_cached(out_dir, model_id, f"task_{i:03d}")
    return acc.finalize(model_id)


def _read_task_record(out_dir, model_id, tid) -> Optional[dict]:
    path = os.path.join(out_dir, f"{model_id.replace('/', '_')}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    for r in data:
        if r.get("type") != "_summary" and r.get("task_id") == tid:
            return r
    return None


def _print_model_summary(model_id, m: Dict[str, Any]) -> None:
    print(f"\n  [{model_id}]  completion={m['completion_rate']:.0%}  "
          f"success={m['success_rate']:.0%}  domain_recall={m['avg_domain_recall']:.0%}  "
          f"plan_call={m['plan_call_rate']:.0%}")
    print(f"    constrained={m['num_constrained']}  free={m['num_free']}  "
          f"avg_tokens={m['avg_tokens']}  errors={m['error_counts']}")


# ---------------------------------------------------------------------------
# NOTE: GPU cleanup is done INLINE inside ``run_group``, not via a helper
# function, because ``del backend`` must drop the same-scope variable so
# ``torch.cuda.empty_cache()`` sees the freed model.  A helper function would
# only delete its own parameter binding, leaving the caller's reference alive.
# See the ``# ---- Release GPU memory before next model ----`` block above.
# ---------------------------------------------------------------------------
