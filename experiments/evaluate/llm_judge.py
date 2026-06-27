"""
LLM-as-Judge trajectory evaluator.

Evaluates agent execution trajectories against the benchmark (``evaluate.json``)
using an OpenAI-compatible LLM API.  Produces per-task dimension scores
(0--5) plus an overall integer rating, with reasoning in Chinese.

Usage::

    # Single trajectory file
    python -m experiments.evaluate.llm_judge \\
        -i experiments/trajectories/exp_a_baseline/Qwen_Qwen3-0.6B.json

    # All trajectory files in a directory
    python -m experiments.evaluate.llm_judge \\
        -i experiments/trajectories/exp_a_baseline/

    # Custom provider (proxy, vLLM, etc.)
    python -m experiments.evaluate.llm_judge \\
        -i experiments/trajectories/exp_a_baseline/ \\
        -o results/judge_scores/ \\
        --model gpt-4.1-mini \\
        --base-url https://your-proxy.example.com/v1

Environment::

    OPENAI_API_KEY   required — your API key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVALUATE_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "evaluate.json")

DEFAULT_MODEL = "gpt-4.1-mini"

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for AI agent trajectories in a remote-sensing \
satellite image processing environment.  The agent has access to tools \
(preprocessing, change detection, file search, etc.) and must call them \
in a sensible order to fulfil a user request.

You will receive:
- The user's question
- An expected answer / outcome
- A list of expected tool-calling steps (with required / optional labels)
- The agent's actual execution trace (each step: tool name, arguments, result)
- The agent's final output (if any)
- Auxiliary metadata (success flag, error type, step counts)

Evaluate the agent's performance on the four dimensions below.  Each \
dimension is an integer 0--5 (0 = worst, 5 = best).

## Dimensions

1. **step_completeness** — Coverage of required steps.
   5 = all required steps executed, plus most optional steps that were useful.
   3 = required steps mostly done, but one missing or incomplete.
   1 = several required steps missing.
   0 = almost no meaningful steps executed.

2. **result_accuracy** — How well the final answer / task_summary matches \
the expected outcome.
   5 = accurate, detailed, fully addresses the user's request.
   3 = roughly correct but missing detail or slightly wrong.
   1 = largely incorrect or irrelevant.
   0 = no useful final answer.

3. **flow_reasonableness** — Logical ordering, absence of redundant or \
nonsensical tool calls.
   5 = optimal order, no wasted steps.
   3 = mostly reasonable with minor inefficiencies (e.g. repeated calls).
   1 = chaotic or clearly wrong order.
   0 = completely random / broken flow.

4. **robustness** — Error handling and recovery.
   5 = encountered no errors, OR recovered gracefully when errors occurred.
   3 = some errors but managed to continue.
   1 = errors derailed the task.
   0 = crashed / exception.

## Overall score

overall = round(mean of the four dimension scores), integer 0--5.

## Output format

You MUST respond with a single JSON object (no markdown fences, no extra text):
{"step_completeness": int, "result_accuracy": int, "flow_reasonableness": int,
 "robustness": int, "overall": int,
 "reasoning": "concise Chinese reasoning (2-4 sentences)"}
"""


def _build_user_prompt(
    question: str,
    expected: str,
    gt_steps: List[Dict[str, Any]],
    trajectory_steps: List[Dict[str, Any]],
    final_answer: Optional[str],
    success: bool,
    error: Optional[str],
    total_steps: int,
) -> str:
    """Build the user message for the judge LLM."""

    # Format expected steps
    gt_lines = []
    for s in gt_steps:
        label = "required" if s.get("required") else "optional"
        gt_lines.append(f"  - [{label}] {s['action']}")
    gt_text = "\n".join(gt_lines) if gt_lines else "(none)"

    # Format actual trajectory
    traj_lines = []
    for s in trajectory_steps:
        action = s.get("action", "?")
        args = s.get("args")
        result = s.get("result", "")
        is_cstr = s.get("is_constrained", False)
        cstr_tag = "[constrained]" if is_cstr else "[free]"
        line = f"  {cstr_tag} {action}"
        if args:
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            line += f"  args={args_str}"
        if result:
            result_str = str(result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            line += f"\n    → {result_str}"
        traj_lines.append(line)
    traj_text = "\n".join(traj_lines) if traj_lines else "(empty)"

    fa = final_answer or "(no final answer)"

    prompt = f"""## 用户问题
{question}

## 预期答案
{expected}

## 预期工具步骤
{gt_text}

## Agent 实际执行轨迹
{traj_text}

## Agent 最终输出
{fa}

## 辅助信息
- success: {success}
- error: {error or 'none'}
- 总步数: {total_steps}"""
    return prompt


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _call_judge_llm(
    client,
    system: str,
    user: str,
    model: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Call OpenAI-compatible API and return parsed JSON response, or None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content or ""

            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            print(f"  [ERR] API call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)

    return None


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

@dataclass
class TaskJudgeResult:
    task_id: str
    question: str
    step_completeness: int
    result_accuracy: int
    flow_reasonableness: int
    robustness: int
    overall: int
    reasoning: str


@dataclass
class JudgeSummary:
    experiment: str
    model_id: str
    total_tasks: int
    evaluated_tasks: int
    skipped_tasks: int
    avg_step_completeness: float
    avg_result_accuracy: float
    avg_flow_reasonableness: float
    avg_robustness: float
    avg_overall: float
    tasks: List[Dict[str, Any]] = field(default_factory=list)


def _judge_output_path(trajectory_path: str, output_dir: str) -> str:
    """Derive the judge-scores output path from a trajectory file."""
    base = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join(output_dir, f"{base}_judge_scores.json")


def _load_completed_judge_ids(output_path: str) -> set:
    """Return the set of already-judged task IDs from a judge scores file."""
    if not os.path.isfile(output_path):
        return set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {t.get("task_id", "") for t in data.get("tasks", [])}
    except Exception:
        return set()


def _save_judge_incremental(summary: JudgeSummary, output_path: str) -> None:
    """Write (or update) judge scores file with current results."""
    write_judge_output(summary, output_path)


def evaluate_trajectory_file(
    trajectory_path: str,
    benchmark: Dict[str, Dict[str, Any]],
    client,
    model: str,
    output_dir: str = "",
) -> Optional[JudgeSummary]:
    """Evaluate a single trajectory JSON file against the benchmark.

    Supports checkpoint/resume: if the judge-scores file already exists,
    already-scored tasks are skipped and evaluation resumes from where
    it left off.  Results are saved after every task.

    Returns a JudgeSummary, or None if the file cannot be read.
    """
    # --- Load trajectory file ---
    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERR] Cannot read {trajectory_path}: {e}")
        return None

    if not isinstance(data, list):
        print(f"  [ERR] Expected JSON array in {trajectory_path}")
        return None

    records = [r for r in data if r.get("type") != "_summary"]
    if not records:
        print(f"  [WARN] No trajectory records found in {trajectory_path}")
        return None

    first = records[0]
    experiment = first.get("experiment", first.get("config", "unknown"))
    model_id = first.get("model_id", "unknown")
    total = len(records)
    print(f"  Records: {total}  Experiment: {experiment}  Model: {model_id}")

    # ---- checkpoint / resume -------------------------------------------
    judge_path = _judge_output_path(trajectory_path, output_dir)
    completed_ids = _load_completed_judge_ids(judge_path)
    if completed_ids and len(completed_ids) >= total:
        print(f"  All {total} tasks already judged — skip")
        return None
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)}/{total} already judged")

    # --- Evaluate each task ---
    # Load existing results to accumulate
    existing_tasks: List[Dict[str, Any]] = []
    try:
        if os.path.isfile(judge_path):
            with open(judge_path, "r", encoding="utf-8") as f:
                existing_tasks = json.load(f).get("tasks", [])
    except Exception:
        pass

    results: List[TaskJudgeResult] = []
    skipped = 0

    for i, record in enumerate(records):
        question = record.get("question", "")
        tid = record.get("task_id", f"task_{i:03d}")

        if tid in completed_ids:
            print(f"  [{i+1}/{total}] {tid} (cached)")
            continue

        # Match with benchmark entry
        bench_entry = benchmark.get(question)
        if bench_entry is None:
            print(f"  [{i+1}/{total}] {tid} — no benchmark match, skip")
            skipped += 1
            continue

        # Skip if trajectory_evaluate is false
        if not bench_entry.get("trajectory_evaluate", True):
            continue

        expected = bench_entry.get("expected", "")
        gt_steps = bench_entry.get("trajectory_ground_truth", {}).get("steps", [])
        traj_steps = record.get("trajectory", {}).get("steps", [])
        final_answer = record.get("final_answer")
        success = record.get("success", False)
        error = record.get("error")

        # Build prompt
        user_prompt = _build_user_prompt(
            question=question,
            expected=expected,
            gt_steps=gt_steps,
            trajectory_steps=traj_steps,
            final_answer=final_answer,
            success=success,
            error=error,
            total_steps=len(traj_steps),
        )

        # Call judge
        print(f"  [{i+1}/{total}] {tid}  ", end="", flush=True)
        parsed = _call_judge_llm(client, JUDGE_SYSTEM_PROMPT, user_prompt, model)

        if parsed is None:
            print("FAILED — using fallback scores")
            parsed = _fallback_score(record, bench_entry)
            parsed["reasoning"] = "[Fallback] API call failed; heuristic score."

        # Validate scores
        for dim in ("step_completeness", "result_accuracy",
                     "flow_reasonableness", "robustness", "overall"):
            val = parsed.get(dim, 0)
            parsed[dim] = max(0, min(5, int(val)))

        reasoning = str(parsed.get("reasoning", ""))[:300]
        overall = parsed["overall"]
        sc = parsed["step_completeness"]
        ra = parsed["result_accuracy"]
        fr = parsed["flow_reasonableness"]
        rb = parsed["robustness"]

        print(f"overall={overall}  S={sc} R={ra} F={fr} B={rb}  {reasoning[:80]}...")

        tr = TaskJudgeResult(
            task_id=tid,
            question=question[:120],
            step_completeness=sc,
            result_accuracy=ra,
            flow_reasonableness=fr,
            robustness=rb,
            overall=overall,
            reasoning=reasoning,
        )
        results.append(tr)
        completed_ids.add(tid)

        # ---- incremental save ------------------------------------------
        all_tasks = existing_tasks + [asdict(r) for r in results]
        # Deduplicate by task_id
        seen = set(); deduped = []
        for t in all_tasks:
            if t["task_id"] not in seen:
                seen.add(t["task_id"]); deduped.append(t)
        n_all = len(deduped)
        interim = JudgeSummary(
            experiment=experiment, model_id=model_id,
            total_tasks=total, evaluated_tasks=n_all, skipped_tasks=skipped,
            avg_step_completeness=round(sum(t["step_completeness"] for t in deduped) / n_all, 2),
            avg_result_accuracy=round(sum(t["result_accuracy"] for t in deduped) / n_all, 2),
            avg_flow_reasonableness=round(sum(t["flow_reasonableness"] for t in deduped) / n_all, 2),
            avg_robustness=round(sum(t["robustness"] for t in deduped) / n_all, 2),
            avg_overall=round(sum(t["overall"] for t in deduped) / n_all, 2),
            tasks=deduped,
        )
        _save_judge_incremental(interim, judge_path)

        # Brief pause to stay under rate limits
        time.sleep(0.3)

    # --- Build final summary ---
    all_tasks = existing_tasks + [asdict(r) for r in results]
    seen = set(); deduped = []
    for t in all_tasks:
        if t["task_id"] not in seen:
            seen.add(t["task_id"]); deduped.append(t)
    n_all = len(deduped)
    if n_all == 0:
        print("  No tasks evaluated.")
        return None

    return JudgeSummary(
        experiment=experiment,
        model_id=model_id,
        total_tasks=total,
        evaluated_tasks=n_all,
        skipped_tasks=skipped,
        avg_step_completeness=round(sum(t["step_completeness"] for t in deduped) / n_all, 2),
        avg_result_accuracy=round(sum(t["result_accuracy"] for t in deduped) / n_all, 2),
        avg_flow_reasonableness=round(sum(t["flow_reasonableness"] for t in deduped) / n_all, 2),
        avg_robustness=round(sum(t["robustness"] for t in deduped) / n_all, 2),
        avg_overall=round(sum(t["overall"] for t in deduped) / n_all, 2),
        tasks=deduped,
    )


# ---------------------------------------------------------------------------
# Fallback heuristic scoring (used when API is unavailable)
# ---------------------------------------------------------------------------

def _fallback_score(
    record: Dict[str, Any],
    bench_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute a simple heuristic score from trajectory vs ground truth."""
    gt_steps = bench_entry.get("trajectory_ground_truth", {}).get("steps", [])
    traj_steps = record.get("trajectory", {}).get("steps", [])
    called_tools = [s.get("action", "") for s in traj_steps]

    required_actions = [s["action"] for s in gt_steps if s.get("required")]
    optional_actions = [s["action"] for s in gt_steps if not s.get("required")]

    req_called = sum(1 for a in required_actions if a in called_tools)
    opt_called = sum(1 for a in optional_actions if a in called_tools)

    req_ratio = req_called / max(len(required_actions), 1)
    opt_ratio = opt_called / max(len(optional_actions), 1)

    sc = min(5, round(req_ratio * 5))
    ra = 5 if record.get("success") else min(3, round(req_ratio * 5))
    fr = min(5, round((req_ratio * 0.7 + (1 - min(opt_ratio, 0.5)) * 0.3) * 5))
    rb = 3 if record.get("success") else (2 if record.get("error") else 1)
    overall = round((sc + ra + fr + rb) / 4)

    return {
        "step_completeness": sc,
        "result_accuracy": ra,
        "flow_reasonableness": fr,
        "robustness": rb,
        "overall": overall,
        "reasoning": "[heuristic fallback]",
    }


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> Dict[str, Dict[str, Any]]:
    """Load evaluate.json and index by question text."""
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    index: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        q = task.get("question", "").strip()
        if q:
            index[q] = task
    print(f"Loaded {len(index)} benchmark entries from {path}")
    return index


# ---------------------------------------------------------------------------
# Collector: gather trajectory files from a path (file or directory)
# ---------------------------------------------------------------------------

def collect_trajectory_files(path: str) -> List[str]:
    """Return a list of absolute paths to trajectory JSON files."""
    if os.path.isfile(path):
        return [os.path.abspath(path)]

    if os.path.isdir(path):
        files = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json") and "_judge_scores" not in fname:
                files.append(os.path.abspath(os.path.join(path, fname)))
        return files

    print(f"[ERR] {path} is neither a file nor a directory")
    return []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_judge_output(summary: JudgeSummary, output_path: str) -> None:
    """Write JudgeSummary to a JSON file."""
    output: Dict[str, Any] = {
        "experiment": summary.experiment,
        "model_id": summary.model_id,
        "total_tasks": summary.total_tasks,
        "evaluated_tasks": summary.evaluated_tasks,
        "skipped_tasks": summary.skipped_tasks,
        "avg_scores": {
            "step_completeness": summary.avg_step_completeness,
            "result_accuracy": summary.avg_result_accuracy,
            "flow_reasonableness": summary.avg_flow_reasonableness,
            "robustness": summary.avg_robustness,
            "overall": summary.avg_overall,
        },
        "tasks": summary.tasks,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Judge scores saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="LLM-as-Judge — evaluate agent trajectories against benchmark",
    )
    p.add_argument("-i", "--input", required=True,
                   help="Trajectory JSON file or directory of such files")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory for judge scores (default: alongside input)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default="",
                   help="OpenAI-compatible API base URL (default: OpenAI official)")
    p.add_argument("--api-key", default=None,
                   help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("--benchmark", default=EVALUATE_JSON_PATH,
                   help="Path to evaluate.json")
    args = p.parse_args()

    # --- API key ---
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERR] No API key. Set OPENAI_API_KEY env var or pass --api-key")
        sys.exit(1)

    # --- Create OpenAI client once (reuse avoids "Too many open files") ---
    from openai import OpenAI
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    print(f"Judge model: {args.model}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Input:      {args.input}")

    # --- Load benchmark ---
    benchmark = load_benchmark(args.benchmark)

    # --- Collect files ---
    files = collect_trajectory_files(args.input)
    if not files:
        print("No trajectory files found.")
        sys.exit(1)
    print(f"Trajectory files to evaluate: {len(files)}")

    # --- Determine output directory ---
    if args.output_dir:
        out_dir = args.output_dir
    elif os.path.isfile(args.input):
        out_dir = os.path.dirname(args.input)
    else:
        out_dir = args.input
    print(f"Output dir: {out_dir}")

    # --- Evaluate each file ---
    for fpath in files:
        print(f"\n{'='*70}")
        print(f"Evaluating: {os.path.basename(fpath)}")
        print(f"{'='*70}")

        summary = evaluate_trajectory_file(
            fpath, benchmark, client, args.model, out_dir,
        )
        if summary is None:
            continue

        # Final save (already incrementally saved; this is the definitive version)
        out_path = _judge_output_path(fpath, out_dir)
        write_judge_output(summary, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
