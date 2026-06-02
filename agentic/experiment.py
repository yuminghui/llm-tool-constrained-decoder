"""
Experiment runner for comparing agent configurations.

Runs an agent across multiple tasks × configs (e.g. constrained vs. free
decoding, different models, different temperatures) and collects metrics
for analysis.
"""

from __future__ import annotations

import json
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.tools import ToolRegistry, load_tools_from_json
from agentic.llm_backend import LLMBackend
from experiments.config import QUANTIZATION
from agentic.agent import Agent, AgentConfig, AgentResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single experiment task (user query + metadata)."""
    task_id: str
    user_query: str
    description: str = ""
    expected_tools: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """A named experiment configuration."""
    name: str
    model_id: str
    use_constrained_decoder: bool
    temperature: float = 0.7
    max_turns: int = 10
    verbose: bool = False


@dataclass
class RunMetrics:
    """Metrics collected from a single agent run."""
    task_id: str
    config_name: str
    model_id: str
    success: bool
    total_time: float
    total_steps: int
    num_constrained_calls: int
    num_free_calls: int
    num_parse_errors: int
    plan_has_tools: bool
    final_answer: Optional[str]
    error: Optional[str] = None
    step_details: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default tasks
# ---------------------------------------------------------------------------

DEFAULT_TASKS = [
    Task(
        task_id="file_explore",
        user_query="Please explore the workspace: find out what's in the current directory, then search for any python files and list any directories containing 'app'.",
        description="File system exploration task",
        expected_tools=["plan", "bash_ls", "file_search", "dir_search"],
    ),
    Task(
        task_id="time_and_workspace",
        user_query="What's the current time? Also tell me the workspace path.",
        description="Simple info gathering: time + workspace path",
        expected_tools=["plan", "get_time", "path_workspace"],
    ),
    Task(
        task_id="remote_sensing",
        user_query=(
            "I need to process satellite imagery. First, list all available remote sensing indices, "
            "then search for TIF data in Shanghai for year 2024, "
            "and finally calculate NDVI index on the found files."
        ),
        description="Remote sensing workflow: list indices + search data + calculate",
        expected_tools=["plan", "list_remote_sensing_index", "search_tif_data", "remote_sensing_calculate_cli"],
    ),
    Task(
        task_id="change_detection",
        user_query=(
            "Perform change detection between two satellite images. "
            "The first image is at /var/opt/data/20240101-Beijing.tif, "
            "the second is at /var/opt/data/20250101-Beijing.tif. "
            "Save the result to /var/opt/output/change.tif."
        ),
        description="DCVA change detection task",
        expected_tools=["plan", "dcva_cd"],
    ),
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Orchestrates running multiple tasks × configs and reporting results.

    Usage::

        runner = ExperimentRunner()
        runner.add_task(DEFAULT_TASKS[0])
        runner.add_config(ExperimentConfig(
            name="constrained", model_id="Qwen/Qwen2.5-0.5B-Instruct",
            use_constrained_decoder=True,
        ))
        runner.add_config(ExperimentConfig(
            name="free", model_id="Qwen/Qwen2.5-0.5B-Instruct",
            use_constrained_decoder=False,
        ))
        runner.run_all()
        runner.print_report()
    """

    def __init__(self, tools_json_path: Optional[str] = None):
        self.tasks: List[Task] = []
        self.configs: List[ExperimentConfig] = []
        self.results: List[RunMetrics] = []

        # Resolve tools.json path
        if tools_json_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tools_json_path = os.path.join(project_root, "tools.json")
        self._tools_json_path = tools_json_path

        # Reusable tool registry (stateless)
        self._tool_registry: Optional[ToolRegistry] = None

        # Cache loaded backends by model_id so we don't re-load the model
        self._backends: Dict[str, LLMBackend] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_tasks(self, tasks: List[Task]) -> None:
        self.tasks.extend(tasks)

    def add_config(self, config: ExperimentConfig) -> None:
        self.configs.append(config)

    def add_configs(self, configs: List[ExperimentConfig]) -> None:
        self.configs.extend(configs)

    @property
    def tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            if os.path.exists(self._tools_json_path):
                self._tool_registry = load_tools_from_json(self._tools_json_path)
            else:
                raise FileNotFoundError(
                    f"tools.json not found at {self._tools_json_path}"
                )
        return self._tool_registry

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_all(self) -> List[RunMetrics]:
        """Run every config × task combination."""
        if not self.tasks:
            print("[ExperimentRunner] No tasks registered.")
            return []
        if not self.configs:
            print("[ExperimentRunner] No configs registered.")
            return []

        total = len(self.tasks) * len(self.configs)
        print(f"\n{'=' * 70}")
        print(f"  Experiment Runner: {len(self.tasks)} tasks × {len(self.configs)} configs = {total} runs")
        print(f"{'=' * 70}")

        self.results = []
        for cfg in self.configs:
            print(f"\n[Config] {cfg.name} (model={cfg.model_id}, "
                  f"constrained={cfg.use_constrained_decoder})")

            for task in self.tasks:
                print(f"  [Task] {task.task_id}: {task.user_query[:80]}...")
                metrics = self._run_single(task, cfg)
                self.results.append(metrics)

                status = "OK" if metrics.success else f"FAIL: {metrics.error}"
                print(f"    → {status} | steps={metrics.total_steps} | "
                      f"time={metrics.total_time:.1f}s | "
                      f"constrained={metrics.num_constrained_calls} | "
                      f"parse_errors={metrics.num_parse_errors}")

        return self.results

    def _run_single(self, task: Task, cfg: ExperimentConfig) -> RunMetrics:
        """Execute one task × config combination and collect metrics."""
        # Get or create backend for this model
        backend = self._get_backend(cfg.model_id)

        agent_config = AgentConfig(
            max_turns=cfg.max_turns,
            temperature=cfg.temperature,
            use_constrained_decoder=cfg.use_constrained_decoder,
            verbose=cfg.verbose,
        )

        agent = Agent(backend, self.tool_registry, agent_config)
        agent_result = agent.run(task.user_query)

        return self._compute_metrics(task, cfg, agent_result)

    def _get_backend(self, model_id: str) -> LLMBackend:
        """Lazy-load and cache model backends."""
        if model_id not in self._backends:
            self._backends[model_id] = LLMBackend(model_id, quantization=QUANTIZATION)
        return self._backends[model_id]

    def _compute_metrics(
        self, task: Task, cfg: ExperimentConfig, result: AgentResult
    ) -> RunMetrics:
        """Extract metrics from an AgentResult."""
        num_constrained = sum(1 for s in result.steps if s.is_constrained)
        num_free = sum(1 for s in result.steps if not s.is_constrained)

        # Count parse errors: steps where tool_name is set but tool_result is None
        num_parse_errors = sum(
            1 for s in result.steps
            if s.tool_name and s.tool_result is None
        )

        # Check if plan mentions any tool via expected_tools or keyword matching
        plan_has_tools = False
        for step in result.steps:
            if step.tool_name == "plan" and step.tool_args:
                steps = step.tool_args.get("steps", [])
                for s in steps:
                    if isinstance(s, dict) and s.get("expected_tools"):
                        plan_has_tools = True
                        break
                    elif isinstance(s, str):
                        from agentic.agent import TOOL_KEYWORDS
                        s_lower = s.lower()
                        for keywords in TOOL_KEYWORDS.values():
                            if any(kw in s_lower for kw in keywords):
                                plan_has_tools = True
                                break
                if plan_has_tools:
                    break

        step_details = []
        for s in result.steps:
            step_details.append({
                "step": s.step_index,
                "tool": s.tool_name,
                "constrained": s.is_constrained,
                "has_result": s.tool_result is not None,
                "elapsed": round(s.elapsed, 2),
            })

        return RunMetrics(
            task_id=task.task_id,
            config_name=cfg.name,
            model_id=cfg.model_id,
            success=result.success,
            total_time=result.total_time,
            total_steps=len(result.steps),
            num_constrained_calls=num_constrained,
            num_free_calls=num_free,
            num_parse_errors=num_parse_errors,
            plan_has_tools=plan_has_tools,
            final_answer=result.final_answer,
            error=result.error,
            step_details=step_details,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a summary table comparing configs across tasks."""
        if not self.results:
            print("No results to report.")
            return

        print("\n" + "=" * 80)
        print("  EXPERIMENT REPORT")
        print("=" * 80)

        # Header
        print(f"{'Task':<28s} {'Config':<16s} {'OK':>4s} {'Steps':>6s} "
              f"{'Time':>7s} {'Cstr':>5s} {'Free':>5s} {'PErr':>5s} "
              f"{'Plan':>5s}")
        print("-" * 80)

        # Per-task, per-config rows
        for r in self.results:
            print(f"{r.task_id:<28s} {r.config_name:<16s} "
                  f"{'Y' if r.success else 'N':>4s} "
                  f"{r.total_steps:>6d} {r.total_time:>6.1f}s "
                  f"{r.num_constrained_calls:>5d} {r.num_free_calls:>5d} "
                  f"{r.num_parse_errors:>5d} "
                  f"{'Y' if r.plan_has_tools else 'N':>5s}")

        print("-" * 80)

        # Comparison summary per task
        print("\n--- Per-Task Comparison ---")
        tasks = sorted(set(r.task_id for r in self.results))
        for task_id in tasks:
            task_results = [r for r in self.results if r.task_id == task_id]
            print(f"\n  Task: {task_id}")
            for r in task_results:
                parse_rate = "N/A"
                total_calls = r.num_constrained_calls + r.num_free_calls
                if total_calls > 0:
                    valid = total_calls - r.num_parse_errors
                    parse_rate = f"{valid}/{total_calls} ({100*valid//total_calls}%)"
                print(f"    {r.config_name}: success={r.success}, "
                      f"steps={r.total_steps}, time={r.total_time:.1f}s, "
                      f"valid_calls={parse_rate}")

        # Overall summary
        print(f"\n--- Overall ---")
        for cfg_name in sorted(set(r.config_name for r in self.results)):
            cfg_results = [r for r in self.results if r.config_name == cfg_name]
            success_rate = sum(1 for r in cfg_results if r.success) / len(cfg_results) * 100
            avg_time = sum(r.total_time for r in cfg_results) / len(cfg_results)
            total_errors = sum(r.num_parse_errors for r in cfg_results)
            print(f"  {cfg_name}: success_rate={success_rate:.0f}%, "
                  f"avg_time={avg_time:.1f}s, total_parse_errors={total_errors}")

        print("\n" + "=" * 80)

    def export_json(self, filepath: str) -> None:
        """Export all run metrics as JSON."""
        records = []
        for r in self.results:
            records.append({
                "task_id": r.task_id,
                "config_name": r.config_name,
                "model_id": r.model_id,
                "success": r.success,
                "total_time": r.total_time,
                "total_steps": r.total_steps,
                "num_constrained_calls": r.num_constrained_calls,
                "num_free_calls": r.num_free_calls,
                "num_parse_errors": r.num_parse_errors,
                "plan_has_tools": r.plan_has_tools,
                "final_answer": r.final_answer,
                "error": r.error,
                "step_details": r.step_details,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[ExperimentRunner] Results exported to {filepath}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run a small experiment from the command line.

    Usage::

        python -m agentic.experiment
    """
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Experiment Runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model ID to use")
    parser.add_argument("--task", choices=["all", "file", "time", "rs", "cd"],
                        default="all", help="Which task(s) to run")
    parser.add_argument("--constrained-only", action="store_true",
                        help="Only run constrained config")
    parser.add_argument("--free-only", action="store_true",
                        help="Only run free config")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose agent output")
    parser.add_argument("--export", metavar="PATH",
                        help="Export results to JSON file")
    parser.add_argument("--tools-json", metavar="PATH",
                        help="Path to tools.json (default: <project_root>/tools.json)")
    args = parser.parse_args()

    # Select tasks
    task_map = {
        "file": [DEFAULT_TASKS[0]],
        "time": [DEFAULT_TASKS[1]],
        "rs":   [DEFAULT_TASKS[2]],
        "cd":   [DEFAULT_TASKS[3]],
        "all":  DEFAULT_TASKS,
    }
    tasks = task_map.get(args.task, DEFAULT_TASKS)

    # Build configs
    configs = []
    if not args.free_only:
        configs.append(ExperimentConfig(
            name="constrained",
            model_id=args.model,
            use_constrained_decoder=True,
            verbose=args.verbose,
        ))
    if not args.constrained_only:
        configs.append(ExperimentConfig(
            name="free",
            model_id=args.model,
            use_constrained_decoder=False,
            verbose=args.verbose,
        ))

    # Run
    runner = ExperimentRunner(tools_json_path=args.tools_json)
    runner.add_tasks(tasks)
    runner.add_configs(configs)
    runner.run_all()
    runner.print_report()

    if args.export:
        runner.export_json(args.export)


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
