"""
Run the three PTC-Decoder comparison groups on external datasets.

Groups (identical design to the main benchmark):
  * ``baseline``     — Baseline, free generation, no plan          (Exp A)
  * ``ptc``          — PTC-Decoder: constrained plan + constrained exec (Exp B)
  * ``plan_wo_tc``   — Plan w/o TC-Decoder: constrained plan + free exec (Exp F)

Backends:
  * ``mock`` — deterministic, torch-free; validates the pipeline offline.
  * ``hf``   — the real ``LLMBackend`` (reuses ``config.ALL_MODELS`` by default).

Usage::

    # offline pipeline check (no torch, no GPU)
    python -m experiments.others.run_experiments --dataset sample --group all --backend mock

    # real run on a GPU box (all models from config.ALL_MODELS)
    python -m experiments.others.run_experiments --dataset seal_tools --group all --backend hf

    # a single model / a quick smoke
    python -m experiments.others.run_experiments --dataset toolalpaca --group ptc \
        --backend hf --model Qwen/Qwen3-1.7B --quick
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, List, Tuple

from experiments.others.common import DatasetSpec, run_group, GROUP_RUNNERS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
REAL_DATASETS = ["toolalpaca", "seal_tools", "api_bank"]


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _mock_factory() -> Callable[[str, bool], Any]:
    from experiments.others.mock_backend import MockBackend

    def factory(model_id: str, quantize: bool):
        return MockBackend(model_id)

    return factory


def _hf_factory() -> Callable[[str, bool], Any]:
    # Imported lazily so the mock path never requires torch.
    from agentic.llm_backend import LLMBackend
    from experiments.config import QUANTIZATION_MODE

    def factory(model_id: str, quantize: bool):
        return LLMBackend(model_id, quantize=quantize, quantization_mode=QUANTIZATION_MODE)

    return factory


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve_specs(dataset: str) -> Tuple[List[DatasetSpec], str]:
    """Return (specs, trajectories_root) for the requested dataset selector."""
    if dataset == "sample":
        specs = [DatasetSpec(name, os.path.join(DATA_ROOT, "sample", name)) for name in REAL_DATASETS]
        return specs, os.path.join(BASE_DIR, "trajectories", "_sample")
    if dataset == "all":
        specs = [DatasetSpec(name, os.path.join(DATA_ROOT, name)) for name in REAL_DATASETS]
        return specs, os.path.join(BASE_DIR, "trajectories")
    specs = [DatasetSpec(dataset, os.path.join(DATA_ROOT, dataset))]
    return specs, os.path.join(BASE_DIR, "trajectories")


def _resolve_models(args) -> List[Any]:
    if args.model:
        return [(args.model, args.quantize)]
    if args.backend == "mock":
        return [("mock/deterministic-oracle", False)]
    from experiments.config import ALL_MODELS
    return ALL_MODELS


def _resolve_groups(group: str) -> List[str]:
    if group == "all":
        return ["baseline", "ptc", "plan_wo_tc"]
    if group not in GROUP_RUNNERS:
        raise SystemExit(f"unknown group {group!r}")
    return [group]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="PTC-Decoder external-dataset experiments")
    p.add_argument("--dataset", required=True,
                   choices=["toolalpaca", "seal_tools", "api_bank", "all", "sample"])
    p.add_argument("--group", default="all",
                   choices=["baseline", "ptc", "plan_wo_tc", "all"])
    p.add_argument("--backend", default="mock", choices=["mock", "hf"])
    p.add_argument("--model", type=str, default=None, help="single model id override")
    p.add_argument("--quantize", action="store_true", help="4-bit quantize the --model (hf only)")
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="5-task smoke test")
    p.add_argument("--no-resume", action="store_true", help="do not skip already-saved tasks")
    args = p.parse_args()

    specs, traj_root = _resolve_specs(args.dataset)
    models = _resolve_models(args)
    groups = _resolve_groups(args.group)
    backend_factory = _mock_factory() if args.backend == "mock" else _hf_factory()

    max_tasks = args.max_tasks or (5 if args.quick else 0)
    skip_existing = not args.no_resume and not args.quick and args.max_tasks == 0

    for spec in specs:
        if not os.path.isfile(spec.evaluate_json):
            print(f"!! {spec.name}: {spec.evaluate_json} missing — run prepare_data first; skipping")
            continue
        for group in groups:
            run_group(
                spec, group, models, backend_factory, traj_root,
                max_tasks=max_tasks, skip_existing=skip_existing,
            )

    print(f"\nDone. Trajectories under: {traj_root}")
    print("Aggregate with:  python -m experiments.others.aggregate "
          f"--dataset {args.dataset}")


if __name__ == "__main__":
    main()
