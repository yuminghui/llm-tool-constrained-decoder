"""
Aggregate saved trajectories into the Baseline / PTC-Decoder / Plan-w/o-TC table.

Reads the per-task ``eval`` blocks written by ``common.run_group`` and produces,
for each dataset, a group × model table plus a group-level mean (the headline
A/B/F comparison).  Writes ``report.md`` and ``report.json`` next to the
trajectories.

Usage::

    python -m experiments.others.aggregate --dataset all
    python -m experiments.others.aggregate --dataset sample
    python -m experiments.others.aggregate --dataset seal_tools
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from experiments.others.common import GROUP_DIRS, GROUP_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATASETS = ["toolalpaca", "seal_tools", "api_bank"]
GROUP_ORDER = ["baseline", "ptc", "plan_wo_tc"]
_DIR_TO_GROUP = {v: k for k, v in GROUP_DIRS.items()}

METRIC_KEYS = [
    ("completion_rate", "Compl"),
    ("avg_domain_recall", "DomRec"),
    ("avg_domain_precision", "DomPrec"),
    ("avg_domain_f1", "DomF1"),
    ("success_rate", "Succ"),
    ("avg_completion_score", "Score"),
    ("plan_call_rate", "Plan"),
    ("avg_tokens", "AvgTok"),
]

META_TOOLS = {"plan", "task_summary", "task_done"}


def _domain_prec_f1(ev: Dict[str, Any]) -> tuple:
    """Return ``(precision, f1)`` from an eval block.

    If ``domain_precision`` / ``domain_f1`` are already saved (new experiments),
    use them directly.  Otherwise compute from ``called_tools`` + domain counts
    (backward-compatible with trajectories written before Precision/F1 were added).
    """
    if "domain_precision" in ev and "domain_f1" in ev:
        return ev["domain_precision"], ev["domain_f1"]

    # Backward-compat: compute from existing fields.
    called = ev.get("called_tools", [])
    agent_domain = len(set(t for t in called if t not in META_TOOLS))
    tp = ev.get("domain_required_called", 0) + ev.get("domain_optional_called", 0)
    fp = max(agent_domain - tp, 0)
    rec = ev.get("domain_recall", 0.0)
    prec = round(tp / max(tp + fp, 1), 3)
    f1 = round(2 * prec * rec / max(prec + rec, 1e-9), 3)
    return prec, f1


def _load_model_file(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict) and r.get("type") != "_summary"]


def _model_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    completed = success = plan = 0
    dom = prec = f1 = score = toks = 0.0
    for r in records:
        ev = r.get("eval", {})
        completed += 1 if ev.get("completed") else 0
        success += 1 if r.get("success") else 0
        dom += ev.get("domain_recall", 0.0)
        p, f = _domain_prec_f1(ev)
        prec += p
        f1 += f
        score += ev.get("completion_score", 0.0)
        if "plan" in ev.get("called_tools", []):
            plan += 1
        toks += r.get("total_tokens", 0)
    return {
        "n": n,
        "completion_rate": completed / n,
        "success_rate": success / n,
        "avg_domain_recall": dom / n,
        "avg_domain_precision": prec / n,
        "avg_domain_f1": f1 / n,
        "avg_completion_score": score / n,
        "plan_call_rate": plan / n,
        "avg_tokens": toks / n,
    }


def _collect_dataset(
    traj_root: str, dataset: str, include_mock: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{group: {model_id: metrics}}`` for one dataset.

    The deterministic ``mock`` backend is only a pipeline-validation stand-in, so
    its files are excluded from result tables unless ``include_mock`` is set.
    """
    ds_dir = os.path.join(traj_root, dataset)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {g: {} for g in GROUP_ORDER}
    if not os.path.isdir(ds_dir):
        return out
    for group_dir in os.listdir(ds_dir):
        group = _DIR_TO_GROUP.get(group_dir)
        if group is None:
            continue
        gpath = os.path.join(ds_dir, group_dir)
        if not os.path.isdir(gpath):
            continue
        for fname in os.listdir(gpath):
            if not fname.endswith(".json"):
                continue
            # skip non-trajectory files that happen to be JSON
            if fname in ("report.json",) or fname.endswith("_judge_scores.json"):
                continue
            if not include_mock and fname.startswith("mock"):
                continue
            model_id = fname[:-5].replace("_", "/", 1)  # best-effort display id
            metrics = _model_metrics(_load_model_file(os.path.join(gpath, fname)))
            if metrics:
                out[group][fname[:-5]] = {"display": model_id, **metrics}
    return out


def _mean_over_models(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not models:
        return {}
    keys = [k for k, _ in METRIC_KEYS]
    agg = {k: 0.0 for k in keys}
    for m in models.values():
        for k in keys:
            agg[k] += m.get(k, 0.0)
    n = len(models)
    return {k: agg[k] / n for k in keys}


def _fmt(metric: str, value: float) -> str:
    if metric == "avg_tokens":
        return f"{value:.0f}"
    return f"{value:.0%}" if value <= 1.0 else f"{value:.2f}"


def _render_markdown(dataset: str, data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    lines = [f"## {dataset}", ""]
    # headline: mean over models per group
    lines.append("### Group means (averaged over models)")
    lines.append("")
    header = "| Group | " + " | ".join(lbl for _, lbl in METRIC_KEYS) + " |"
    sep = "|" + "---|" * (len(METRIC_KEYS) + 1)
    lines += [header, sep]
    for g in GROUP_ORDER:
        mean = _mean_over_models(data.get(g, {}))
        if not mean:
            continue
        row = f"| {GROUP_LABELS[g]} | " + " | ".join(_fmt(k, mean[k]) for k, _ in METRIC_KEYS) + " |"
        lines.append(row)
    lines.append("")
    # per-model detail
    lines.append("### Per-model detail")
    lines.append("")
    models = sorted({m for g in GROUP_ORDER for m in data.get(g, {})})
    lines.append("| Model | Group | " + " | ".join(lbl for _, lbl in METRIC_KEYS) + " |")
    lines.append("|" + "---|" * (len(METRIC_KEYS) + 2))
    for model in models:
        for g in GROUP_ORDER:
            m = data.get(g, {}).get(model)
            if not m:
                continue
            row = (f"| {m['display']} | {GROUP_LABELS[g]} | "
                   + " | ".join(_fmt(k, m[k]) for k, _ in METRIC_KEYS) + " |")
            lines.append(row)
    lines.append("")
    return "\n".join(lines)


def _print_console(dataset: str, data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    print(f"\n{'='*92}\n  {dataset}  —  group means (avg over models)\n{'='*92}")
    header = f"  {'Group':<26s} " + " ".join(f"{lbl:>7s}" for _, lbl in METRIC_KEYS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for g in GROUP_ORDER:
        mean = _mean_over_models(data.get(g, {}))
        if not mean:
            continue
        vals = " ".join(f"{_fmt(k, mean[k]):>7s}" for k, _ in METRIC_KEYS)
        print(f"  {GROUP_LABELS[g]:<26s} {vals}")

    # per-model detail
    all_models = sorted({m for g in GROUP_ORDER for m in data.get(g, {})})
    if not all_models:
        return
    print(f"\n  --- per-model detail ({len(all_models)} model(s)) ---")
    for model in all_models:
        print(f"\n  [{model}]")
        print(f"  {'Group':<26s} " + " ".join(f"{lbl:>7s}" for _, lbl in METRIC_KEYS))
        for g in GROUP_ORDER:
            m = data.get(g, {}).get(model)
            if not m:
                continue
            vals = " ".join(f"{_fmt(k, m[k]):>7s}" for k, _ in METRIC_KEYS)
            print(f"  {GROUP_LABELS[g]:<26s} {vals}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate PTC-Decoder external-dataset results")
    p.add_argument("--dataset", required=True,
                   choices=["toolalpaca", "seal_tools", "api_bank", "all", "sample"])
    p.add_argument("--out", type=str, default=None, help="output dir (default: alongside trajectories)")
    p.add_argument("--include-mock", action="store_true",
                   help="include the mock backend's files (excluded by default)")
    args = p.parse_args()

    if args.dataset == "sample":
        traj_root = os.path.join(BASE_DIR, "trajectories", "_sample")
        datasets = REAL_DATASETS
    elif args.dataset == "all":
        traj_root = os.path.join(BASE_DIR, "trajectories")
        datasets = REAL_DATASETS
    else:
        traj_root = os.path.join(BASE_DIR, "trajectories")
        datasets = [args.dataset]

    out_dir = args.out or traj_root
    os.makedirs(out_dir, exist_ok=True)

    report: Dict[str, Any] = {}
    md_parts = ["# PTC-Decoder — external-dataset results", ""]
    for ds in datasets:
        data = _collect_dataset(traj_root, ds, include_mock=args.include_mock)
        if not any(data.get(g) for g in GROUP_ORDER):
            print(f"  [{ds}] no trajectories found under {traj_root} — skipping")
            continue
        report[ds] = {
            g: {mid: {k: v for k, v in m.items()} for mid, m in data[g].items()}
            for g in GROUP_ORDER
        }
        report[ds]["_group_means"] = {g: _mean_over_models(data.get(g, {})) for g in GROUP_ORDER}
        _print_console(ds, data)
        md_parts.append(_render_markdown(ds, data))

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_parts))
    print(f"\nWrote {os.path.join(out_dir, 'report.md')} and report.json")


if __name__ == "__main__":
    main()
