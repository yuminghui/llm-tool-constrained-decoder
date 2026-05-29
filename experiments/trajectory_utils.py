"""
Trajectory save/load utilities for agent execution traces.

Two helpers:

- ``save_trajectory()`` — write a single agent run to a new JSON file.
- ``append_trajectory()`` — append to an existing JSON array (or create new).

The output format matches ``evaluate.json`` item structure so saved traces
can be fed directly into the same evaluation pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentic.agent import AgentResult, AgentStep


# ---------------------------------------------------------------------------
# AgentResult → trajectory record
# ---------------------------------------------------------------------------

def agent_result_to_record(
    agent_result: AgentResult,
    model_id: str = "",
    config_name: str = "",
    task_id: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert an ``AgentResult`` into a trajectory record dict.

    The output format mirrors ``evaluate.json`` items, with added fields
    for agent metadata:

    .. code-block:: json

        {
            "question": "...",
            "task_id": "...",
            "model_id": "...",
            "config": "...",
            "timestamp": "2026-05-28T12:00:00",
            "success": true,
            "error": null,
            "total_time": 5.2,
            "total_tokens": 1234,
            "final_answer": "...",
            "trajectory_evaluate": true,
            "trajectory": {
                "type": "trajectory",
                "steps": [
                    {"action": "plan", "is_constrained": true, "args": {...}},
                    ...
                ]
            },
            ...extra_meta
        }
    """
    steps = []
    for s in agent_result.steps:
        step_record: Dict[str, Any] = {
            "action": s.tool_name or "(text_response)",
            "step_index": s.step_index,
            "is_constrained": s.is_constrained,
            "elapsed": round(s.elapsed, 3),
        }
        if s.tool_args:
            step_record["args"] = s.tool_args
        if s.tool_result:
            step_record["result"] = _truncate(s.tool_result, 500)
        steps.append(step_record)

    record: Dict[str, Any] = {
        "question": agent_result.user_query,
        "task_id": task_id,
        "model_id": model_id,
        "config": config_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": agent_result.success,
        "error": agent_result.error,
        "total_time": round(agent_result.total_time, 3),
        "total_tokens": agent_result.total_tokens,
        "final_answer": _truncate(agent_result.final_answer, 1000) if agent_result.final_answer else None,
        "trajectory_evaluate": True,
        "trajectory": {
            "type": "trajectory",
            "steps": steps,
        },
    }

    if extra_meta:
        record.update(extra_meta)

    return record


# ---------------------------------------------------------------------------
# Save (new file)
# ---------------------------------------------------------------------------

def save_trajectory(
    agent_result: AgentResult,
    filepath: str,
    model_id: str = "",
    config_name: str = "",
    task_id: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a single agent execution trace to a new JSON file.

    Overwrites *filepath* if it already exists.

    Args:
        agent_result: The result returned by ``Agent.run()``.
        filepath: Path to the output JSON file.
        model_id: Identifier of the model used.
        config_name: Experiment config name (e.g. "constrained", "free").
        task_id: Optional task identifier.
        extra_meta: Arbitrary extra keys to merge into the record.

    Returns:
        The absolute path to the written file.
    """
    record = agent_result_to_record(
        agent_result,
        model_id=model_id,
        config_name=config_name,
        task_id=task_id,
        extra_meta=extra_meta,
    )

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([record], f, ensure_ascii=False, indent=2)

    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Append (existing or new file)
# ---------------------------------------------------------------------------

def append_trajectory(
    agent_result: AgentResult,
    filepath: str,
    model_id: str = "",
    config_name: str = "",
    task_id: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Append an agent execution trace to a JSON file.

    - If *filepath* exists and contains a valid JSON **array**, the new
      record is appended.
    - If *filepath* does not exist (or is empty), a new array is created.
    - If *filepath* exists but is **not** a JSON array, it is overwritten
      with a new array containing the existing content + new record.

    Args:
        agent_result: The result returned by ``Agent.run()``.
        filepath: Path to the JSON file.
        model_id: Identifier of the model used.
        config_name: Experiment config name.
        task_id: Optional task identifier.
        extra_meta: Arbitrary extra keys to merge into the record.

    Returns:
        The absolute path to the written file.
    """
    record = agent_result_to_record(
        agent_result,
        model_id=model_id,
        config_name=config_name,
        task_id=task_id,
        extra_meta=extra_meta,
    )

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # Try to read existing content
    existing: List[Dict[str, Any]] = []
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
            else:
                # Single object — wrap in array
                existing = [data]
        except (json.JSONDecodeError, Exception):
            # Corrupted or empty — start fresh
            existing = []

    existing.append(record)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Batch convenience
# ---------------------------------------------------------------------------

def save_trajectories_batch(
    results: List[AgentResult],
    filepath: str,
    model_id: str = "",
    config_name: str = "",
    task_ids: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write multiple agent results to one JSON file (overwrite)."""
    if task_ids is None:
        task_ids = [""] * len(results)

    records = []
    for i, r in enumerate(results):
        tid = task_ids[i] if i < len(task_ids) else ""
        records.append(agent_result_to_record(
            r, model_id=model_id, config_name=config_name,
            task_id=tid, extra_meta=extra_meta,
        ))

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: Optional[str], max_len: int) -> Optional[str]:
    """Truncate *text* to *max_len* characters, adding '…' if truncated."""
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"