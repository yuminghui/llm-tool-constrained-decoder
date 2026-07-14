"""
Seal-Tools → evaluate.json / tools.json.

Source: https://github.com/fairyshine/Seal-Tools  (dataset under its repo license)
Files:  Seal-Tools_Dataset/tool.jsonl (tool schemas),
        Seal-Tools_Dataset/test_in_domain.jsonl, test_out_domain.jsonl (tasks)

Each task record is ``{id, query, calling:[{api, parameters, responses}]}``; the
ground-truth domain tools are the ordered ``api`` names.  Because Seal-Tools has
a large (~4k) tool pool, each task exposes only its ground-truth tools plus a
sampled set of distractors, keeping SLM prompts bounded while still testing tool
selection.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

from experiments.others.converters import (
    build_eval_record, schema_from_named_params, write_converted,
    read_jsonl, dedup, map_type,
)

RAW_BASE = "https://raw.githubusercontent.com/fairyshine/Seal-Tools/master/Seal-Tools_Dataset"
SOURCES = [
    (f"{RAW_BASE}/tool.jsonl", "tool.jsonl"),
    (f"{RAW_BASE}/test_in_domain.jsonl", "test_in_domain.jsonl"),
    (f"{RAW_BASE}/test_out_domain.jsonl", "test_out_domain.jsonl"),
]

_TASK_FILES = ["test_in_domain.jsonl", "test_out_domain.jsonl"]


def _response_stub(resp_schema: Any) -> str:
    """Build a structured reference-output stub from a tool's response schema.

    Seal-Tools ships response *field* schemas (not concrete gold values), so the
    stub advertises the output shape, e.g. ``{"analysis_results": "<string>"}``.
    Returns "" when no schema is available (falls back to the generic executor).
    """
    if not isinstance(resp_schema, dict) or not resp_schema:
        return ""
    out: Dict[str, str] = {}
    for field, spec in resp_schema.items():
        t = spec.get("type", "str") if isinstance(spec, dict) else "str"
        out[field] = f"<{map_type(t)}>"
    return json.dumps(out, ensure_ascii=False)


def _load_tool_pool(raw_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return ``(tool_defs, response_stubs)`` keyed by api_name."""
    path = os.path.join(raw_dir, "tool.jsonl")
    pool: Dict[str, Dict[str, Any]] = {}
    resp_stub: Dict[str, str] = {}
    if not os.path.isfile(path):
        return pool, resp_stub
    for t in read_jsonl(path):
        name = t.get("api_name")
        if not name:
            continue
        pool[name] = {
            "name": name,
            "description": str(t.get("api_description", f"Tool {name}.")),
            "parameters": schema_from_named_params(
                t.get("parameters", {}), t.get("required", [])
            ),
        }
        stub = _response_stub(t.get("responses"))
        if stub:
            resp_stub[name] = stub
    return pool, resp_stub


def convert(
    raw_dir: str,
    out_dir: str,
    limit: int = 0,
    seed: int = 0,
    n_distractors: int = 12,
) -> Dict[str, int]:
    rng = random.Random(seed)
    pool, resp_stub = _load_tool_pool(raw_dir)
    pool_names = list(pool.keys())

    raw_tasks: List[Dict[str, Any]] = []
    for fname in _TASK_FILES:
        path = os.path.join(raw_dir, fname)
        if os.path.isfile(path):
            raw_tasks.extend(read_jsonl(path))

    if limit and limit > 0:
        rng.shuffle(raw_tasks)
        raw_tasks = raw_tasks[:limit]

    records: List[Dict[str, Any]] = []
    used_tools: set = set()
    for t in raw_tasks:
        query = t.get("query")
        calling = t.get("calling") or []
        if not query or not calling:
            continue
        gt = dedup([c.get("api") for c in calling if isinstance(c, dict) and c.get("api")])
        if not gt:
            continue
        # distractors sampled deterministically per task
        excl = set(gt)
        distractors = [n for n in pool_names if n not in excl]
        rng.shuffle(distractors)
        distractors = distractors[:n_distractors]
        available = dedup(gt + distractors)
        used_tools.update(available)
        tool_responses = {api: resp_stub[api] for api in gt if api in resp_stub}
        records.append(build_eval_record(
            question=str(query),
            gt_domain_tools=gt,
            available_tools=available,
            expected="",
            tool_responses=tool_responses,
        ))

    tool_defs = {n: pool[n] for n in used_tools if n in pool}
    return write_converted(out_dir, records, tool_defs)
