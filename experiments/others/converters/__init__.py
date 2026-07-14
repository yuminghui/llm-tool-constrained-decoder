"""
Dataset converters: external tool-calling benchmarks → this repo's contract.

Each converter turns a raw open-source dataset into the pair of files the
experiment harness consumes:

* ``evaluate.json`` — a list of task records, each::

      {
        "question": "<NL task>",
        "expected": "<optional expected-outcome text for the LLM judge>",
        "available_tools": ["toolA", "toolB", ...],
        "trajectory_evaluate": true,
        "trajectory_ground_truth": {
          "type": "trajectory", "required_score": 1.0, "optional_score": 0.5,
          "steps": [{"action": "...", "required": true/false}, ...]
        }
      }

* ``tools.json`` — the union of every tool referenced by any task's
  ``available_tools`` (name / description / JSON-Schema parameters / enabled),
  plus the method's meta-tools ``plan`` / ``task_summary`` / ``task_done``.

This ``__init__`` holds the shared helpers; ``toolalpaca.py`` / ``seal_tools.py``
/ ``api_bank.py`` hold the per-dataset logic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from experiments.others.generic_tools import canonical_meta_defs, META_TOOLS

# Actions that mark trajectory termination in some datasets — never treated as
# domain tools.
_TERMINAL_ACTIONS = {"finish", "Finish", "stop", "Stop", "done", "END", "final_answer"}

# python-ish type strings (Seal-Tools / API-Bank) → JSON-Schema types.
_TYPE_MAP = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer", "long": "integer",
    "float": "number", "double": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array",
    "dict": "object", "object": "object", "json": "object",
}


def json_type_of(value: Any) -> str:
    """Infer a JSON-Schema type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def map_type(t: Any) -> str:
    if not isinstance(t, str):
        return "string"
    return _TYPE_MAP.get(t.strip().lower(), "string")


def dedup(seq: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication."""
    seen: set = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def schema_from_named_params(params: Dict[str, Any], required: Optional[List[str]]) -> Dict[str, Any]:
    """Seal-Tools style ``{name: {type, description}}`` → JSON-Schema object."""
    props: Dict[str, Any] = {}
    if isinstance(params, dict):
        for pname, spec in params.items():
            if isinstance(spec, dict):
                props[pname] = {
                    "type": map_type(spec.get("type", "string")),
                    "description": str(spec.get("description", "")),
                }
            else:
                props[pname] = {"type": "string", "description": str(spec)}
    req = [r for r in (required or []) if r in props]
    return {"type": "object", "properties": props, "required": req}


def schema_from_examples(arg_dicts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer an object schema from observed argument dicts (API-Bank/ToolAlpaca).

    Union of keys; type from the first non-null value seen; a key is ``required``
    only if it appears in *every* example.
    """
    props: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    n = 0
    for args in arg_dicts:
        if not isinstance(args, dict):
            continue
        n += 1
        for k, v in args.items():
            counts[k] = counts.get(k, 0) + 1
            if k not in props or props[k]["type"] == "string":
                props[k] = {"type": json_type_of(v), "description": ""}
    required = [k for k, c in counts.items() if n and c == n]
    return {"type": "object", "properties": props, "required": required}


def parse_json_loose(s: Any) -> Dict[str, Any]:
    """Parse a possibly-stringified JSON args blob; return {} on failure."""
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_eval_record(
    question: str,
    gt_domain_tools: List[str],
    available_tools: List[str],
    expected: str = "",
    *,
    plan_optional: bool = True,
    optional_domain_tools: Optional[List[str]] = None,
    tool_responses: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble one evaluate.json record.

    Ground-truth steps mirror the main benchmark convention:
    optional leading ``plan`` → each domain tool (required, in order) →
    ``task_summary`` (required) → ``task_done`` (required).

    ``tool_responses`` (optional) maps a tool name to a reference return string
    from the source dataset (e.g. API-Bank's real ``result.output``).  When
    present the harness feeds it back into the agent loop via the simulated
    executor, so dependency-carrying tasks (auth-then-act) see realistic output.
    """
    optional_domain_tools = set(optional_domain_tools or [])
    steps: List[Dict[str, Any]] = []
    if plan_optional:
        steps.append({"action": "plan", "required": False})
    for t in dedup(gt_domain_tools):
        steps.append({"action": t, "required": t not in optional_domain_tools})
    steps.append({"action": "task_summary", "required": True})
    steps.append({"action": "task_done", "required": True})

    available = dedup(list(available_tools) + list(gt_domain_tools))
    record: Dict[str, Any] = {
        "question": question.strip(),
        "expected": expected.strip() if expected else "",
        "available_tools": available,
        "trajectory_evaluate": True,
        "trajectory_ground_truth": {
            "type": "trajectory",
            "required_score": 1.0,
            "optional_score": 0.5,
            "steps": steps,
        },
    }
    if tool_responses:
        cleaned = {k: v for k, v in tool_responses.items() if k and v}
        if cleaned:
            record["tool_responses"] = cleaned
    return record


def sample_distractors(all_names: List[str], exclude: Iterable[str], k: int, rng) -> List[str]:
    """Pick up to *k* distractor tool names not in *exclude* (deterministic rng)."""
    excl = set(exclude)
    pool = [n for n in all_names if n not in excl]
    if k <= 0 or not pool:
        return []
    rng.shuffle(pool)
    return pool[:k]


def write_converted(
    out_dir: str,
    records: List[Dict[str, Any]],
    tool_defs: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Write ``evaluate.json`` + ``tools.json`` for a converted dataset.

    ``tools.json`` is the union of every tool referenced by any record's
    ``available_tools`` (using *tool_defs* where available, a permissive stub
    otherwise) plus the canonical meta-tools.
    """
    os.makedirs(out_dir, exist_ok=True)

    referenced: List[str] = []
    for r in records:
        referenced.extend(r.get("available_tools", []))
    referenced = dedup(referenced)

    meta = canonical_meta_defs()
    tools_out: List[Dict[str, Any]] = []
    for name in referenced:
        if name in META_TOOLS:
            continue  # added below from canonical defs
        d = tool_defs.get(name)
        if d is None:
            d = {
                "name": name,
                "description": f"Tool {name}.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        tools_out.append({
            "name": d["name"],
            "description": d.get("description", f"Tool {name}."),
            "parameters": d.get("parameters", {"type": "object", "properties": {}, "required": []}),
            "enabled": True,
        })
    for name, d in meta.items():
        tools_out.append({**d, "enabled": True})

    with open(os.path.join(out_dir, "evaluate.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "tools.json"), "w", encoding="utf-8") as f:
        json.dump(tools_out, f, ensure_ascii=False, indent=1)

    return {"tasks": len(records), "tools": len(tools_out)}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file robustly (skip blanks, tolerate CRLF/BOM)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
