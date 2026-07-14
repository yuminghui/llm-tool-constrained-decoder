"""
ToolAlpaca → evaluate.json / tools.json.

Source: https://github.com/tangqiaoyu/ToolAlpaca  (Apache-2.0)
Files:  data/eval_simulated.json, data/eval_real.json

Each raw record is a *toolset*: ``Function_Description`` (NL schemas keyed by
function name, plus a ``components`` entry), ``Instructions`` (a list of NL
queries) and ``Golden_Answers`` (parallel list; each a ``[{Action, Action_Input}]``
sequence).  One evaluate.json task is emitted per instruction; the ground-truth
domain tools are the ``Action`` names, and the task's ``available_tools`` are all
functions of the toolset (natural in-domain distractors).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

from experiments.others.converters import (
    build_eval_record, schema_from_examples, parse_json_loose,
    write_converted, dedup, _TERMINAL_ACTIONS,
)

RAW_BASE = "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main"
SOURCES = [
    (f"{RAW_BASE}/data/eval_simulated.json", "eval_simulated.json"),
    (f"{RAW_BASE}/data/eval_real.json", "eval_real.json"),
]

_SKIP_FUNC_KEYS = {"components"}


def _toolset_functions(record: Dict[str, Any]) -> List[str]:
    fd = record.get("Function_Description") or {}
    return [k for k in fd.keys() if k not in _SKIP_FUNC_KEYS]


def _short_desc(text: Any, limit: int = 400) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s[:limit]


def convert(raw_dir: str, out_dir: str, limit: int = 0, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    tool_defs: Dict[str, Dict[str, Any]] = {}
    # collect argument examples per (global) tool name to infer schemas
    arg_examples: Dict[str, List[Dict[str, Any]]] = {}
    desc_map: Dict[str, str] = {}

    for _url, fname in SOURCES:
        path = os.path.join(raw_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            toolsets = json.load(f)

        for ts in toolsets:
            funcs = _toolset_functions(ts)
            if not funcs:
                continue
            fd = ts.get("Function_Description") or {}
            for fn in funcs:
                desc_map.setdefault(fn, _short_desc(fd.get(fn)))

            instructions = ts.get("Instructions") or []
            goldens = ts.get("Golden_Answers") or []
            for instr, ga in zip(instructions, goldens):
                if not isinstance(ga, list):
                    continue
                gt: List[str] = []
                for step in ga:
                    if not isinstance(step, dict):
                        continue
                    action = step.get("Action")
                    if not action or action in _TERMINAL_ACTIONS:
                        continue
                    gt.append(action)
                    arg_examples.setdefault(action, []).append(
                        parse_json_loose(step.get("Action_Input"))
                    )
                gt = dedup(gt)
                if not gt:
                    continue
                records.append(build_eval_record(
                    question=str(instr),
                    gt_domain_tools=gt,
                    available_tools=funcs,
                    expected="",
                ))

    # build tool defs (schema inferred from observed Action_Input examples)
    referenced = dedup([t for r in records for t in r["available_tools"]])
    for name in referenced:
        tool_defs[name] = {
            "name": name,
            "description": desc_map.get(name, f"ToolAlpaca function {name}."),
            "parameters": schema_from_examples(arg_examples.get(name, [])),
        }

    if limit and limit > 0:
        rng.shuffle(records)
        records = records[:limit]

    return write_converted(out_dir, records, tool_defs)
