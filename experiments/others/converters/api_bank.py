"""
API-Bank → evaluate.json / tools.json.

Source: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank  (MIT)
Files:  api-bank/lv1-lv2-samples/level-1-given-desc/*.jsonl
        api-bank/lv1-lv2-samples/level-2-toolsearcher/*.jsonl

Each raw file is a multi-turn dialog with ``User`` / ``AI`` / ``API`` turns.  We
flatten the ``User`` turns into a single task request; the ground-truth domain
tools are the ordered ``api_name`` values of the ``API`` turns.  API-Bank ships
its tools as Python source, so parameter schemas are *inferred* from the observed
``param_dict`` values (robust and fully offline).  Each task exposes its
ground-truth tools plus sampled distractors from the global API pool.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

from experiments.others.converters import (
    build_eval_record, schema_from_examples, parse_json_loose,
    write_converted, read_jsonl, dedup,
)

# GitHub contents API dirs to enumerate for downloads (used by prepare_data).
_API = "https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents"
LIST_DIRS = [
    "api-bank/lv1-lv2-samples/level-1-given-desc",
    "api-bank/lv1-lv2-samples/level-2-toolsearcher",
]
SOURCES: List[Tuple[str, str]] = []  # dynamic — see remote_file_urls()


def remote_file_urls(limit: int = 0, opener=None) -> List[Tuple[str, str]]:
    """Enumerate dialog .jsonl download URLs via the GitHub contents API.

    Returns ``[(download_url, local_relpath), ...]``.  ``limit`` caps the number
    of files (0 = all).  ``opener`` is a callable ``url -> bytes`` (injected by
    prepare_data so this module stays import-light).
    """
    if opener is None:
        import urllib.request

        def opener(url: str) -> bytes:  # noqa
            req = urllib.request.Request(url, headers={"User-Agent": "ptc-decoder-eval"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()

    out: List[Tuple[str, str]] = []
    for d in LIST_DIRS:
        try:
            listing = json.loads(opener(f"{_API}/{d}").decode("utf-8"))
        except Exception:
            continue
        subdir = d.split("/")[-1]
        for item in listing:
            if item.get("type") == "file" and item.get("name", "").endswith(".jsonl"):
                out.append((item["download_url"], os.path.join(subdir, item["name"])))
    if limit and limit > 0:
        out = out[:limit]
    return out


def _iter_dialog_files(raw_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True))


def _stringify(value: Any) -> str:
    """Render an API ``result.output`` as a compact reference string."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def convert(
    raw_dir: str,
    out_dir: str,
    limit: int = 0,
    seed: int = 0,
    n_distractors: int = 10,
) -> Dict[str, int]:
    rng = random.Random(seed)

    parsed: List[Dict[str, Any]] = []          # per-dialog {question, gt, expected, responses}
    arg_examples: Dict[str, List[Dict[str, Any]]] = {}

    for path in _iter_dialog_files(raw_dir):
        turns = read_jsonl(path)
        if not turns:
            continue
        user_texts = [t.get("text", "") for t in turns
                      if t.get("role") == "User" and t.get("text")]
        api_turns = [t for t in turns
                     if t.get("role") == "API" and t.get("api_name")]
        if not user_texts or not api_turns:
            continue
        gt = dedup([t["api_name"] for t in api_turns])
        responses: Dict[str, str] = {}
        for t in api_turns:
            name = t["api_name"]
            arg_examples.setdefault(name, []).append(parse_json_loose(t.get("param_dict")))
            # real reference output (first occurrence per api) — includes tokens etc.
            result = t.get("result") or {}
            if name not in responses and isinstance(result, dict) and result.get("output") is not None:
                responses[name] = _stringify(result["output"])
        ai_texts = [t.get("text", "") for t in turns if t.get("role") == "AI" and t.get("text")]
        parsed.append({
            "question": " ".join(user_texts),
            "gt": gt,
            "expected": ai_texts[-1] if ai_texts else "",
            "responses": responses,
        })

    if limit and limit > 0:
        rng.shuffle(parsed)
        parsed = parsed[:limit]

    pool_names = sorted(arg_examples.keys())

    records: List[Dict[str, Any]] = []
    used: set = set()
    for p in parsed:
        gt = p["gt"]
        if not gt:
            continue
        excl = set(gt)
        distractors = [n for n in pool_names if n not in excl]
        rng.shuffle(distractors)
        distractors = distractors[:n_distractors]
        available = dedup(gt + distractors)
        used.update(available)
        records.append(build_eval_record(
            question=p["question"],
            gt_domain_tools=gt,
            available_tools=available,
            expected=p["expected"],
            tool_responses=p["responses"],
        ))

    tool_defs = {
        name: {
            "name": name,
            "description": f"API-Bank tool `{name}`. Parameter schema inferred from reference calls.",
            "parameters": schema_from_examples(arg_examples.get(name, [])),
        }
        for name in used
    }
    return write_converted(out_dir, records, tool_defs)
