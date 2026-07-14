"""
Generic simulated tools for external tool-calling benchmarks.

The in-house benchmark maps each tool name to a concrete Python class in
``agentic.tools.TOOL_CLASS_MAP`` (a real remote-sensing CLI).  External
datasets (ToolAlpaca, Seal-Tools, API-Bank) reference *hundreds* of arbitrary
tool names with no local implementation.  Writing a class per tool is neither
feasible nor necessary: the trajectory-completion metric
(``experiments.trajectory_utils.evaluate_completion``) scores which **tool
names** the agent selects against the ground-truth sequence — the tool's return
*content* only needs to be a plausible, non-crashing string so the agent loop
keeps progressing and the LLM judge has something readable to read.

``GenericTool`` provides exactly that: a deterministic simulated executor.
``registry_from_defs`` assembles a :class:`~agentic.tools.ToolRegistry` from a
list of tool-definition dicts, reusing the real implementations for the
meta-tools (``plan`` / ``task_summary`` / ``task_done``) and wrapping every
other tool in a ``GenericTool``.  Because building a registry touches no model
weights it is cheap enough to rebuild per task, which lets each task expose only
its own ``available_tools`` (keeping SLM prompts bounded).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentic.tools import (
    AgentTool,
    ToolEntry,
    ToolRegistry,
    TOOL_CLASS_MAP,
    _instantiate_tool,
    WaitUserInstruction,
)

# Meta-tools that the PTC-Decoder method always needs, regardless of dataset.
# They have real implementations in ``TOOL_CLASS_MAP`` and are appended to every
# per-task registry even when the task's ``available_tools`` omits them.
META_TOOLS = ("plan", "task_summary", "task_done")


class GenericTool(AgentTool):
    """A simulated tool that returns a deterministic, plausible success string.

    Used for external-dataset tools that have no local implementation.  The
    returned text is intentionally generic; it never affects the rule-based
    tool-selection score (which only inspects *which* tools were called), it
    merely keeps the agent loop moving and gives the LLM judge readable output.

    An optional ``canned_response`` (e.g. a sample response shipped with the
    dataset) is echoed when provided.
    """

    def __init__(self, name: str, canned_response: Optional[str] = None) -> None:
        super().__init__()
        self._name = name
        self._canned = canned_response

    def get_name(self) -> str:
        return self._name

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {"_raw": arguments}
        if self._canned:
            return self._canned
        arg_preview = json.dumps(args, ensure_ascii=False)
        if len(arg_preview) > 300:
            arg_preview = arg_preview[:300] + "…"
        return f"[{self._name}] executed successfully with arguments {arg_preview}."


def _normalize_parameters(params: Any) -> Dict[str, Any]:
    """Coerce a tool's parameter spec into a valid JSON-Schema object.

    External datasets provide parameter schemas in varying shapes; the
    constrained decoder expects an object schema with ``properties``/``required``.
    Anything unrecognized degrades to a permissive empty object.
    """
    if isinstance(params, dict) and params.get("type") == "object":
        params.setdefault("properties", {})
        params.setdefault("required", [])
        return params
    if isinstance(params, dict) and "properties" in params:
        params.setdefault("type", "object")
        params.setdefault("required", [])
        return params
    return {"type": "object", "properties": {}, "required": []}


def registry_from_defs(
    all_defs: Dict[str, Dict[str, Any]],
    available: Optional[List[str]] = None,
    *,
    benchmark_mode: bool = True,
    canned_responses: Optional[Dict[str, str]] = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` from tool-definition dicts.

    Args:
        all_defs: Mapping ``tool_name -> {name, description, parameters, ...}``
            (the union tool pool for a dataset, keyed by name).
        available: Subset of tool names to expose for this task.  ``None`` means
            expose every tool in ``all_defs``.  The meta-tools
            (``plan``/``task_summary``/``task_done``) are always added.
        benchmark_mode: Forwarded to ``WaitUserInstruction`` so it never blocks
            on ``input()`` during automated runs.
        canned_responses: Optional ``name -> response`` overrides for GenericTool.

    Returns:
        A fully registered ``ToolRegistry``.
    """
    WaitUserInstruction.benchmark_mode = benchmark_mode
    canned_responses = canned_responses or {}

    if available is None:
        names = list(all_defs.keys())
    else:
        names = [n for n in available if n in all_defs]

    # Always include the meta-tools (needed by the plan/execution machinery).
    for meta in META_TOOLS:
        if meta not in names:
            names.append(meta)

    registry = ToolRegistry()
    for name in names:
        tdef = all_defs.get(name)
        if tdef is None:
            # Meta-tool not present in the dataset pool — fall back to the
            # canonical definition shipped in the repo's tools.json.
            tdef = _CANONICAL_META_DEFS.get(name)
            if tdef is None:
                continue

        cls = TOOL_CLASS_MAP.get(name)
        if cls is not None:
            tool_instance: AgentTool = _instantiate_tool(cls, name, "my_env", "", "")
        else:
            tool_instance = GenericTool(name, canned_responses.get(name))

        entry = ToolEntry(
            tool=tool_instance,
            name=name,
            description=tdef.get("description", f"Tool {name}."),
            parameters=_normalize_parameters(tdef.get("parameters", {})),
        )
        registry.register(entry)

    registry._finalize()
    return registry


# ---------------------------------------------------------------------------
# Canonical meta-tool definitions (verbatim schema from the repo's tools.json).
# Used as a fallback so a dataset whose tools.json somehow omits a meta-tool
# still gets a valid schema for constrained decoding.
# ---------------------------------------------------------------------------

_CANONICAL_META_DEFS: Dict[str, Dict[str, Any]] = {
    "task_done": {
        "name": "task_done",
        "description": "Task complete. No additional parameters. Calling this signals that the task has ended.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "task_summary": {
        "name": "task_summary",
        "description": (
            "Task summary tool. Produce a logical summary of the entire execution process of "
            "the current task before calling task_done. The summary should include key steps "
            "executed, core data obtained, and the final conclusion for the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary_title": {"type": "string", "description": "Title of the summary report"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of key findings or completed steps",
                },
                "final_conclusion": {
                    "type": "string",
                    "description": "The final concluding statement, to be relayed directly to the user",
                },
            },
            "required": ["summary_title", "final_conclusion"],
        },
    },
    "plan": {
        "name": "plan",
        "description": (
            "Decompose and plan a complex task into multiple clear, actionable steps to be "
            "executed one by one. Use this before initiating other tool calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title"},
                "steps": {
                    "type": "array",
                    "description": "List of planned task execution steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer", "description": "Step number"},
                            "step_name": {"type": "string", "description": "Step name"},
                            "description": {"type": "string", "description": "Detailed description of this step"},
                            "expected_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of expected tool names to use",
                            },
                        },
                        "required": ["step_number", "step_name", "description"],
                    },
                },
            },
            "required": ["title", "steps"],
        },
    },
}


def canonical_meta_defs() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the canonical meta-tool definitions for converters."""
    import copy

    return copy.deepcopy(_CANONICAL_META_DEFS)
