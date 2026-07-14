"""
Deterministic mock backend for offline pipeline validation.

The real :class:`agentic.llm_backend.LLMBackend` loads a HuggingFace SLM and
requires a working torch/CUDA stack.  This mock is a drop-in, duck-typed
replacement that runs entirely on CPU with no heavy dependencies, so the full
experiment pipeline (planning subtask → constrained/free agent loop → trajectory
save → aggregation) can be exercised end-to-end in CI or on a machine where
torch is unavailable.

It is an *oracle-ish* stub: given the ground-truth domain tools for a task (set
via :meth:`MockBackend.set_task_context`), it deterministically walks
``[domain tools…] → task_summary → task_done``.  The numbers it produces are
therefore meaningless as a scientific result — its only purpose is to prove the
plumbing is correct.  Real results come from ``--backend hf`` with real SLMs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Minimal tokenizer / decoder-result stand-ins
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Whitespace tokenizer good enough for token-count bookkeeping."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, return_tensors: Optional[str] = None) -> List[int]:
        toks = (text or "").split()
        return list(range(len(toks)))


@dataclass
class MockDecoderResult:
    """Mirrors ``constrained_decoding.DecoderResult`` fields the runners read."""

    tool_call: Dict[str, Any]
    text: str
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockBackend:
    """Duck-typed replacement for ``LLMBackend`` (no torch, deterministic)."""

    # Qwen-style wrappers so parse_tool_call round-trips with the real parser.
    tool_call_prefix = "<tool_call>"
    tool_call_suffix = "</tool_call>"

    def __init__(self, model_id: str = "mock/deterministic-oracle", **_: Any) -> None:
        self.model_id = model_id
        self.tokenizer = MockTokenizer()
        # Per-task scripted plan: domain tools then closure tools.
        self._script: List[str] = ["task_summary", "task_done"]
        self._free_idx = 0

    # -- task context ---------------------------------------------------------

    def set_task_context(self, domain_tools: List[str]) -> None:
        """Reset the free-generation script for a new task.

        Args:
            domain_tools: ordered ground-truth domain tools for this task
                (meta-tools excluded).  The mock will "call" these, then
                ``task_summary`` and ``task_done``.
        """
        dedup: List[str] = []
        for t in domain_tools:
            if t and t not in dedup and t not in ("plan", "task_summary", "task_done"):
                dedup.append(t)
        self._script = dedup + ["task_summary", "task_done"]
        self._free_idx = 0

    # -- prompt building ------------------------------------------------------

    def build_prompt(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        parts = [f"[TOOLS:{len(tools)}]"]
        for m in messages:
            parts.append(f"{m.get('role', '')}: {m.get('content', '')}")
        return "\n".join(parts)

    # -- free generation ------------------------------------------------------

    def generate_free(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Return the next scripted tool call (as a wrapped tool-call string)."""
        if self._free_idx < len(self._script):
            name = self._script[self._free_idx]
        else:
            name = "task_done"
        self._free_idx += 1
        return self.format_tool_call_message(name, self._dummy_args(name))

    # -- constrained generation ----------------------------------------------

    def generate_constrained(
        self,
        prompt: str,
        tool_name: str,
        args_schema: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> MockDecoderResult:
        if tool_name == "plan":
            args = self._plan_args()
        else:
            args = self._dummy_args_from_schema(args_schema)
        text = self.format_tool_call_message(tool_name, args)
        return MockDecoderResult(
            tool_call={"name": tool_name, "arguments": args},
            text=text,
            token_ids=list(range(max(1, len(text.split())))),
            finish_reason="stop",
        )

    # -- parsing / formatting -------------------------------------------------

    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        body = text
        i = body.find(self.tool_call_prefix)
        if i >= 0:
            body = body[i + len(self.tool_call_prefix):]
        j = body.rfind(self.tool_call_suffix)
        if j >= 0:
            body = body[:j]
        start = body.find("{")
        if start == -1:
            return None
        depth, end = 0, -1
        for k, ch in enumerate(body[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        chunk = body[start:end + 1] if end >= 0 else body[start:]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None

    def format_tool_call_message(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        payload = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
        return f"{self.tool_call_prefix}{payload}{self.tool_call_suffix}"

    # -- helpers --------------------------------------------------------------

    def _plan_args(self) -> Dict[str, Any]:
        """A well-formed plan whose expected_tools cover the scripted tools."""
        domain = [t for t in self._script if t not in ("task_summary", "task_done")]
        steps = []
        n = 1
        for t in domain:
            steps.append({
                "step_number": n,
                "step_name": f"Call {t}",
                "description": f"Invoke {t} to make progress on the task.",
                "expected_tools": [t],
            })
            n += 1
        steps.append({
            "step_number": n,
            "step_name": "Summarize the work",
            "description": "Summarize the completed work for the user.",
            "expected_tools": ["task_summary"],
        })
        steps.append({
            "step_number": n + 1,
            "step_name": "End the task",
            "description": "Call task_done to finish.",
            "expected_tools": ["task_done"],
        })
        return {"title": "Mock execution plan", "steps": steps}

    def _dummy_args(self, tool_name: str) -> Dict[str, Any]:
        if tool_name == "task_summary":
            return {
                "summary_title": "Task summary",
                "final_conclusion": "Completed the requested tool calls (mock run).",
            }
        return {}

    def _dummy_args_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fill required properties with type-appropriate placeholders."""
        if not isinstance(schema, dict):
            return {}
        props = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        out: Dict[str, Any] = {}
        for key in required:
            spec = props.get(key, {}) if isinstance(props, dict) else {}
            out[key] = self._placeholder(spec)
        return out

    def _placeholder(self, spec: Dict[str, Any]) -> Any:
        t = spec.get("type") if isinstance(spec, dict) else None
        if t == "integer":
            return 1
        if t == "number":
            return 1.0
        if t == "boolean":
            return True
        if t == "array":
            return []
        if t == "object":
            return {}
        return "placeholder"
