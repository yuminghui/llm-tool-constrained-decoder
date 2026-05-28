"""
Plan-then-Act agent with optional constrained decoding.

The agent always forces a ``plan`` tool call as its first action (using
constrained decoding).  Subsequent steps either:

- **Constrained mode**: infer which tool to call from the plan, then force
  it via constrained decoding; after all planned tools are exhausted,
  fall back to free generation for the final answer.
- **Free mode**: use unconstrained generation; parse any tool calls that
  appear and execute them; stop when the model emits a plain-text answer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tools import ToolRegistry
from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → tool mapping for plan parsing (fallback when expected_tools absent)
# ---------------------------------------------------------------------------
TOOL_KEYWORDS: Dict[str, List[str]] = {
    "get_weather":                  ["weather", "天气", "temperature", "forecast", "climate"],
    "get_time":                     ["time", "时间", "date", "日期", "current"],
    "chat_response":                ["reply", "respond", "回复", "answer user"],
    "task_summary":                 ["summar", "总结", "summary", "conclusion"],
    "task_done":                    ["done", "finish", "完成", "结束"],
    "path_workspace":               ["workspace", "工作区", "path", "路径", "directory"],
    "list_remote_sensing_index":    ["index list", "可用指数", "supported index", "指数列表"],
    "remote_sensing_calculate_cli": ["calculat", "compute index", "计算指数", "波段", "遥感指数"],
    "bash_ls":                      ["ls", "list file", "列出文件", "directory listing"],
    "gf_pms_preprocess_cli":        ["preprocess", "预处理", "gaofen", "高分", "卫星影像预处理"],
    "get_tools_list":               ["tool list", "工具列表", "available tool"],
    "dcva_cd":                      ["change detect", "变化检测", "dcva", "两期"],
    "file_search":                  ["file search", "搜索文件", "find file"],
    "wait_user_instruction":        ["wait user", "等待用户", "user input", "confirm"],
    "dir_search":                   ["dir search", "搜索目录", "find director"],
    "search_tif_data":              ["tif", "卫星影像数据检索", "search tif", "影像数据"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """A single step in the agent's execution trace."""
    step_index: int
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_result: Optional[str]
    generated_text: str
    is_constrained: bool
    elapsed: float = 0.0


@dataclass
class AgentResult:
    """Result of a complete agent run."""
    user_query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_time: float = 0.0
    total_tokens: int = 0
    success: bool = False
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""
    max_turns: int = 10
    temperature: float = 0.7
    use_constrained_decoder: bool = True
    verbose: bool = True
    system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Plan-then-Act agent that uses an LLM backend with optional constraint.

    Usage::

        backend = LLMBackend("google/gemma-4-E2B-it")
        tools = create_default_tools()
        config = AgentConfig(use_constrained_decoder=True)
        agent = Agent(backend, tools, config)
        result = agent.run("What is 25°C in Fahrenheit?")
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_registry: ToolRegistry,
        config: AgentConfig,
    ):
        self.llm = llm_backend
        self.tools = tool_registry
        self.config = config

        self._system_prompt = config.system_prompt or (
            "You are a helpful assistant with access to external tools. "
            "Always call the 'plan' tool first to create a step-by-step plan "
            "before taking any action. Execute tools one at a time. "
            "When you have gathered all needed information, provide a final "
            "answer to the user. Respond in Chinese."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_query: str,
        *,
        pre_seeded_plan: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute the agent loop for *user_query* and return an AgentResult.

        Args:
            user_query: The user's task description.
            pre_seeded_plan: If provided, skip the plan-generation step and inject
                this plan directly into the conversation.  Must contain keys
                ``tool_name``, ``tool_args``, ``generated_text``, and optionally
                ``tool_result``.  Used for Experiment 2 (separate plan + agent).
        """
        start_time = time.time()
        result = AgentResult(user_query=user_query)

        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_query},
            ]
            tool_defs = self.tools.get_definitions()

            if pre_seeded_plan is not None:
                # ---- Plan pre-seeded: inject plan, skip step 0 ----
                plan_step = AgentStep(
                    step_index=-1,                    # -1 = external / pre-seeded
                    tool_name=pre_seeded_plan.get("tool_name", "plan"),
                    tool_args=pre_seeded_plan.get("tool_args", {}),
                    tool_result=pre_seeded_plan.get("tool_result"),
                    generated_text=pre_seeded_plan.get("generated_text", ""),
                    is_constrained=True,
                    elapsed=0.0,
                )
                result.steps.append(plan_step)
                self._append_tool_call(messages, plan_step)
                start_turn = 1
            else:
                # ---- Step 1: Force plan (always constrained) ----
                plan_step = self._force_plan(messages, tool_defs, start_time)
                result.steps.append(plan_step)
                result.total_tokens += len(plan_step.generated_text)

                if plan_step.tool_result is None:
                    result.success = False
                    result.error = "Plan step failed to produce valid tool call"
                    result.total_time = time.time() - start_time
                    return result

                # Add plan to conversation
                self._append_tool_call(messages, plan_step)
                start_turn = 1

            # ---- Steps 2+: Action loop ----
            plan_args = plan_step.tool_args or {}
            planned_tools = self._extract_tools_from_plan(plan_args)

            if self.config.use_constrained_decoder:
                result = self._run_constrained_loop(
                    result, messages, tool_defs, planned_tools, start_time,
                    start_turn=start_turn,
                )
            else:
                result = self._run_free_loop(
                    result, messages, tool_defs, start_time,
                    start_turn=start_turn,
                )

            result.total_time = time.time() - start_time
            return result

        except Exception as exc:
            logger.exception("Agent run failed")
            result.success = False
            result.error = str(exc)
            result.total_time = time.time() - start_time
            return result

    # ------------------------------------------------------------------
    # Step 1: Plan
    # ------------------------------------------------------------------

    def _force_plan(
        self,
        messages: List[Dict[str, str]],
        tool_defs: List[Dict[str, Any]],
        start_time: float,
    ) -> AgentStep:
        """Force 'plan' tool call via constrained decoder."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("STEP 1: Forcing 'plan' tool (constrained decoding)")
            print("=" * 60)

        prompt = self.llm.build_prompt(messages, tool_defs)
        plan_schema = self.tools.get_schema("plan")

        decoder_result = self.llm.generate_constrained(
            prompt=prompt,
            tool_name="plan",
            args_schema=plan_schema,
            max_new_tokens=256,
            temperature=self.config.temperature,
        )

        step = AgentStep(
            step_index=0,
            tool_name="plan",
            tool_args=decoder_result.tool_call.get("arguments", {}),
            tool_result=None,
            generated_text=decoder_result.text,
            is_constrained=True,
            elapsed=time.time() - start_time,
        )

        if self.config.verbose:
            print(f"  Finish: {decoder_result.finish_reason}")
            print(f"  Text: {decoder_result.text[:200]}...")

        # Execute plan tool
        tool_call = decoder_result.tool_call
        if "_parse_error" not in tool_call:
            step.tool_result = self.tools.execute("plan", step.tool_args or {})

            if self.config.verbose:
                args = step.tool_args or {}
                title = args.get("title", args.get("reasoning", "N/A"))
                print(f"  Plan: {title[:120]}")
                steps_list = args.get("steps", [])
                for i, s in enumerate(steps_list, 1):
                    if isinstance(s, dict):
                        print(f"    {i}. {s.get('step_name', s)}")
                    else:
                        print(f"    {i}. {s}")

        return step

    # ------------------------------------------------------------------
    # Constrained action loop
    # ------------------------------------------------------------------

    def _run_constrained_loop(
        self,
        result: AgentResult,
        messages: List[Dict[str, str]],
        tool_defs: List[Dict[str, Any]],
        planned_tools: List[str],
        start_time: float,
        start_turn: int = 1,
    ) -> AgentResult:
        """Execute remaining steps with constrained decoding.

        Iterates over *planned_tools* in order, forcing each one.
        After all planned tools are called, does one free generation for the final answer.
        """
        tool_queue = list(planned_tools)

        for turn in range(start_turn, self.config.max_turns + 1):
            if self.config.verbose:
                print(f"\n--- Turn {turn} (constrained mode) ---")

            # Try to determine which tool to force
            target_tool = tool_queue.pop(0) if tool_queue else None

            if target_tool:
                step = self._constrained_tool_call(
                    messages, tool_defs, target_tool, turn, start_time
                )
                result.steps.append(step)

                if step.tool_result is not None:
                    self._append_tool_call(messages, step)

                if self.config.verbose:
                    print(f"  Constrained → {target_tool}")
                    args_preview = json.dumps(step.tool_args, ensure_ascii=False)
                    print(f"  Args: {args_preview[:120]}")
            else:
                # All planned tools exhausted — free generation for final answer
                if self.config.verbose:
                    print("  All planned tools done, generating final answer...")

                final_text = self._free_step(messages, tool_defs)
                result.final_answer = final_text
                result.total_tokens += len(final_text)
                result.success = True

                result.steps.append(AgentStep(
                    step_index=turn,
                    tool_name=None,
                    tool_args=None,
                    tool_result=None,
                    generated_text=final_text,
                    is_constrained=False,
                    elapsed=time.time() - start_time,
                ))

                if self.config.verbose:
                    print(f"  Final answer: {final_text[:200]}...")
                return result

        # Max turns reached
        result.success = False
        result.error = "Max turns reached"
        return result

    def _constrained_tool_call(
        self,
        messages: List[Dict[str, str]],
        tool_defs: List[Dict[str, Any]],
        tool_name: str,
        turn: int,
        start_time: float,
    ) -> AgentStep:
        """Force a specific tool call via constrained decoder."""
        prompt = self.llm.build_prompt(messages, tool_defs)
        schema = self.tools.get_schema(tool_name)

        decoder_result = self.llm.generate_constrained(
            prompt=prompt,
            tool_name=tool_name,
            args_schema=schema,
            max_new_tokens=256,
            temperature=self.config.temperature,
        )

        step = AgentStep(
            step_index=turn,
            tool_name=tool_name,
            tool_args=decoder_result.tool_call.get("arguments", {}),
            tool_result=None,
            generated_text=decoder_result.text,
            is_constrained=True,
            elapsed=time.time() - start_time,
        )

        if "_parse_error" not in decoder_result.tool_call:
            step.tool_result = self.tools.execute(tool_name, step.tool_args or {})

        return step

    # ------------------------------------------------------------------
    # Free action loop
    # ------------------------------------------------------------------

    def _run_free_loop(
        self,
        result: AgentResult,
        messages: List[Dict[str, str]],
        tool_defs: List[Dict[str, Any]],
        start_time: float,
        start_turn: int = 1,
    ) -> AgentResult:
        """Execute remaining steps with free generation.

        At each turn the model may emit a tool call or a plain-text answer.
        """
        for turn in range(start_turn, self.config.max_turns + 1):
            if self.config.verbose:
                print(f"\n--- Turn {turn} (free mode) ---")

            raw_output = self._free_step(messages, tool_defs)
            result.total_tokens += len(raw_output)

            if self.config.verbose:
                print(f"  Output: {raw_output[:200]}...")

            tool_call = self.llm.parse_tool_call(raw_output)

            if tool_call and "name" in tool_call:
                # It looks like a tool call
                t_name = tool_call.get("name", "")
                t_args = tool_call.get("arguments", {})

                step = AgentStep(
                    step_index=turn,
                    tool_name=t_name,
                    tool_args=t_args,
                    tool_result=None,
                    generated_text=raw_output,
                    is_constrained=False,
                    elapsed=time.time() - start_time,
                )

                if self.tools.get(t_name):
                    step.tool_result = self.tools.execute(t_name, t_args)
                    if self.config.verbose:
                        print(f"  Tool call: {t_name}({json.dumps(t_args, ensure_ascii=False)[:100]})")
                else:
                    if self.config.verbose:
                        print(f"  Unknown tool '{t_name}' — treating as final answer")
                    result.final_answer = raw_output
                    result.success = True
                    result.steps.append(step)
                    return result

                result.steps.append(step)
                self._append_tool_call(messages, step)
            else:
                # Plain-text answer — agent is done
                if self.config.verbose:
                    print("  → Final answer (no tool call detected)")

                result.final_answer = raw_output
                result.success = True
                result.steps.append(AgentStep(
                    step_index=turn,
                    tool_name=None,
                    tool_args=None,
                    tool_result=None,
                    generated_text=raw_output,
                    is_constrained=False,
                    elapsed=time.time() - start_time,
                ))
                return result

        result.success = False
        result.error = "Max turns reached"
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_step(
        self,
        messages: List[Dict[str, str]],
        tool_defs: List[Dict[str, Any]],
    ) -> str:
        """Run one free-generation step."""
        prompt = self.llm.build_prompt(messages, tool_defs)
        return self.llm.generate_free(
            prompt=prompt,
            max_new_tokens=512,
            temperature=self.config.temperature,
        )

    def _append_tool_call(
        self,
        messages: List[Dict[str, str]],
        step: AgentStep,
    ) -> None:
        """Append an assistant tool-call message and tool-result message."""
        assistant_msg = self.llm.format_tool_call_message(
            step.tool_name or "unknown",
            step.tool_args or {},
        )
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "tool", "content": step.tool_result or ""})

    def _extract_tools_from_plan(self, plan_args: Dict[str, Any]) -> List[str]:
        """Extract tool names referenced in the plan steps.

        Prefers ``expected_tools`` from step objects (new schema).
        Falls back to keyword matching for plain-string steps (legacy schema).

        Returns a deduplicated, ordered list of tool names.
        """
        plan_steps: list = plan_args.get("steps", [])
        seen: set = set()
        result: List[str] = []

        for step in plan_steps:
            if isinstance(step, dict):
                # New schema: step has expected_tools field
                expected = step.get("expected_tools", [])
                if isinstance(expected, list):
                    for t in expected:
                        if isinstance(t, str) and t not in seen and self.tools.get(t):
                            seen.add(t)
                            result.append(t)
                # Also try keyword matching on step_name/description as fallback
                if not expected:
                    text = step.get("step_name", "") + " " + step.get("description", "")
                    self._keyword_match_tools(text.lower(), seen, result)
            elif isinstance(step, str):
                # Legacy schema: plain string steps
                self._keyword_match_tools(step.lower(), seen, result)

        return result

    def _keyword_match_tools(
        self, text: str, seen: set, result: List[str],
    ) -> None:
        """Match tool names from *text* using TOOL_KEYWORDS and append new matches."""
        for tool_name, keywords in TOOL_KEYWORDS.items():
            if tool_name in seen:
                continue
            if not self.tools.get(tool_name):
                continue
            if any(kw in text for kw in keywords):
                seen.add(tool_name)
                result.append(tool_name)
