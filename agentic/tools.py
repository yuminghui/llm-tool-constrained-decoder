"""
Agent tool system — JSON-driven, extensible.

Tools are defined in ``tools.json`` with an ``enabled`` flag.
Adding/changing tools only requires editing ``tools.json`` and adding
the implementation class to this file.

Architecture::

    tools.json  ──→  load_tools_from_json()  ──→  ToolRegistry
                         │
                         ├── reads metadata (name, description, parameters, enabled)
                         ├── looks up class in TOOL_CLASS_MAP
                         ├── instantiates only ``enabled: true`` tools
                         └── wraps each in a ToolEntry

Usage::

    from agentic.tools import load_tools_from_json

    registry = load_tools_from_json("tools.json")
    definitions = registry.get_definitions()   # for chat template
    result = registry.execute("get_time", {})  # run a tool
"""

from __future__ import annotations

import json
import os
import re
import sys
import glob as glob_module
import datetime
import subprocess
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Logging — self-contained replacement for utils.log_util.get_logger
# ---------------------------------------------------------------------------

def _get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create or retrieve a logger with a console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = _get_logger(__name__, level=logging.DEBUG)


# ---------------------------------------------------------------------------
# app2_summary — LLM-based remote sensing output summarization
# ---------------------------------------------------------------------------

APP2_SUMMARY_PROMPT = """# 用户任务
{query}

# 遥感指数分析工具计算出的 JSON 数据
{json_content}

# 你的任务
你的任务是基于遥感指数分析工具计算出的 JSON 数据，对用户任务进行分析总结，并且只能返回分析总结后的结果。

# 要求
- 只需要返回分析总结的内容
- 以Markdown格式
- 请使用中文"""

_summary_backend = None  # set by experiment via set_summary_backend()


def set_summary_backend(backend) -> None:
    """Set the LLM backend used by ``_app2_summary`` for remote sensing summaries.

    Call once per model in the experiment loop.  If never called (or set to
    None), ``_app2_summary`` returns None and callers fall back to raw output.
    """
    global _summary_backend
    _summary_backend = backend


def _app2_summary(model: str, query: str, json_content: str) -> Optional[str]:
    """Summarize remote sensing index calculation output via LLM.

    Uses the backend set by ``set_summary_backend()``.  Returns None if no
    backend is available, so callers fall back to the raw tool output.
    """
    if _summary_backend is None:
        return None
    prompt = APP2_SUMMARY_PROMPT.format(query=query, json_content=json_content)
    try:
        raw = _summary_backend.generate_free(
            prompt, max_new_tokens=512, temperature=0.3,
        )
        return raw.strip() if raw else None
    except Exception:
        logger.exception("_app2_summary failed")
        return None


# ===================================================================
# AgentTool — abstract base class
# ===================================================================

class AgentTool(ABC):
    """Abstract base class for all agent tools.

    Subclasses must implement ``get_name()`` and ``execute()``.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name (must match the name in tools.json)."""
        ...

    @abstractmethod
    def execute(self, arguments: str) -> str:
        """Execute the tool with JSON-string arguments, return a string result."""
        ...

    def auto_execute(self) -> bool:
        """Whether the agent may auto-execute this tool without user confirmation."""
        return True

    def needs_sandbox(self) -> bool:
        """Whether this tool requires sandbox isolation (file writes, subprocess)."""
        return False

    def get_output_paths(self, arguments: str) -> list:
        """Return parameter keys whose values are output file paths (for sandbox)."""
        return []

    def get_input_paths(self, arguments: str) -> list:
        """Return parameter keys whose values are input file paths (for sandbox)."""
        return []


# ===================================================================
# Tool implementations — one class per entry in tools.json
# ===================================================================

class GetWeather(AgentTool):
    def get_name(self) -> str:
        return "get_weather"

    def execute(self, arguments: str) -> str:
        return "天气晴朗，温度22摄氏度"


class GetTime(AgentTool):
    def get_name(self) -> str:
        return "get_time"

    def execute(self, arguments: str) -> str:
        now = datetime.datetime.now()
        return f"当前时间为：{now.year}-{now.month:02d}-{now.day:02d}"


class TaskDone(AgentTool):
    """Signal that the task is finished. No-op that returns a confirmation."""

    def get_name(self) -> str:
        return "task_done"

    def execute(self, arguments: str) -> str:
        return "任务已完成。"


class ChatResponse(AgentTool):
    def get_name(self) -> str:
        return "chat_response"

    def execute(self, arguments: str) -> str:
        try:
            tmp = json.loads(arguments)
            if "text" not in tmp:
                return "-1"
            return "0"
        except Exception:
            return "-1"


class TaskSummary(AgentTool):
    def get_name(self) -> str:
        return "task_summary"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        task_summary = f"# {args['summary_title']}\n"
        if "key_findings" in args:
            task_summary += "## 关键发现"
            for finding in args["key_findings"]:
                task_summary += f"\n- {finding}"
        task_summary += "\n"
        task_summary += args["final_conclusion"]
        return task_summary


class PathWorkspace(AgentTool):
    def get_name(self) -> str:
        return "path_workspace"

    def execute(self, arguments: str) -> str:
        return "/var/opt"


class ListSenseIndex(AgentTool):
    def get_name(self) -> str:
        return "list_remote_sensing_index"

    def execute(self, arguments: str) -> str:
        return """
## 支持的指数类型 ##
- 植被指数: ndvi, savi, evi, gndvi, arvi, msavi, lai, sr, osavi, rdvi, cvi, cigreen, grvi, gli, vari
- 水体指数: ndwi, mndwi, awei_nsh, awei_sh, ndti
- 土壤指数: gsi, ci, bi, bi2, ri, sci, ndshi, ndsvi
- 藻类指数: fai, ndci, sabi, atbi, cig
- 燃烧/色素/建筑指数: bai, nbr, sipi, vgnir_bi, building_bi
"""


class RemoteSenseIndexCalculate(AgentTool):
    """Remote sensing index calculation via subprocess call."""

    def __init__(self, conda_env: str = "my_env", query: str = "", model: str = "") -> None:
        super().__init__()
        self.conda_env = conda_env
        self.query = query
        self.model = model
        self.success_flag = "全部任务成功"
        self.success = "运行成功，文件保存到{output_path}"
        self.failed = "运行失败, 工具日志如下:\n{error}"

    def get_name(self) -> str:
        return "remote_sensing_calculate_cli"

    def needs_sandbox(self) -> bool:
        return True

    def get_output_paths(self, arguments: str) -> list:
        return ["output_path"]

    def get_input_paths(self, arguments: str) -> list:
        return ["input_path"]

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        index = args.get("index")
        workers = args.get("workers")

        if not input_path or not index:
            return "input_path and index are required parameters"
        if not isinstance(index, list):
            return "index must be a list of strings"

        index_para = ",".join(index) if len(index) > 1 else index[0]
        output_path = output_path if output_path else ""
        workers_para = f"--workers {workers}" if workers else ""

        cmd_str = (
            f"conda run -n {self.conda_env} python /var/opt/my_app_2/main.py "
            f"{input_path} {output_path} {index_para} {workers_para}"
        )
        logger.debug("Will run command: %s", cmd_str)

        try:
            result = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True,
                cwd="/var/opt/my_app_2",
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            if self.success_flag in output:
                stats_path = output_path[:-len(".tif")] + ".stats.json"
                if os.path.isfile(stats_path):
                    with open(stats_path, "r", encoding="utf-8") as f:
                        json_content = json.dumps(json.load(f), ensure_ascii=False)
                    summary = _app2_summary(self.model, self.query, json_content)
                    return summary or self.success.format(output_path=output_path)
                else:
                    logger.warning(".stats.json not found: %s", stats_path)
                    return self.success.format(output_path=output_path)
            return self.failed.format(error=output)
        except Exception as e:
            return self.failed.format(error=f"执行命令失败: {str(e)}")


class BashLs(AgentTool):
    def get_name(self) -> str:
        return "bash_ls"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        target_path = args.get("target_path", "")
        show_hidden = args.get("show_hidden", False)
        show_detail = args.get("show_detail", False)
        pattern = args.get("pattern")

        show_hidden_para = "-a" if show_hidden else ""
        show_detail_para = "-l" if show_detail else ""
        pattern_para = f'| grep -E "{pattern}"' if pattern else ""
        cmd_str = f"ls {target_path} {show_hidden_para} {show_detail_para} {pattern_para}"
        logger.debug("Will run command: %s", cmd_str)

        try:
            result = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt",
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"
            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"


class GFPMSPreprocess(AgentTool):
    """Gaofen satellite image preprocessing pipeline."""

    def __init__(self, conda_env: str = "my_env") -> None:
        super().__init__()
        self.conda_env = conda_env

    def get_name(self) -> str:
        return "gf_pms_preprocess_cli"

    def needs_sandbox(self) -> bool:
        return True

    def get_output_paths(self, arguments: str) -> list:
        return ["output_path"]

    def get_input_paths(self, arguments: str) -> list:
        return ["input_path"]

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        case = args.get("case")
        mem = args.get("mem", "balanced")

        if not input_path or not output_path or case is None:
            return "`input_path`, `output_path`, `case` are required parameters"
        if not isinstance(case, int):
            return "`case` must be an integer"
        if mem not in ("balanced", "low", "ultra_low", "high"):
            return "`mem` must be one of balanced, low, ultra_low, high"
        if case not in (1, 2, 3, 4):
            return "`case` must be one of 1, 2, 3, 4"

        cmd_str = (
            f"conda run -n {self.conda_env} python /var/opt/my_app_1/main.py "
            f"{input_path} {output_path} case={case} mem={mem}"
        )
        logger.debug("Will run command: %s", cmd_str)

        try:
            result = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True,
                cwd="/var/opt/my_app_1",
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            logger.debug("app1 full log: %s", output)

            fin_index = output.rfind("批量处理完成")
            if fin_index != -1:
                output = output[fin_index:]
                output = f"{output}, 所有成功结果已保存到{output_path}中"
            else:
                error_index = output.rfind("ERROR")
                if error_index != -1:
                    output = output[error_index:]
                else:
                    output = "批量处理失败"
            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"


class GetToolsList(AgentTool):
    """Return the list of available tool names.

    ``tools_map`` is injected after construction by the registry.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tools_map: Dict[str, Any] = {}

    def set_tools_map(self, tools_map: Dict[str, Any]) -> None:
        self._tools_map = tools_map

    def get_name(self) -> str:
        return "get_tools_list"

    def execute(self, arguments: str) -> str:
        return json.dumps(list(self._tools_map.keys()))


class DcvaCD(AgentTool):
    """DCVA change detection tool."""

    def __init__(self, conda_env: str = "my_env") -> None:
        super().__init__()
        self.conda_env = conda_env

    def get_name(self) -> str:
        return "dcva_cd"

    def needs_sandbox(self) -> bool:
        return True

    def get_output_paths(self, arguments: str) -> list:
        return ["output_path"]

    def get_input_paths(self, arguments: str) -> list:
        return ["input_path", "img2", "gt"]

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        img2 = args.get("img2")
        gt = args.get("gt")
        limit = args.get("limit")

        if not input_path or not output_path:
            return "`input_path`, `output_path` are required parameters"
        if limit is not None and not isinstance(limit, int):
            return "`limit` must be an integer"

        cmd_str = (
            f"conda run -n {self.conda_env} python /var/opt/my_app_4/main.py "
            f"{input_path} {output_path}"
        )
        if img2:
            cmd_str += f" --img2 {img2}"
        if gt:
            cmd_str += f" --gt {gt}"
        if limit is not None:
            cmd_str += f" --limit {limit}"
        logger.debug("Will run command: %s", cmd_str)

        try:
            result = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True,
                cwd="/var/opt/my_app_4",
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            fin_idx = output.rfind("检测完成!")
            if fin_idx != -1:
                output = output[fin_idx:]
            else:
                output = f"工具调用失败: {output}"
                logger.error(output)
            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"


class WaitUserInstruction(AgentTool):
    def get_name(self) -> str:
        return "wait_user_instruction"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        tip = args.get("tip")
        if not tip:
            return "`tip` is required parameter"
        result = input(tip + "\n> ")
        return result


class DirSearch(AgentTool):
    def __init__(self) -> None:
        super().__init__()
        self.excluded_dirs = [
            "__pycache__", "build", "dist", ".devops", ".claude",
            ".gemini", ".github", ".vscode", ".idea",
        ]
        self.workspace = "/var/opt"

    def get_name(self) -> str:
        return "dir_search"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)

        results = []
        for root, dirs, _ in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            for dirname in dirs:
                if case_sensitive:
                    if keyword not in dirname:
                        continue
                else:
                    if keyword.lower() not in dirname.lower():
                        continue
                full_path = os.path.join(root, dirname)
                results.append(full_path)

        if not results:
            return "The followings are the results:\n\nNo directories found matching the keyword."

        result_str = "The followings are the results:\n"
        for path in results:
            result_str += f"{path}\n"
        return result_str


class FileSearch(AgentTool):
    def __init__(self) -> None:
        super().__init__()
        self.excluded_exts = {".pyc", ".pyd", ".so", ".o", ".a", ".bin", ".log"}
        self.excluded_dirs = [
            "__pycache__", "build", "dist", ".devops", ".claude",
            ".gemini", ".github", ".vscode", ".idea",
        ]
        self.workspace = "/var/opt"

    def get_name(self) -> str:
        return "file_search"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)

        results = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            for filename in files:
                if case_sensitive:
                    if keyword not in filename:
                        continue
                else:
                    if keyword.lower() not in filename.lower():
                        continue

                _, ext = os.path.splitext(filename)
                if ext in self.excluded_exts:
                    continue

                full_path = os.path.join(root, filename)
                results.append(full_path)

        if not results:
            return "The followings are the results:\n\nNo files found matching the keyword."

        result_str = "The followings are the results:\n"
        for path in results:
            result_str += f"{path}\n"
        return result_str


class Plan(AgentTool):
    def get_name(self) -> str:
        return "plan"

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return "规划提交失败：参数格式无效，请提供合法的JSON"

        title = args.get("title", "未命名规划")
        steps = args.get("steps", [])

        if not steps:
            return "规划提交失败：steps 不能为空，请至少提供一个步骤"

        # Backward-compatible: if steps are plain strings, convert to objects
        if steps and isinstance(steps[0], str):
            steps = [
                {"step_number": i + 1, "step_name": s, "description": s}
                for i, s in enumerate(steps)
            ]

        plan_text = f"# 📋 任务规划：{title}\n\n"
        plan_text += f"**总体目标**：{title}\n\n"
        plan_text += "## 执行步骤\n\n"

        for step in steps:
            if isinstance(step, str):
                step = {"step_name": step, "description": step}
            step_num = step.get("step_number", "?")
            step_name = step.get("step_name", "未命名步骤")
            description = step.get("description", "")
            expected_tools = step.get("expected_tools", [])

            plan_text += f"### 步骤 {step_num}：{step_name}\n"
            plan_text += f"- **描述**：{description}\n"
            if expected_tools:
                tools_str = "、".join(expected_tools) if isinstance(expected_tools, list) else str(expected_tools)
                plan_text += f"- **预计使用工具**：{tools_str}\n"
            plan_text += "\n"

        plan_text += "---\n"
        plan_text += "规划已记录。请严格按照上述步骤顺序执行。"
        plan_text += "执行过程中如需调整规划，可再次调用 plan 工具更新。"
        return plan_text


class SearchTifData(AgentTool):
    """Search for satellite TIF files by date and region."""

    def __init__(self) -> None:
        super().__init__()
        self.data_dir = "/var/opt/data"

    def get_name(self) -> str:
        return "search_tif_data"

    def execute(self, arguments: str) -> str:
        args = json.loads(arguments)
        year = args.get("year")
        region = args.get("region")
        latest_descending = args.get("latest_descending", True)

        if not year or not region:
            return "`year` and `region` are required parameters"

        month = args.get("month")
        day = args.get("day")

        if day and month:
            date_pattern = f"{year}{month}{day}"
        elif month:
            date_pattern = f"{year}{month}??"
        else:
            date_pattern = f"{year}????"

        glob_pattern = f"{date_pattern}-{region}-*.tif"
        logger.debug("SearchTifData glob pattern: %s", glob_pattern)

        results = []
        try:
            search_path = os.path.join(self.data_dir, glob_pattern)
            matched = glob_module.glob(search_path)
            results = [os.path.abspath(p) for p in matched]
        except Exception as e:
            logger.error("SearchTifData error: %s", str(e))
            return f"搜索过程发生错误: {str(e)}"

        if not results:
            return "The followings are the results:\n\nNo tif files found matching the criteria."

        def _extract_date(path: str) -> str:
            basename = os.path.basename(path)
            match = re.match(r"(\d{8})", basename)
            return match.group(1) if match else ""

        results.sort(key=_extract_date, reverse=latest_descending)

        result_str = "The followings are the results:\n"
        for path in results:
            result_str += f"{path}\n"
        return result_str


# ===================================================================
# Tool class registry — maps tool name → class
# ===================================================================

TOOL_CLASS_MAP: Dict[str, Type[AgentTool]] = {
    "get_weather":                  GetWeather,
    "get_time":                     GetTime,
    "task_done":                    TaskDone,
    "chat_response":                ChatResponse,
    "task_summary":                 TaskSummary,
    "path_workspace":               PathWorkspace,
    "list_remote_sensing_index":    ListSenseIndex,
    "remote_sensing_calculate_cli": RemoteSenseIndexCalculate,
    "bash_ls":                      BashLs,
    "gf_pms_preprocess_cli":        GFPMSPreprocess,
    "get_tools_list":               GetToolsList,
    "dcva_cd":                      DcvaCD,
    "file_search":                  FileSearch,
    "wait_user_instruction":        WaitUserInstruction,
    "dir_search":                   DirSearch,
    "plan":                         Plan,
    "search_tif_data":              SearchTifData,
}


# ===================================================================
# ToolEntry — wraps an AgentTool with its JSON metadata
# ===================================================================

@dataclass
class ToolEntry:
    """Wraps an AgentTool instance with its metadata from tools.json.

    Provides a uniform interface regardless of the underlying tool class.
    """
    tool: AgentTool
    name: str
    description: str
    parameters: Dict[str, Any]

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the tool with a dict of arguments.

        Converts the dict to a JSON string (as expected by AgentTool.execute)
        and returns the string result.
        """
        json_args = json.dumps(arguments, ensure_ascii=False)
        return self.tool.execute(json_args)

    def auto_execute(self) -> bool:
        return self.tool.auto_execute()

    def needs_sandbox(self) -> bool:
        return self.tool.needs_sandbox()

    def get_output_paths(self, arguments: Dict[str, Any]) -> list:
        return self.tool.get_output_paths(json.dumps(arguments, ensure_ascii=False))

    def get_input_paths(self, arguments: Dict[str, Any]) -> list:
        return self.tool.get_input_paths(json.dumps(arguments, ensure_ascii=False))


# ===================================================================
# ToolRegistry — manages a collection of tools
# ===================================================================

class ToolRegistry:
    """Manages tool entries: lookup, definition export, and execution."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        self._tools[entry.name] = entry
        logger.debug("Registered tool: %s", entry.name)

    def get(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Definitions for chat template
    # ------------------------------------------------------------------

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI / chat-template compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters,
                },
            }
            for entry in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Schema for constrained decoder
    # ------------------------------------------------------------------

    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Return the JSON Schema for a tool's parameters."""
        entry = self._tools.get(tool_name)
        return entry.parameters if entry else {}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name with dict arguments."""
        entry = self._tools.get(tool_name)
        if entry is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return entry.execute(arguments)
        except Exception as exc:
            logger.exception("Tool %s execution failed", tool_name)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Post-init hook (e.g., inject tools_map into get_tools_list)
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """Called after all tools are registered to run post-init wiring."""
        tools_list_tool = self._tools.get("get_tools_list")
        if tools_list_tool is not None and hasattr(tools_list_tool.tool, "set_tools_map"):
            tools_list_tool.tool.set_tools_map({
                name: entry.description for name, entry in self._tools.items()
            })


# ===================================================================
# Factory — load enabled tools from tools.json
# ===================================================================

def load_tools_from_json(
    json_path: str,
    conda_env: str = "my_env",
    query: str = "",
    model: str = "",
) -> ToolRegistry:
    """Load tools from *json_path*, instantiating only those with ``enabled: true``.

    Args:
        json_path: Path to the ``tools.json`` file.
        conda_env: Conda environment name for CLI tools.
        query: User query (passed to RemoteSenseIndexCalculate).
        model: Model name (passed to RemoteSenseIndexCalculate).

    Returns:
        A fully initialized ``ToolRegistry`` with all enabled tools registered.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tool_defs = json.load(f)

    registry = ToolRegistry()

    for tool_def in tool_defs:
        name = tool_def["name"]
        enabled = tool_def.get("enabled", True)

        if not enabled:
            logger.info("Skipping disabled tool: %s", name)
            continue

        cls = TOOL_CLASS_MAP.get(name)
        if cls is None:
            logger.warning("No implementation found for tool %r — skipping", name)
            continue

        # Instantiate with appropriate constructor args
        tool_instance = _instantiate_tool(cls, name, conda_env, query, model)

        entry = ToolEntry(
            tool=tool_instance,
            name=name,
            description=tool_def["description"],
            parameters=tool_def["parameters"],
        )
        registry.register(entry)

    registry._finalize()

    logger.info(
        "Loaded %d tools from %s (skipped %d disabled)",
        len(registry.tool_names),
        json_path,
        sum(1 for t in tool_defs if not t.get("enabled", True)),
    )
    return registry


def _instantiate_tool(
    cls: Type[AgentTool],
    name: str,
    conda_env: str,
    query: str,
    model: str,
) -> AgentTool:
    """Instantiate a tool class with the appropriate constructor arguments."""
    # Tools that accept (conda_env, query, model)
    if name == "remote_sensing_calculate_cli":
        return cls(conda_env=conda_env, query=query, model=model)
    # Tools that accept (conda_env)
    if name in ("gf_pms_preprocess_cli", "dcva_cd"):
        return cls(conda_env=conda_env)
    # All others use the default constructor
    return cls()
