from abc import ABC, abstractmethod

import os
import re
import json
import subprocess
import logging
import datetime
import glob as glob_module

from utils.log_util import get_logger
from agent.subtask import app2_summary

logger = get_logger(__name__, level=logging.DEBUG)

class AgentTool(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    def execute(self, arguments: str) -> str:
        """Execute the tool and return the result."""
        pass

    def auto_execute(self) -> bool:
        """Whether auto-execution is allowed."""
        return True  # default: true

    def needs_sandbox(self) -> bool:
        """Whether execution requires a sandbox (for dangerous file-write operations)."""
        return False

    def get_output_paths(self, arguments: str) -> list:
        """Return the parameter keys for output file paths (used to copy outputs after sandbox execution).
        Subclasses may override to declare output path parameter names."""
        return []

    def get_input_paths(self, arguments: str) -> list:
        """Return the parameter keys for input file paths (used to copy inputs before sandbox execution).
        Subclasses may override to declare input path parameter names."""
        return []

class GetWeather(AgentTool):
    def get_name(self) -> str:
        return "get_weather"

    def execute(self, arguments: str) -> str:
        """Get weather information."""
        return "Clear skies, temperature 22°C"

class GetTime(AgentTool):
    def get_name(self) -> str:
        return "get_time"

    def execute(self, arguments: str) -> str:
        """Get the current system time with day-level granularity. Returns format: yyyy-mm-dd"""
        now = datetime.datetime.now()
        return f"Current time: {now.year}-{now.month:02d}-{now.day:02d}"

class ChatResponse(AgentTool):
    def get_name(self) -> str:
        return "chat_response"

    def execute(self, arguments: str) -> str:
        """Reply to the user."""
        try:
            print(f"arguments: {arguments}, type: {type(arguments)}")
            tmp = json.loads(arguments)
            if "text" not in tmp:
                return "-1"
            return "0"
        except Exception:
            return "-1"

class PathWorkspace(AgentTool):
    def get_name(self) -> str:
        return "path_workspace"

    def execute(self, arguments: str) -> str:
        """Get the current workspace path."""
        # No arguments needed; returns the workspace path as a string
        return "/var/opt"

class RemoteSenseIndexCalculate(AgentTool):
    def __init__(self, conda_env: str, query: str, model: str) -> None:
        super().__init__()
        self.conda_env = conda_env
        self.query = query
        self.model = model
        self.success_flag = "All tasks succeeded"
        self.success = "Execution succeeded. Output saved to {output_path}"
        self.failed = "Execution failed. Tool log:\n{error}"

    def get_name(self) -> str:
        return "remote_sensing_calculate_cli"

    def needs_sandbox(self) -> bool:
        return True

    def get_output_paths(self, arguments: str) -> list:
        return ["output_path"]

    def get_input_paths(self, arguments: str) -> list:
        return ["input_path"]

    def execute(self, arguments: str) -> str:
        """Execute remote sensing index calculation."""
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        index = args.get("index")
        workers = args.get("workers")
        if not input_path or not index:
            return "input_path and index are required parameters"
        if not isinstance(index, list):
            return "index must be a list of strings"
        # The third-party application handles remaining defaults (None parameters)

        cmd_str = "conda run -n {conda_env} python /var/opt/my_app_2/main.py {input_path} {output_path} {index} {workers}"
        index_para = ",".join(index) if len(index) > 1 else index[0]
        output_path = output_path if output_path else ""
        workers_para = f"--workers {workers}" if workers else ""
        cmd_str = cmd_str.format(conda_env=self.conda_env, input_path=input_path, output_path=output_path, index=index_para, workers=workers_para)
        logger.debug(msg=f"Will run command: {cmd_str}")

        # Execute command
        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt/my_app_2")
            # Return command output, including stdout and stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"
            if self.success_flag in output:
                stats_path = output_path[:-len(".tif")] + ".stats.json"
                if os.path.isfile(stats_path):
                    with open(stats_path, "r", encoding="utf-8") as f:
                        json_content = json.dumps(json.load(f), ensure_ascii=False)
                    return app2_summary(self.model, self.query, json_content) or self.success.format(output_path=output_path)
                else:
                    logger.warning(msg=f".stats.json file not found: {stats_path}")
                    return self.success.format(output_path=output_path)
            return self.failed.format(error=output)
        except Exception as e:
            return self.failed.format(error=f"Command execution failed: {str(e)}")

class TaskSummary(AgentTool):
    def get_name(self) -> str:
        return "task_summary"

    def execute(self, arguments: str) -> str:
        """Summarize the task."""
        args = json.loads(arguments)
        task_summary = f'# {args["summary_title"]}\n'
        if "key_findings" in args:
            task_summary += "## Key Findings"
            for finding in args["key_findings"]:
                task_summary += f"\n- {finding}"
        task_summary += "\n"
        task_summary += args["final_conclusion"]
        return task_summary

class ListSenseIndex(AgentTool):
    def get_name(self) -> str:
        return "list_remote_sensing_index"

    def execute(self, arguments: str) -> str:
        return """
## Supported Index Types ##
- Vegetation indices: ndvi, savi, evi, gndvi, arvi, msavi, lai, sr, osavi, rdvi, cvi, cigreen, grvi, gli, vari
- Water indices: ndwi, mndwi, awei_nsh, awei_sh, ndti
- Soil indices: gsi, ci, bi, bi2, ri, sci, ndshi, ndsvi
- Algae indices: fai, ndci, sabi, atbi, cig
- Burn/Pigment/Building indices: bai, nbr, sipi, vgnir_bi, building_bi
"""

class BashLs(AgentTool):
    """
    Provides a bash ls tool.
    """
    def get_name(self) -> str:
        return "bash_ls"

    def execute(self, arguments: str) -> str:
        """Execute the ls command."""
        # ls parameters are actually optional
        args = json.loads(arguments)
        target_path = args.get("target_path")
        show_hidden = args.get("show_hidden", False)
        show_detail = args.get("show_detail", False)
        pattern = args.get("pattern")

        target_path = target_path if target_path else ""
        show_hidden_para = "-a" if show_hidden else ""
        show_detail_para = "-l" if show_detail else ""
        pattern_para = f'| grep -E "{pattern}"' if pattern else ""
        cmd_str = f"ls {target_path} {show_hidden_para} {show_detail_para} {pattern_para}"
        logger.debug(msg=f"Will run command: {cmd_str}")

        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt")
            # Return command output, including stdout and stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"
            return output
        except Exception as e:
            return f"Command execution failed: {str(e)}"

class GFPMSPreprocess(AgentTool):
    def __init__(self, conda_env: str) -> None:
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
        """Execute Gaofen satellite image preprocessing."""
        # All parameters are actually required; defaults are provided on behalf of the LLM
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        case = args.get("case")
        mem = args.get("mem", "balanced")
        if not input_path or not output_path or case is None:
            return "`input_path`, `output_path`, `case` are required parameters"
        if not isinstance(case, int):
            return "`case` must be an integer"
        if not mem in ["balanced", "low", "ultra_low", "high"]:
            return "`mem` must be one of balanced, low, ultra_low, high, default is balanced; for each mem meaning please refer to tool description"
        if not case in [1, 2, 3, 4]:
            return "`case` must be one of 1, 2, 3, 4; for each case meaning please refer to tool description"

        cmd_str = f"conda run -n {self.conda_env} python /var/opt/my_app_1/main.py {input_path} {output_path} case={case} mem={mem}"
        logger.debug(msg=f"Will run command: {cmd_str}")

        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt/my_app_1")
            # Return command output, including stdout and stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            # Log the full app1 output for debugging
            logger.debug(msg=f"app1 full execution log: {output}")

            # Extract the last occurrence of "batch processing complete" and everything after it
            fin_index = output.rfind("Batch processing complete")
            if fin_index != -1:
                output = output[fin_index:]
                output = f"{output}. All successful results saved to {output_path}"
            else:
                # Match error log info
                error_index = output.rfind("ERROR")
                if error_index != -1:
                    output = output[error_index:]
                else:
                    output = "Batch processing failed"
            return output
        except Exception as e:
            return f"Command execution failed: {str(e)}"

class GetToolsList(AgentTool):
    def __init__(self, tools_map: dict) -> None:
        super().__init__()
        self.tools_map = tools_map  # injected by the tool factory

    def get_name(self) -> str:
        return "get_tools_list"

    def execute(self, arguments: str) -> str:
        """Get the list of all available tools."""
        return json.dumps(list(self.tools_map.keys()))

class DcvaCD(AgentTool):
    def __init__(self, conda_env: str) -> None:
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
        """Execute DCVA change detection."""
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

        cmd_str = f"conda run -n {self.conda_env} python /var/opt/my_app_4/main.py {input_path} {output_path}"
        if img2:
            cmd_str += f" --img2 {img2}"
        if gt:
            cmd_str += f" --gt {gt}"
        if limit is not None:
            cmd_str += f" --limit {limit}"
        logger.debug(msg=f"Will run command: {cmd_str}")

        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt/my_app_4")
            # Return command output, including stdout and stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            fin_idx = output.rfind("Detection complete!")
            if fin_idx != -1:
                output = output[fin_idx:]
            else:
                output = f"Tool invocation failed: {output}"
                logger.error(msg=output)

            return output
        except Exception as e:
            return f"Command execution failed: {str(e)}"

class WaitUserInstruction(AgentTool):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return "wait_user_instruction"

    def execute(self, arguments: str) -> str:
        """Wait for user input."""
        args = json.loads(arguments)
        tip = args.get("tip")
        if not tip:
            return "`tip` is required parameter"
        # Wait for user input
        result = input(tip + "\n> ")
        return result

class DirSearch(AgentTool):
    def __init__(self) -> None:
        super().__init__()
        self.excluded_dirs = ["__pycache__", "build", "dist", ".devops", ".claude", ".gemini", ".github", ".vscode", ".idea"]
        self.workspace = "/var/opt"

    def get_name(self) -> str:
        return "dir_search"

    def execute(self, arguments: str) -> str:
        """Search for directories in the workspace."""
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)

        results = []
        import os

        for root, dirs, _ in os.walk(self.workspace):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for dirname in dirs:
                # Check if directory name contains the keyword
                if case_sensitive:
                    if keyword not in dirname:
                        continue
                else:
                    if keyword.lower() not in dirname.lower():
                        continue

                # Build full path
                full_path = os.path.join(root, dirname)
                results.append(full_path)

        # Build return string
        if not results:
            return "The followings are the results:\n\nNo directories found matching the keyword."

        result_str = "The followings are the results:\n"
        for path in results:
            result_str += f"{path}\n"

        return result_str

class FileSearch(AgentTool):
    def __init__(self) -> None:
        super().__init__()
        self.excluded_files = [".pyc", ".pyd", ".so", ".o", ".a", ".o", ".a", ".bin", ".log"]
        self.excluded_dirs = ["__pycache__", "build", "dist", ".devops", ".claude", ".gemini", ".github", ".vscode", ".idea"]
        self.workspace = "/var/opt"

    def get_name(self) -> str:
        return "file_search"

    def execute(self, arguments: str) -> str:
        """Search for files in the workspace."""
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)

        results = []

        for root, dirs, files in os.walk(self.workspace):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for filename in files:
                # Check if filename contains the keyword
                if case_sensitive:
                    if keyword not in filename:
                        continue
                else:
                    if keyword.lower() not in filename.lower():
                        continue

                # Check if file extension is in the exclusion list
                excluded = False
                for ext in self.excluded_files:
                    if filename.endswith(ext):
                        excluded = True
                        break
                if excluded:
                    continue

                # Build full path
                full_path = os.path.join(root, filename)
                results.append(full_path)

        # Build return string
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
        """Receive and format a task plan for subsequent execution."""
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return "Plan submission failed: invalid argument format, please provide valid JSON"

        title = args.get("title", "Untitled Plan")
        steps = args.get("steps", [])

        if not steps:
            return "Plan submission failed: steps cannot be empty, please provide at least one step"

        # Backward compatibility: if steps are plain strings, convert to object list
        if steps and isinstance(steps[0], str):
            steps = [{"step_number": i + 1, "step_name": s, "description": s} for i, s in enumerate(steps)]

        plan_text = f"# Task Plan: {title}\n\n"
        plan_text += f"**Overall Goal**: {title}\n\n"
        plan_text += "## Execution Steps\n\n"

        for step in steps:
            if isinstance(step, str):
                step = {"step_name": step, "description": step}
            step_num = step.get("step_number", "?")
            step_name = step.get("step_name", "Unnamed Step")
            description = step.get("description", "")
            expected_tools = step.get("expected_tools", [])

            plan_text += f"### Step {step_num}: {step_name}\n"
            plan_text += f"- **Description**: {description}\n"
            if expected_tools:
                tools_str = ", ".join(expected_tools) if isinstance(expected_tools, list) else str(expected_tools)
                plan_text += f"- **Expected Tools**: {tools_str}\n"
            plan_text += "\n"

        plan_text += "---\n"
        plan_text += "Plan recorded. Please strictly follow the above step order for execution."
        plan_text += " If adjustments are needed during execution, call the plan tool again to update."

        return plan_text


class SearchTifData(AgentTool):
    """
    Satellite image data retrieval tool. Searches for matching TIF files under /data directory
    based on time and region information.
    """
    def __init__(self) -> None:
        super().__init__()
        self.data_dir = "/var/opt/data"

    def get_name(self) -> str:
        return "search_tif_data"

    def execute(self, arguments: str) -> str:
        """Search for matching TIF file absolute paths by time and region."""
        args = json.loads(arguments)
        year = args.get("year")
        region = args.get("region")
        latest_descending = args.get("latest_descending", True)

        if not year or not region:
            return "`year` and `region` are required parameters"

        month = args.get("month")
        day = args.get("day")

        # Build date wildcard portion

        if day and month:
            date_pattern = f"{year}{month}{day}"
        elif month:
            date_pattern = f"{year}{month}??"
        else:
            date_pattern = f"{year}????"

        # Build full match pattern: {date_wildcard}-{region}-*.tif
        glob_pattern = f"{date_pattern}-{region}-*.tif"
        logger.debug(msg=f"SearchTifData glob pattern: {glob_pattern}")

        results = []
        try:
            # Search under /data (currently top-level only; recursive search can be added in the future)
            search_path = os.path.join(self.data_dir, glob_pattern)
            matched = glob_module.glob(search_path)
            results = [os.path.abspath(p) for p in matched]
        except Exception as e:
            logger.error(msg=f"SearchTifData error: {str(e)}")
            return f"Search error occurred: {str(e)}"

        if not results:
            return "The followings are the results:\n\nNo tif files found matching the criteria."

        # Sort by date label (YYYYMMDD) in the filename
        def _extract_date(path: str):
            basename = os.path.basename(path)
            match = re.match(r"(\d{8})", basename)
            return match.group(1) if match else ""

        results.sort(key=_extract_date, reverse=latest_descending)

        result_str = "The followings are the results:\n"
        for path in results:
            result_str += f"{path}\n"

        return result_str

# search_tool = SearchTifData()
# print(search_tool.execute('{"year": "2026", "region": "Shanghai"}'))
