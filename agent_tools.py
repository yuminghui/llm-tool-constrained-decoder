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
        """获取工具名称"""
        pass

    @abstractmethod
    def execute(self, arguments: str) -> str:
        """运行工具，返回结果"""
        pass

    def auto_execute(self) -> bool:
        """是否允许自动执行"""
        return True  # 默认true

    def needs_sandbox(self) -> bool:
        """是否需要在沙箱中执行（适用于涉及文件写入的危险操作）"""
        return False

    def get_output_paths(self, arguments: str) -> list:
        """返回该工具的'输出文件路径'参数key列表，用于沙箱执行后回拷输出文件。
        子类可覆盖此方法以声明输出路径参数名。"""
        return []

    def get_input_paths(self, arguments: str) -> list:
        """返回该工具的'输入文件路径'参数key列表，用于沙箱执行前复制输入文件。
        子类可覆盖此方法以声明输入路径参数名。"""
        return []

class GetWeather(AgentTool):
    def get_name(self) -> str:
        return "get_weather"
        
    def execute(self, arguments: str) -> str:
        """获取天气信息"""
        return "天气晴朗，温度22摄氏度"

class GetTime(AgentTool):
    def get_name(self) -> str:
        return "get_time"
        
    def execute(self, arguments: str) -> str:
        """获取当前系统时间，以日为最小时间粒度，返回格式为当前时间为：yyyy-mm-dd"""
        now = datetime.datetime.now()
        return f"当前时间为：{now.year}-{now.month:02d}-{now.day:02d}"

class ChatResponse(AgentTool):
    def get_name(self) -> str:
        return "chat_response"
        
    def execute(self, arguments: str) -> str:
        """答复用户"""
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
        """获取当前工作区路径"""
        # 无需参数，获取工作区路径并返回字符串
        return "/var/opt"
    
class RemoteSenseIndexCalculate(AgentTool):
    def __init__(self, conda_env: str, query: str, model: str) -> None:
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
        """执行遥感指数计算"""
        args = json.loads(arguments)
        input_path = args.get("input_path")
        output_path = args.get("output_path")
        index = args.get("index")
        workers = args.get("workers")
        if not input_path or not index:
            return "input_path and index are required parameters"
        if not isinstance(index, list):
            return "index must be a list of strings"
        # 调用的第三方应用会处理剩余的默认值（None参数）

        cmd_str = "conda run -n {conda_env} python /var/opt/my_app_2/main.py {input_path} {output_path} {index} {workers}"
        index_para = ",".join(index) if len(index) > 1 else index[0]
        output_path = output_path if output_path else ""
        workers_para = f"--workers {workers}" if workers else ""
        cmd_str = cmd_str.format(conda_env=self.conda_env, input_path=input_path, output_path=output_path, index=index_para, workers=workers_para)
        logger.debug(msg=f"Will run command: {cmd_str}")

        # 执行指令
        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd="/var/opt/my_app_2")
            # 返回命令输出，包括stdout和stderr
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
                    logger.warning(msg=f".stats.json 文件不存在: {stats_path}")
                    return self.success.format(output_path=output_path)
            return self.failed.format(error=output)
        except Exception as e:
            return self.failed.format(error=f"执行命令失败: {str(e)}")

class TaskSummary(AgentTool):
    def get_name(self) -> str:
        return "task_summary"
    
    def execute(self, arguments: str) -> str:
        """总结任务"""
        args = json.loads(arguments)
        task_summary = f'# {args["summary_title"]}\n'
        if "key_findings" in args:
            task_summary += "## 关键发现"
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
## 支持的指数类型 ##
- 植被指数: ndvi, savi, evi, gndvi, arvi, msavi, lai, sr, osavi, rdvi, cvi, cigreen, grvi, gli, vari
- 水体指数: ndwi, mndwi, awei_nsh, awei_sh, ndti
- 土壤指数: gsi, ci, bi, bi2, ri, sci, ndshi, ndsvi
- 藻类指数: fai, ndci, sabi, atbi, cig
- 燃烧/色素/建筑指数: bai, nbr, sipi, vgnir_bi, building_bi
"""

class BashLs(AgentTool):
    """
    提供一个basg工具ls
    """
    def get_name(self) -> str:
        return "bash_ls"

    def execute(self, arguments: str) -> str:
        """执行ls命令"""
        # ls参数实际真可选
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
            # 返回命令输出，包括stdout和stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"
            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"

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
        """执行高分卫星影像预处理"""
        # 该工具所有参数实际都是必选，只是我替LLM为工具设置了默认参数
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
            # 返回命令输出，包括stdout和stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            # 需要记录app1的完整日志方便后续调试
            logger.debug(msg=f"app1运行完整日志: {output}")

            # 提取output最后一次info和之后的信息
            fin_index = output.rfind("批量处理完成")
            if fin_index != -1:
                output = output[fin_index:]
                output = f"{output}, 所有成功结果已保存到{output_path}中"
            else:
                # 匹配错误日志信息
                error_index = output.rfind("ERROR")
                if error_index != -1:
                    output = output[error_index:]
                else:
                    output = "批量处理失败"
            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"

class GetToolsList(AgentTool):
    def __init__(self, tools_map: dict) -> None:
        super().__init__()
        self.tools_map = tools_map  # 这玩意来自工具工厂

    def get_name(self) -> str:
        return "get_tools_list"

    def execute(self, arguments: str) -> str:
        """获取所有工具列表"""
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
        """执行DCVA变化检测"""
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
            # 返回命令输出，包括stdout和stderr
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            fin_idx = output.rfind("检测完成!")
            if fin_idx != -1:
                output = output[fin_idx:]
            else:
                output = f"工具调用失败: {output}"
                logger.error(msg=output)

            return output
        except Exception as e:
            return f"执行命令失败: {str(e)}"

class WaitUserInstruction(AgentTool):
    def __init__(self) -> None:
        super().__init__()
    
    def get_name(self) -> str:
        return "wait_user_instruction"

    def execute(self, arguments: str) -> str:
        """等待用户输入指令"""
        args = json.loads(arguments)
        tip = args.get("tip")
        if not tip:
            return "`tip` is required parameter"
        # 等待用户输入
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
        """搜索工作区中的目录"""
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)
        
        results = []
        import os
        
        for root, dirs, _ in os.walk(self.workspace):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for dirname in dirs:
                # 检查目录名是否包含关键词
                if case_sensitive:
                    if keyword not in dirname:
                        continue
                else:
                    if keyword.lower() not in dirname.lower():
                        continue
                
                # 构建完整路径
                full_path = os.path.join(root, dirname)
                results.append(full_path)
        
        # 构建返回字符串
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
        """搜索工作区中的文件"""
        args = json.loads(arguments)
        keyword = args.get("keyword")
        if not keyword:
            return "`keyword` is required parameter"
        case_sensitive = args.get("case_sensitive", False)
        
        results = []
        
        for root, dirs, files in os.walk(self.workspace):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for filename in files:
                # 检查文件名是否包含关键词
                if case_sensitive:
                    if keyword not in filename:
                        continue
                else:
                    if keyword.lower() not in filename.lower():
                        continue
                
                # 检查文件后缀是否在排除列表中
                excluded = False
                for ext in self.excluded_files:
                    if filename.endswith(ext):
                        excluded = True
                        break
                if excluded:
                    continue
                
                # 构建完整路径
                full_path = os.path.join(root, filename)
                results.append(full_path)
        
        # 构建返回字符串
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
        """接收并格式化任务规划，供后续执行参考"""
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return "规划提交失败：参数格式无效，请提供合法的JSON"

        title = args.get("title", "未命名规划")
        steps = args.get("steps", [])

        if not steps:
            return "规划提交失败：steps 不能为空，请至少提供一个步骤"

        # 向后兼容：如果 steps 是字符串列表，转换为对象列表
        if steps and isinstance(steps[0], str):
            steps = [{"step_number": i + 1, "step_name": s, "description": s} for i, s in enumerate(steps)]

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
        plan_text += "⏭️ 规划已记录。请严格按照上述步骤顺序执行，每完成一个步骤后再进入下一步。"
        plan_text += "执行过程中如需调整规划，可再次调用 plan 工具更新。"

        return plan_text


class SearchTifData(AgentTool):
    """
    卫星影像数据检索工具，根据时间和区域信息在 /data 目录下搜索匹配的 tif 文件。
    """
    def __init__(self) -> None:
        super().__init__()
        self.data_dir = "/var/opt/data"

    def get_name(self) -> str:
        return "search_tif_data"

    def execute(self, arguments: str) -> str:
        """根据时间、区域信息搜索匹配的 tif 文件绝对路径"""
        args = json.loads(arguments)
        year = args.get("year")
        region = args.get("region")
        latest_descending = args.get("latest_descending", True)

        if not year or not region:
            return "`year` and `region` are required parameters"

        month = args.get("month")
        day = args.get("day")

        # 构建日期通配部分
        
        if day and month:
            date_pattern = f"{year}{month}{day}"
        elif month:
            date_pattern = f"{year}{month}??"
        else:
            date_pattern = f"{year}????"

        # 构建完整匹配模式: {日期通配}-{region}-*.tif
        glob_pattern = f"{date_pattern}-{region}-*.tif"
        logger.debug(msg=f"SearchTifData glob pattern: {glob_pattern}")

        results = []
        try:
            # 在 /data 下递归搜索（当前仅搜索顶层，未来可扩展递归）
            search_path = os.path.join(self.data_dir, glob_pattern)
            matched = glob_module.glob(search_path)
            results = [os.path.abspath(p) for p in matched]
        except Exception as e:
            logger.error(msg=f"SearchTifData error: {str(e)}")
            return f"搜索过程发生错误: {str(e)}"

        if not results:
            return "The followings are the results:\n\nNo tif files found matching the criteria."
        
        # 按文件名中的时间标签（YYYYMMDD）排序
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
# print(search_tool.execute('{"year": "2026", "region": "上海"}'))
