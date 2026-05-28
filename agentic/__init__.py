"""
Agentic experiment framework for testing tool-constrained decoding.

Provides a Plan-then-Act agent that can operate in constrained or free
generation mode, plus an experiment runner for comparing configurations
across multiple tasks and models.

Tools are loaded from ``tools.json`` via ``load_tools_from_json()``.
"""

from .tools import (
    AgentTool,
    ToolEntry,
    ToolRegistry,
    TOOL_CLASS_MAP,
    load_tools_from_json,
)

from .llm_backend import (
    LLMBackend,
    GEMMA_TOOL_CALL_TEMPLATE,
)

from .agent import (
    Agent,
    AgentConfig,
    AgentStep,
    AgentResult,
    TOOL_KEYWORDS,
)

from .experiment import (
    Task,
    ExperimentConfig,
    RunMetrics,
    ExperimentRunner,
    DEFAULT_TASKS,
    main as run_experiment_cli,
)

__all__ = [
    # Tools
    "AgentTool",
    "ToolEntry",
    "ToolRegistry",
    "TOOL_CLASS_MAP",
    "load_tools_from_json",
    # LLM Backend
    "LLMBackend",
    "GEMMA_TOOL_CALL_TEMPLATE",
    # Agent
    "Agent",
    "AgentConfig",
    "AgentStep",
    "AgentResult",
    "TOOL_KEYWORDS",
    # Experiment
    "Task",
    "ExperimentConfig",
    "RunMetrics",
    "ExperimentRunner",
    "DEFAULT_TASKS",
    "run_experiment_cli",
]
