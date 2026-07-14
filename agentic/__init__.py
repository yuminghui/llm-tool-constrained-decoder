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

from .agent import (
    Agent,
    AgentConfig,
    AgentStep,
    AgentResult,
    TOOL_KEYWORDS,
)

# ``llm_backend`` and ``experiment`` pull in ``torch``.  Importing them eagerly
# here would force every consumer of the *torch-free* parts of this package
# (e.g. ``from agentic.tools import ToolRegistry`` used by the offline dataset
# tooling and tests) to have a working CUDA/torch stack.  Expose them lazily via
# PEP 562 so ``from agentic import LLMBackend`` / ``ExperimentRunner`` still work
# exactly as before, but torch is only imported on first access.
_LAZY_EXPORTS = {
    "LLMBackend": ("llm_backend", "LLMBackend"),
    "TEMPLATE_REGISTRY": ("llm_backend", "TEMPLATE_REGISTRY"),
    "DEFAULT_TEMPLATE": ("llm_backend", "DEFAULT_TEMPLATE"),
    "Task": ("experiment", "Task"),
    "ExperimentConfig": ("experiment", "ExperimentConfig"),
    "RunMetrics": ("experiment", "RunMetrics"),
    "ExperimentRunner": ("experiment", "ExperimentRunner"),
    "DEFAULT_TASKS": ("experiment", "DEFAULT_TASKS"),
    "run_experiment_cli": ("experiment", "main"),
}


def __getattr__(name):  # PEP 562 — module-level lazy attribute access
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{target[0]}", __name__)
    value = getattr(module, target[1])
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = [
    # Tools
    "AgentTool",
    "ToolEntry",
    "ToolRegistry",
    "TOOL_CLASS_MAP",
    "load_tools_from_json",
    # LLM Backend
    "LLMBackend",
    "TEMPLATE_REGISTRY",
    "DEFAULT_TEMPLATE",
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
