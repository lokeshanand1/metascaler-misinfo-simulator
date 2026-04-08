"""Tasks module — defines Easy, Medium, and Hard task scenarios."""

from tasks.task_definitions import TASK_REGISTRY, TaskConfig, get_task

__all__ = ["TASK_REGISTRY", "TaskConfig", "get_task"]
