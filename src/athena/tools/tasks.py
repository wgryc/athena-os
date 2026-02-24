"""Scheduled tasks tool (LLM-callable) and dashboard widget."""

from __future__ import annotations

from typing import Any, Callable

from athena.tasks import ScheduledTask, parse_schedule
from athena.tools import Tool, VisualTool


def _esc(s: str) -> str:
    """Escape ``&``, ``<``, and ``>`` for safe HTML embedding.

    Args:
        s: Raw string to escape.

    Returns:
        The escaped string with HTML-special characters replaced by their
        entity equivalents.
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ManageTasks(Tool):
    """LLM-callable tool for listing, adding, updating, and deleting scheduled tasks."""

    def __init__(
        self,
        get_tasks: Callable[[], list[ScheduledTask]],
        add_task: Callable[[str, str, str], ScheduledTask],
        update_task: Callable[[int, dict], bool],
        delete_task: Callable[[int], bool],
    ):
        """Initialize the task manager tool.

        Args:
            get_tasks: Callable returning the current list of scheduled tasks.
            add_task: Callable ``(name, schedule, description)`` that creates
                and returns a new ``ScheduledTask``.
            update_task: Callable ``(index, updates_dict)`` that patches a task;
                returns ``True`` on success.
            delete_task: Callable ``(index)`` that removes a task; returns
                ``True`` on success.
        """
        self._get_tasks = get_tasks
        self._add_task = add_task
        self._update_task = update_task
        self._delete_task = delete_task

    @property
    def name(self) -> str:
        return "manage_tasks"

    @property
    def label(self) -> str:
        return "Task Manager"

    @property
    def description(self) -> str:
        return (
            "Manage scheduled tasks. Actions: "
            "'list' to see all tasks, "
            "'add' to create a new task (provide task_name, schedule, task_description), "
            "'update' to modify an existing task (provide index and fields to change), "
            "'delete' to remove a task (provide index)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "update", "delete"],
                    "description": "The action to perform.",
                },
                "task_name": {
                    "type": "string",
                    "description": "Name of the task (for add/update).",
                },
                "schedule": {
                    "type": "string",
                    "description": (
                        "Schedule in plain English, e.g. 'Every hour', "
                        "'1230pm ET daily' (for add/update)."
                    ),
                },
                "task_description": {
                    "type": "string",
                    "description": "What the task should do when triggered (for add/update).",
                },
                "index": {
                    "type": "integer",
                    "description": "Task index (0-based) for update/delete.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if action == "list":
            return self._do_list()
        elif action == "add":
            return self._do_add(kwargs)
        elif action == "update":
            return self._do_update(kwargs)
        elif action == "delete":
            return self._do_delete(kwargs)

        return f"Unknown action: {action}"

    def _do_list(self) -> str:
        tasks = self._get_tasks()
        if not tasks:
            return "No scheduled tasks."
        lines = [f"Scheduled Tasks ({len(tasks)}):\n"]
        for i, t in enumerate(tasks):
            last = t.last_run.strftime("%Y-%m-%d %H:%M UTC") if t.last_run else "never"
            lines.append(
                f"  [{i}] {t.name}\n"
                f"      Schedule: {t.schedule}\n"
                f"      Description: {t.description}\n"
                f"      Last run: {last}\n"
                f"      Added by: {t.added_by}"
            )
        return "\n".join(lines)

    def _do_add(self, kwargs: dict) -> str:
        name = kwargs.get("task_name", "").strip()
        schedule = kwargs.get("schedule", "").strip()
        desc = kwargs.get("task_description", "").strip()
        if not name or not schedule or not desc:
            return "Error: task_name, schedule, and task_description are all required for 'add'."
        try:
            parse_schedule(schedule)
        except ValueError as e:
            return f"Error: {e}"
        task = self._add_task(name, schedule, desc)
        return f"Task '{task.name}' added successfully with schedule: {task.schedule}"

    def _do_update(self, kwargs: dict) -> str:
        index = kwargs.get("index")
        if index is None:
            return "Error: 'index' is required for update."
        updates: dict[str, str] = {}
        if "task_name" in kwargs:
            updates["name"] = kwargs["task_name"]
        if "schedule" in kwargs:
            try:
                parse_schedule(kwargs["schedule"])
            except ValueError as e:
                return f"Error: {e}"
            updates["schedule"] = kwargs["schedule"]
        if "task_description" in kwargs:
            updates["description"] = kwargs["task_description"]
        if not updates:
            return "Error: provide at least one field to update (task_name, schedule, or task_description)."
        ok = self._update_task(index, updates)
        return f"Task [{index}] updated." if ok else f"Error: invalid task index {index}."

    def _do_delete(self, kwargs: dict) -> str:
        index = kwargs.get("index")
        if index is None:
            return "Error: 'index' is required for delete."
        ok = self._delete_task(index)
        return f"Task [{index}] deleted." if ok else f"Error: invalid task index {index}."


class ScheduledTasksWidget(VisualTool):
    """Dashboard widget showing scheduled tasks and their status."""

    def __init__(self, get_tasks: Callable[[], list[ScheduledTask]]):
        """Initialize the scheduled tasks widget.

        Args:
            get_tasks: Callable returning the current list of scheduled tasks.
        """
        self._get_tasks = get_tasks
        self._tasks: list[ScheduledTask] = []

    @property
    def name(self) -> str:
        return "scheduled_tasks_widget"

    @property
    def label(self) -> str:
        return "Scheduled Tasks"

    @property
    def description(self) -> str:
        return "Display scheduled tasks and their last-run status."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> str:
        self._tasks = list(self._get_tasks())
        return self.to_context()

    def to_context(self) -> str:
        if not self._tasks:
            return "(no scheduled tasks)"
        lines = [f"Scheduled Tasks ({len(self._tasks)}):"]
        for t in self._tasks:
            last = t.last_run.strftime("%Y-%m-%d %H:%M") if t.last_run else "never"
            lines.append(f"  - {t.name} ({t.schedule}) -- last run: {last}")
        return "\n".join(lines)

    def to_html(self) -> str:
        if not self._tasks:
            return '<div class="widget-card widget-error">No scheduled tasks</div>'

        rows = ""
        for t in self._tasks:
            last = t.last_run.strftime("%Y-%m-%d %H:%M") if t.last_run else "never"
            badge_class = "task-badge-user" if t.added_by == "user" else "task-badge-athena"
            rows += (
                '<div class="task-item">'
                f'<div class="task-name">{_esc(t.name)}</div>'
                f'<div class="task-meta">'
                f'<span class="task-schedule">{_esc(t.schedule)}</span>'
                f'<span class="task-last-run">Last: {last}</span>'
                f'<span class="task-badge {badge_class}">{_esc(t.added_by)}</span>'
                f'</div>'
                f'<div class="task-desc">{_esc(t.description[:120])}</div>'
                '</div>'
            )

        return (
            '<div class="widget-card">'
            '<h3 class="widget-title">Scheduled Tasks</h3>'
            f'<div class="task-list">{rows}</div>'
            '</div>'
        )
