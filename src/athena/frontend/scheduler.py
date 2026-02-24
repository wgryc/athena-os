"""Background scheduler that checks tasks every 60 seconds and triggers LLM calls."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from ..tasks import ScheduledTask, save_tasks_to_excel

if TYPE_CHECKING:
    from .gateways import Gateway


class TaskScheduler:
    """Simple background scheduler that runs due tasks via the shared LLM chat.

    Follows the same daemon-thread pattern as the gateway system.
    """

    def __init__(
        self,
        get_tasks: Callable[[], list[ScheduledTask]],
        process_message: Callable[[str, str], str],
        tasks_file: str,
        check_interval: int = 60,
        gateways: list[Gateway] | None = None,
    ):
        """Initialize the task scheduler.

        Args:
            get_tasks: Callable returning the current list of scheduled tasks.
            process_message: Callback that sends a prompt to the LLM and
                returns the response (signature: ``(text, source) -> str``).
            tasks_file: Path to the Excel file for persisting task state.
            check_interval: Seconds between task-due checks.
            gateways: Optional list of active gateways to forward task
                output to.
        """
        self._get_tasks = get_tasks
        self._process_message = process_message
        self._tasks_file = tasks_file
        self._check_interval = check_interval
        self._gateways = gateways or []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the scheduler background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Main loop: check tasks, sleep, repeat."""
        while not self._stop_event.is_set():
            self._check_tasks()
            self._stop_event.wait(self._check_interval)

    def _check_tasks(self) -> None:
        """Iterate over tasks and execute any that are due."""
        now = datetime.now(timezone.utc)
        tasks = self._get_tasks()
        due_names = [t.name for t in tasks if t.is_due(now)]
        if due_names:
            print(f"[Scheduler] {now.strftime('%H:%M:%S UTC')} — {len(due_names)} task(s) due: {', '.join(due_names)}", flush=True)
        else:
            print(f"[Scheduler] {now.strftime('%H:%M:%S UTC')} — checked {len(tasks)} task(s), none due.", flush=True)
        any_ran = False

        for task in tasks:
            if self._stop_event.is_set():
                break
            if task.is_due(now):
                self._execute_task(task, now)
                any_ran = True

        if any_ran:
            try:
                save_tasks_to_excel(tasks, self._tasks_file)
            except Exception as e:
                print(f"[Scheduler] Failed to save tasks: {e}", flush=True)

    def _execute_task(self, task: ScheduledTask, now: datetime) -> None:
        """Send the task description to the LLM via the shared chat.

        Args:
            task: The scheduled task to execute.
            now: Current UTC datetime (used to stamp ``last_run``).
        """
        print(f"[Scheduler] Running task: {task.name}", flush=True)

        prompt = (
            f"[Scheduled Task: {task.name}]\n"
            f"Schedule: {task.schedule}\n\n"
            f"{task.description}"
        )

        try:
            response = self._process_message(prompt, "scheduler")
            task.last_run = now
            print(f"[Scheduler] Task '{task.name}' completed.", flush=True)

            # Forward response to all active gateways (e.g. Telegram)
            for gw in self._gateways:
                try:
                    gw.send_message(f"[{task.name}]\n{response}")
                except Exception as e:
                    print(f"[Scheduler] Failed to forward to {gw.name}: {e}", flush=True)

        except Exception as e:
            print(f"[Scheduler] Task '{task.name}' failed: {e}", flush=True)
