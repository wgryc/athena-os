"""Tools for reading and editing the bot's personality, memory, and todo files.

These tools allow the LLM to introspect and modify its own markdown-based
configuration files at runtime. Each tool is parameterized with the file
path at construction time (resolved from the bot JSON config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import Tool


class ReadAgentFile(Tool):
    """Read the contents of a bot markdown file (personality, memory, or todo).

    Args:
        file_path: Absolute path to the markdown file.
        file_label: Human-readable label (e.g. "personality", "memory", "todo").
        tool_name: The LLM-facing function name.
    """

    def __init__(self, file_path: str | Path, file_label: str, tool_name: str) -> None:
        self._file_path = Path(file_path)
        self._file_label = file_label
        self._tool_name = tool_name

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return f"Read your current {self._file_label} file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        """Read and return the file contents.

        Returns:
            The markdown contents of the file, or a message if the file
            does not exist or is empty.
        """
        if not self._file_path.exists():
            return f"The {self._file_label} file does not exist yet. Use the update tool to create it."
        content = self._file_path.read_text().strip()
        if not content:
            return f"The {self._file_label} file is empty. Use the update tool to add content."
        return content


class UpdateAgentFile(Tool):
    """Overwrite the contents of a bot markdown file.

    Args:
        file_path: Absolute path to the markdown file.
        file_label: Human-readable label (e.g. "personality", "memory", "todo").
        tool_name: The LLM-facing function name.
    """

    def __init__(self, file_path: str | Path, file_label: str, tool_name: str) -> None:
        self._file_path = Path(file_path)
        self._file_label = file_label
        self._tool_name = tool_name

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return (
            f"Replace the entire contents of your {self._file_label} file with new markdown. "
            f"Use this to update your {self._file_label}. Pass the full new content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": f"The new markdown content for the {self._file_label} file. This replaces the entire file.",
                },
            },
            "required": ["content"],
        }

    def execute(self, **kwargs: Any) -> str:
        """Write new content to the file.

        Args:
            content: The full markdown string to write.

        Returns:
            Confirmation message.
        """
        content = kwargs.get("content", "")
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(content.strip() + "\n")
        return f"Successfully updated {self._file_label} file."
