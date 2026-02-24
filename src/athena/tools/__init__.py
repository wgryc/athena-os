from abc import ABC, abstractmethod
from typing import Any
import json


class Tool(ABC):
    """Abstract base class for LiteLLM-compatible tool calls.

    Subclasses must define the tool's name, description, parameter schema,
    and execution logic. Use `to_json()` to produce the dict that LiteLLM
    expects in the `tools` list, and `execute()` to run the tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Function name used in the tool call (e.g. 'get_stock_price')."""
        ...

    @property
    def label(self) -> str:
        """Short human-friendly label shown in the UI (e.g. 'Stock Price Lookup')."""
        return self.name.replace("_", " ").title()

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema object describing the tool's accepted parameters.

        Must follow the format:
            {
                "type": "object",
                "properties": { ... },
                "required": [...]
            }
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return a string result."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize this tool into the LiteLLM tool-call format.

        Returns a dict suitable for inclusion in the `tools` parameter of
        `litellm.completion()`:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def handle_tool_call(self, arguments_json: str) -> str:
        """Parse a JSON arguments string from the model and execute the tool.

        This is a convenience wrapper for processing tool calls returned by
        LiteLLM: `tool_call.function.arguments` is a JSON string, which this
        method decodes before passing to `execute()`.
        """
        kwargs = json.loads(arguments_json)
        return self.execute(**kwargs)


class VisualTool(Tool):
    """User-facing dashboard widget. Not LLM-callable.

    Subclasses implement `execute()` to fetch data (storing results on self),
    then `to_html()` and `to_context()` to render the stored data as an HTML
    widget or a plain-text summary respectively.
    """

    @abstractmethod
    def to_html(self) -> str:
        """Render the widget as a self-contained HTML snippet.

        Must be called after `execute()` so that internal state is populated.
        """
        ...

    @abstractmethod
    def to_context(self) -> str:
        """Return a plain-text summary of the widget's data.

        Must be called after `execute()` so that internal state is populated.
        """
        ...
