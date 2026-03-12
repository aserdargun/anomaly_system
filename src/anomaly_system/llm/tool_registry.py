"""Tool registry — maps Python functions to Ollama tool definitions."""

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry that maps Python functions to Ollama tool call schemas."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
    ) -> Callable:
        """Decorator to register a function as an LLM-callable tool.

        Args:
            name: Tool name (must match what the LLM will call).
            description: Human-readable description for the LLM.
            parameters: JSON Schema for the function parameters.
        """
        def decorator(func: Callable) -> Callable:
            self._tools[name] = func
            self._schemas[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
            logger.debug("Registered tool: %s", name)
            return func
        return decorator

    def get_definitions(self) -> list[dict]:
        """Return tool schemas for Ollama API."""
        return list(self._schemas.values())

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict) -> dict[str, Any]:
        """Execute a registered tool and return result as dict.

        Args:
            name: Tool name to execute.
            arguments: Keyword arguments for the tool function.

        Returns:
            Dict with 'success' and 'result' keys, or 'error' key on failure.
        """
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}

        try:
            func = self._tools[name]
            result = func(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return {"success": True, "result": result}
        except Exception as e:
            logger.exception("Tool '%s' failed", name)
            return {"error": f"{type(e).__name__}: {e}"}
