"""Async HTTP client for Ollama REST API with Metal acceleration on Mac Mini M4."""

import base64
import logging
from pathlib import Path

import httpx

from anomaly_system.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_THINKING,
    OLLAMA_VISION_TIMEOUT,
    VISION_MAX_IMAGE_WIDTH,
)

logger = logging.getLogger(__name__)


class OllamaNotRunning(Exception):
    """Raised when Ollama server is not reachable."""


class ModelNotFound(Exception):
    """Raised when the requested model is not available in Ollama."""


class ToolCallError(Exception):
    """Raised when a tool call response is malformed."""


class OllamaClient:
    """Async client for Ollama REST API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        thinking: bool = OLLAMA_THINKING,
    ) -> None:
        """Initialize Ollama client."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.thinking = thinking

    def _inject_thinking_tag(self, messages: list[dict]) -> list[dict]:
        """Inject /think or /no_think tag into the first user message.

        Qwen 3.5 uses these tags to enable/disable its internal reasoning.
        Disabling thinking significantly speeds up responses.
        """
        tag = "/think" if self.thinking else "/no_think"
        messages = [m.copy() for m in messages]
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                if "/think" not in m["content"] and "/no_think" not in m["content"]:
                    m["content"] = f"{tag}\n\n{m['content']}"
                break
        return messages

    async def health_check(self) -> dict:
        """Check Ollama status and verify model availability.

        Returns dict with model_name, model_size, and metal status.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaNotRunning(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Try: brew services restart ollama"
            ) from e

        data = resp.json()
        models = data.get("models", [])
        model_names = [m["name"] for m in models]

        # Find our model (match with or without :latest tag)
        target = self.model.replace(":latest", "")
        found = None
        for m in models:
            if m["name"].startswith(target):
                found = m
                break

        if found is None:
            raise ModelNotFound(
                f"Model '{self.model}' not found. Available: {model_names}. "
                f"Run: ollama pull {self.model}"
            )

        return {
            "model_name": found["name"],
            "model_size": found.get("size", "unknown"),
            "metal_accelerated": True,  # Ollama always uses Metal on Apple Silicon
            "available_models": model_names,
        }

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send chat completion request to Ollama.

        Args:
            messages: Conversation history.
            tools: Optional list of tool definitions.

        Returns:
            Full response dict from Ollama API.
        """
        # Inject /no_think or /think into the first user message for Qwen 3.5
        messages = self._inject_thinking_tag(messages)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        logger.debug("Ollama chat request: model=%s, messages=%d, tools=%d, thinking=%s",
                      self.model, len(messages), len(tools or []), self.thinking)

        last_error = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    logger.debug("Ollama chat response: %s", str(result)[:500])
                    return result
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = e
                logger.warning("Ollama chat attempt %d failed: %s", attempt + 1, e)

        raise OllamaNotRunning(
            f"Failed after 3 attempts: {last_error}. "
            "Try: brew services restart ollama"
        )

    @staticmethod
    def _resize_image_for_vision(image_path: str, max_width: int = VISION_MAX_IMAGE_WIDTH) -> bytes:
        """Resize image to reduce vision processing time.

        Args:
            image_path: Path to image file.
            max_width: Maximum width in pixels.

        Returns:
            PNG bytes of the resized image.
        """
        from io import BytesIO
        from PIL import Image

        img = Image.open(image_path)
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def chat_with_vision(self, prompt: str, image_path: str) -> str:
        """Send a vision request with an image to the model.

        Args:
            prompt: Text prompt describing what to analyze.
            image_path: Path to image file (PNG, JPG).

        Returns:
            LLM's text response.
        """
        image_data = self._resize_image_for_vision(image_path)
        b64_image = base64.b64encode(image_data).decode("utf-8")

        tag = "/think" if self.thinking else "/no_think"
        messages = [
            {
                "role": "user",
                "content": f"{tag}\n\n{prompt}",
                "images": [b64_image],
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        vision_timeout = OLLAMA_VISION_TIMEOUT
        logger.debug("Ollama vision request: image=%s (resized to %dpx, timeout=%ds)",
                      image_path, VISION_MAX_IMAGE_WIDTH, vision_timeout)

        try:
            async with httpx.AsyncClient(timeout=vision_timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                return result["message"]["content"]
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaNotRunning(
                f"Vision call failed (connection): {e}. "
                "Try: brew services restart ollama"
            ) from e
        except httpx.ReadTimeout as e:
            raise OllamaNotRunning(
                f"Vision call timed out after {vision_timeout}s. "
                "Try --no-think for faster processing."
            ) from e
