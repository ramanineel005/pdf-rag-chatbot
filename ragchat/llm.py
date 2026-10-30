"""Local LLM through Ollama."""
from __future__ import annotations

from typing import Iterator, Protocol

import httpx
import ollama

from .models import LLMUnavailable


class LLM(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class OllamaLLM:
    def __init__(self, host: str, model: str, temperature: float = 0.1, num_ctx: int = 4096, timeout: float = 600):
        self.host, self.model = host, model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self._client = ollama.Client(host=host, timeout=timeout)

    def status(self) -> tuple[str, str]:
        """('ok'|'no_server'|'no_model', human-readable message)."""
        try:
            names = {m.model for m in self._client.list().models}
        except Exception:
            return "no_server", (
                "Ollama is not running. Install it from https://ollama.com/download, start it, "
                "then reload this page."
            )
        if self.model in names or f"{self.model}:latest" in names:
            return "ok", f"Model “{self.model}” is ready."
        return "no_model", f"Model “{self.model}” is not installed. Run:  ollama pull {self.model}"

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            for part in self._client.chat(model=self.model, messages=messages, stream=True, options=self.options):
                yield part.message.content or ""
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailable(f"Model “{self.model}” is not installed. Run:  ollama pull {self.model}") from exc
            raise LLMUnavailable("The local model returned an error. Try again, or use a smaller model.") from exc
        except (ConnectionError, OSError, httpx.HTTPError) as exc:
            raise LLMUnavailable("Could not reach Ollama. Make sure it is running (https://ollama.com/download).") from exc
