import os
from abc import ABC, abstractmethod

import ollama as _ollama


class LLMProvider(ABC):
    """
    Minimal provider interface: generate(prompt) -> str.

    Deliberately thin — this is the only contract the rest of the pipeline
    depends on, so a future GeminiProvider or ClaudeProvider slots in here
    without touching any other file.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    """Local Ollama backend. Requires `ollama serve` and the model to be pulled."""

    def __init__(self, model: str | None = None):
        # OLLAMA_MODEL from .env overrides the default; llama3 is the fallback.
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, prompt: str) -> str:
        response = _ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
