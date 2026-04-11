"""Provider interfaces and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(slots=True)
class StructuredProviderResult(Generic[SchemaT]):
    parsed: SchemaT
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class StructuredVisionProvider(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        response_model: type[SchemaT],
        image_paths: Sequence[Path] = (),
        metadata: dict[str, str] | None = None,
        max_completion_tokens: int = 2048,
    ) -> StructuredProviderResult[SchemaT]:
        """Run a structured vision-language completion."""


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for all input texts."""
