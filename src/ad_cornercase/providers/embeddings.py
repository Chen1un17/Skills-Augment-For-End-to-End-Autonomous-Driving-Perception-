"""Embedding adapters."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from openai import AsyncOpenAI

from ad_cornercase.providers.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]


class HashEmbeddingProvider(EmbeddingProvider):
    """Cheap deterministic fallback used only for tests."""

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [0.0] * self._dimensions
        for index, byte in enumerate(digest):
            values[index % self._dimensions] += (byte / 255.0) - 0.5
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]
