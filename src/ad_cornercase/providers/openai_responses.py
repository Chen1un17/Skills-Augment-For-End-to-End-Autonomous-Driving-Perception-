"""OpenAI-compatible chat completions adapter."""

from __future__ import annotations

import base64
import json
import mimetypes
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ad_cornercase.providers.base import SchemaT, StructuredProviderResult, StructuredVisionProvider


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # Remove control characters that break JSON parsing
    import re
    stripped = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', stripped)
    return stripped


VISION_MODEL_MARKERS = (
    "-vl-",
    "/qvq",
    "glm-4.1v",
    "glm-4.5v",
    "glm-4.6v",
    "paddleocr",
    "ocr-vl",
    "omni",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen/qwen3.5-",
    "moonshotai/kimi-k2.5",
)


def _supports_vision(model: str) -> bool:
    normalized = model.lower()
    return any(marker in normalized for marker in VISION_MODEL_MARKERS)


def _image_detail(model: str) -> str:
    normalized = model.lower()
    if normalized.startswith("qwen/qwen3.5-"):
        return "low"
    if "moonshotai/kimi-k2.5" in normalized:
        return "auto"
    return "auto"


def _request_overrides(model: str) -> dict[str, object]:
    normalized = model.lower()
    if (
        normalized.startswith("qwen/qwen3")
        and "-vl" not in normalized
        and "omni" not in normalized
        and "thinking" not in normalized
    ):
        return {"enable_thinking": False}
    return {}


def _extract_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(parts)
    return ""


def _repair_prompt(schema: dict[str, object], draft: str) -> str:
    return json.dumps(
        {
            "task": "repair_structured_json",
            "requirements": [
                "Return valid JSON only.",
                "Conform exactly to the provided JSON schema.",
                "Preserve the draft semantics.",
                "If required fields are missing, infer them conservatively from the draft instead of leaving them out.",
            ],
            "json_schema": schema,
            "draft": draft,
        },
        ensure_ascii=False,
    )


class OpenAIResponsesVisionProvider(StructuredVisionProvider):
    """Structured adapter over OpenAI-compatible chat completions APIs."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float, max_retries: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._max_retries = max_retries

    def _maybe_compact_image(self, path: Path, model: str) -> tuple[bytes, str]:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        should_compact = (
            platform.system() == "Darwin"
            and path.stat().st_size > 1_000_000
            and _supports_vision(model)
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if should_compact:
            with tempfile.NamedTemporaryFile(prefix="ad_vlm_", suffix=".jpg", delete=False) as handle:
                temp_path = Path(handle.name)
            try:
                subprocess.run(
                    [
                        "/usr/bin/sips",
                        "-s",
                        "format",
                        "jpeg",
                        "-s",
                        "formatOptions",
                        "70",
                        "-Z",
                        "1280",
                        str(path),
                        "--out",
                        str(temp_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return temp_path.read_bytes(), "image/jpeg"
            finally:
                temp_path.unlink(missing_ok=True)

        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return path.read_bytes(), mime

    def _encode_image(self, path: Path, model: str) -> str:
        image_bytes, mime = self._maybe_compact_image(path, model)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    async def _repair_structured_output(
        self,
        *,
        model: str,
        response_model: type[SchemaT],
        raw_text: str,
    ) -> SchemaT:
        completion = await self._client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": "Repair the draft into valid JSON that exactly matches the requested schema. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _repair_prompt(response_model.model_json_schema(), raw_text)}],
                },
            ],
            response_format={"type": "json_object"},
            extra_body=_request_overrides(model),
        )
        repaired_text = _extract_content_text(completion.choices[0].message.content)
        return response_model.model_validate_json(_strip_json_fences(repaired_text))

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
        attempts = max(1, self._max_retries)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
            reraise=True,
        ):
            with attempt:
                start = time.perf_counter()
                user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
                if _supports_vision(model):
                    detail = _image_detail(model)
                    for path in image_paths:
                        if path.exists():
                            user_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": self._encode_image(path, model),
                                        "detail": detail,
                                    },
                                }
                            )
                elif image_paths:
                    user_content.append(
                        {
                            "type": "text",
                            "text": "Selected model is treated as text-only. Use the textual scene hints and metadata instead of the image bytes.",
                        }
                    )

                completion = await self._client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_completion_tokens=max_completion_tokens,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    extra_body=_request_overrides(model),
                )
                raw_text = _extract_content_text(completion.choices[0].message.content)
                try:
                    parsed = response_model.model_validate_json(_strip_json_fences(raw_text))
                except ValidationError:
                    parsed = await self._repair_structured_output(
                        model=model,
                        response_model=response_model,
                        raw_text=raw_text,
                    )
                usage = completion.usage
                return StructuredProviderResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
        raise RuntimeError("Structured provider exhausted retries without returning a result.")


class FakeStructuredVisionProvider(StructuredVisionProvider):
    """Deterministic provider used by tests and local dry runs."""

    def __init__(self, handlers: dict[str, callable]) -> None:
        self._handlers = handlers

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
        del model, instructions, image_paths, max_completion_tokens
        key = response_model.__name__
        if key not in self._handlers:
            raise KeyError(f"No fake handler registered for {key}")
        payload = self._handlers[key](prompt=prompt, metadata=metadata or {})
        parsed = response_model.model_validate(payload)
        raw_text = json.dumps(parsed.model_dump(mode="json"))
        return StructuredProviderResult(
            parsed=parsed,
            raw_text=raw_text,
            input_tokens=128,
            output_tokens=64,
            latency_ms=10.0,
        )
