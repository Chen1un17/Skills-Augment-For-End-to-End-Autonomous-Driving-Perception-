"""Skill matching logic."""

from __future__ import annotations

import math
import re

from ad_cornercase.providers.base import EmbeddingProvider
from ad_cornercase.schemas.skill import SkillManifest, SkillMatch, SkillMatchRequest, SkillMatchResult
from ad_cornercase.skill_store.repository import SkillRepository

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "given",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "main",
    "most",
    "of",
    "on",
    "or",
    "provide",
    "safest",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "with",
}
LOW_SIGNAL_LABEL_TOKENS = {
    "classification",
    "critical",
    "detection",
    "hazard",
    "object",
    "obstacle",
    "scene",
    "unknown",
}
WEATHER_TERMS = {
    "clear",
    "cloudy",
    "day",
    "daylight",
    "dawn",
    "dusk",
    "fog",
    "foggy",
    "night",
    "nighttime",
    "overcast",
    "rain",
    "rainy",
    "snow",
    "snowy",
    "sunny",
}
LABEL_HINT_PREFIX = "label_must_match:"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(l * r for l, r in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return numerator / (left_norm * right_norm)


def build_prompt_patch(skill_id: str, focus_region: str, dynamic_question_tree: list[str], output_constraints: list[str]) -> str:
    questions = "\n".join(f"- {question}" for question in dynamic_question_tree) or "- none"
    constraints = "\n".join(f"- {item}" for item in output_constraints) or "- none"
    return (
        f"Skill `{skill_id}`\n"
        f"Focus region: {focus_region}\n"
        f"Question tree:\n{questions}\n"
        f"Output constraints:\n{constraints}"
    )


def _tokenize(text: str, *, drop_low_signal_labels: bool = False) -> tuple[str, ...]:
    blocked = STOPWORDS | (LOW_SIGNAL_LABEL_TOKENS if drop_low_signal_labels else set())
    return tuple(
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in blocked
    )


def _token_set(text: str, *, drop_low_signal_labels: bool = False) -> set[str]:
    return set(_tokenize(text, drop_low_signal_labels=drop_low_signal_labels))


def _ratio_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _manifest_label_phrases(manifest: SkillManifest) -> list[tuple[str, ...]]:
    phrases = [_tokenize(manifest.name, drop_low_signal_labels=True)]
    for item in manifest.output_constraints:
        normalized = item.strip()
        lower = normalized.lower()
        if lower.startswith("label_") and ":" in normalized:
            phrases.append(_tokenize(normalized.split(":", 1)[1], drop_low_signal_labels=True))
    return [phrase for phrase in phrases if phrase]


def _manifest_terms(manifest: SkillManifest) -> set[str]:
    parts = [
        manifest.name,
        manifest.trigger_embedding_text,
        manifest.focus_region,
        *manifest.trigger_tags,
        *manifest.dynamic_question_tree,
        *manifest.output_constraints,
    ]
    return _token_set(" ".join(parts))


def _manifest_weather_terms(manifest: SkillManifest) -> set[str]:
    weather = set()
    for item in manifest.trigger_tags:
        weather |= (_token_set(item) & WEATHER_TERMS)
    weather |= (_manifest_terms(manifest) & WEATHER_TERMS)
    return weather


def _request_terms(request: SkillMatchRequest) -> set[str]:
    parts = [
        request.trigger_text,
        request.sensor_context,
        *request.weather_tags,
        *request.top_k_labels,
    ]
    return _token_set(" ".join(parts))


def _request_label_phrases(request: SkillMatchRequest) -> list[tuple[str, ...]]:
    return [phrase for phrase in (_tokenize(label, drop_low_signal_labels=True) for label in request.top_k_labels) if phrase]


def _request_weather_terms(request: SkillMatchRequest) -> set[str]:
    return _token_set(" ".join(request.weather_tags)) & WEATHER_TERMS


def _hybrid_match_score(
    *,
    request: SkillMatchRequest,
    manifest: SkillManifest,
    embedding_score: float,
) -> float:
    manifest_terms = _manifest_terms(manifest)
    request_terms = _request_terms(request)
    manifest_label_phrases = _manifest_label_phrases(manifest)
    request_label_phrases = _request_label_phrases(request)
    manifest_weather_terms = _manifest_weather_terms(manifest)
    request_weather_terms = _request_weather_terms(request)

    label_overlap = max(
        (_ratio_overlap(set(label_phrase), manifest_terms) for label_phrase in request_label_phrases),
        default=0.0,
    )
    keyword_overlap = _jaccard_overlap(request_terms, manifest_terms)
    exact_label_match = any(label_phrase == manifest_phrase for label_phrase in request_label_phrases for manifest_phrase in manifest_label_phrases)

    weather_overlap = 0.0
    weather_conflict_penalty = 0.0
    if request_weather_terms and manifest_weather_terms:
        weather_overlap = _jaccard_overlap(request_weather_terms, manifest_weather_terms)
        if not (request_weather_terms & manifest_weather_terms):
            weather_conflict_penalty = 0.50

    return (
        embedding_score
        + (0.12 if exact_label_match else 0.0)
        + (0.14 * label_overlap)
        + (0.08 * keyword_overlap)
        + (0.10 * weather_overlap)
        - weather_conflict_penalty
    )


def _skill_family_key(manifest: SkillManifest) -> str:
    label_phrases = _manifest_label_phrases(manifest)
    if label_phrases:
        family_terms = max(label_phrases, key=len)
    else:
        family_terms = _tokenize(manifest.name, drop_low_signal_labels=True)
    if family_terms:
        return "::".join(family_terms)
    return manifest.skill_id


class SkillMatcher:
    def __init__(
        self,
        *,
        repository: SkillRepository,
        embedding_provider: EmbeddingProvider,
        threshold: float,
        max_matches: int,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._threshold = threshold
        self._max_matches = max_matches

    async def match(self, request: SkillMatchRequest) -> SkillMatchResult:
        entries = self._repository.list_index_entries()
        if not entries:
            return SkillMatchResult()
        query_embedding = (await self._embedding_provider.embed([request.trigger_text]))[0]
        bundles = {manifest.skill_id: manifest for manifest in self._repository.list_manifests()}
        ranked: list[SkillMatch] = []
        for entry in entries:
            manifest = bundles.get(entry.skill_id)
            if manifest is None:
                continue
            embedding_score = cosine_similarity(query_embedding, entry.embedding)
            score = _hybrid_match_score(
                request=request,
                manifest=manifest,
                embedding_score=embedding_score,
            )
            if score < self._threshold:
                continue
            ranked.append(
                SkillMatch(
                    skill_id=entry.skill_id,
                    score=score,
                    prompt_patch=build_prompt_patch(
                        skill_id=entry.skill_id,
                        focus_region=manifest.focus_region,
                        dynamic_question_tree=manifest.dynamic_question_tree,
                        output_constraints=manifest.output_constraints,
                    ),
                    manifest=manifest,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        deduped: list[SkillMatch] = []
        seen_families: set[str] = set()
        for item in ranked:
            family_key = _skill_family_key(item.manifest)
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            deduped.append(item)
            if len(deduped) >= self._max_matches:
                break
        return SkillMatchResult(matches=deduped)
