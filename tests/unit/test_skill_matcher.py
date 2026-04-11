from pathlib import Path

import pytest

from ad_cornercase.providers.base import EmbeddingProvider
from ad_cornercase.schemas.skill import SkillBundle, SkillManifest, SkillMatchRequest
from ad_cornercase.skill_store.matcher import SkillMatcher
from ad_cornercase.skill_store.repository import SkillRepository


class StaticEmbeddingProvider(EmbeddingProvider):
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[text] for text in texts]


def _save_skill(repo: SkillRepository, manifest: SkillManifest, embedding: list[float]) -> None:
    repo.save_bundle(
        SkillBundle(manifest=manifest, skill_markdown=f"# {manifest.name}\n"),
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_hybrid_rerank_filters_weather_conflict_skill(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path / "skills")
    query = (
        "front_camera_daylight_urban_internal_road;daylight,clear;"
        "Lead_Vehicle_Pedestrian_Caution,Traffic_Construction_Zone,Critical_Unknown_Obstacle;"
        "Identify the main driving-relevant entities ahead and provide the safest ego-vehicle action."
    )
    lead_primary = SkillManifest(
        skill_id="lead-primary",
        name="Lead_Vehicle_Pedestrian_Caution",
        trigger_tags=["lead_vehicle", "pedestrian_present", "traffic_cones"],
        trigger_embedding_text="lead vehicle pedestrians cones urban road proceed cautiously",
        focus_region="center_lane_30m_ahead",
        output_constraints=["label_must_match: Lead_Vehicle_Pedestrian_Caution"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-1",
    )
    lead_secondary = SkillManifest(
        skill_id="lead-secondary",
        name="Lead_Vehicle_Pedestrian_Caution",
        trigger_tags=["lead_vehicle", "pedestrian_present", "worker_zone"],
        trigger_embedding_text="lead vehicle pedestrians workers curbside road yield cautiously",
        focus_region="center_lane_left_right_margin",
        output_constraints=["label_must_match: Lead_Vehicle_Pedestrian_Caution"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-2",
    )
    truck = SkillManifest(
        skill_id="truck",
        name="Overturned_Truck_Hazard_Detection",
        trigger_tags=["fog", "night", "lane_blockage", "reflective"],
        trigger_embedding_text="night fog overturned truck reflective metallic debris blocking lane obstacle",
        focus_region="lower_center_lane",
        output_constraints=["label_must_match: Overturned_Truck"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-3",
    )
    _save_skill(repository, lead_primary, [1.0, 0.0])
    _save_skill(repository, lead_secondary, [0.98, 0.02])
    _save_skill(repository, truck, [0.96, 0.04])

    provider = StaticEmbeddingProvider({query: [1.0, 0.0]})
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=provider,
        threshold=0.55,
        max_matches=3,
    )
    result = await matcher.match(
        SkillMatchRequest(
            case_id="image-experiment-1",
            sensor_context="front_camera_daylight_urban_internal_road",
            weather_tags=["daylight", "clear"],
            top_k_labels=[
                "Lead_Vehicle_Pedestrian_Caution",
                "Traffic_Construction_Zone",
                "Critical_Unknown_Obstacle",
            ],
            entropy=0.7,
            trigger_text=query,
        )
    )

    assert [match.skill_id for match in result.matches] == ["lead-primary"]


@pytest.mark.asyncio
async def test_hybrid_rerank_keeps_fog_skill_without_label_overlap(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path / "skills")
    query = (
        "front_camera_fog_degraded;fog,night;"
        "Critical_Unknown_Obstacle,Construction_Debris,Traffic_Sign;"
        "What is the most likely hazardous obstacle ahead, and what should the ego vehicle do?"
    )
    truck = SkillManifest(
        skill_id="truck",
        name="Overturned_Truck_Hazard_Detection",
        trigger_tags=["fog", "night", "lane_blockage", "reflective"],
        trigger_embedding_text="night fog overturned truck reflective metallic debris blocking lane obstacle",
        focus_region="lower_center_lane",
        output_constraints=["must_include_action: True"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-3",
    )
    daylight_skill = SkillManifest(
        skill_id="lead-daylight",
        name="Lead_Vehicle_Pedestrian_Caution",
        trigger_tags=["daylight", "clear", "lead_vehicle", "pedestrian_present"],
        trigger_embedding_text="daylight clear lead vehicle pedestrians curbside yield cautiously",
        focus_region="center_lane_left_right_margin",
        output_constraints=["label_must_match: Lead_Vehicle_Pedestrian_Caution"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-4",
    )
    _save_skill(repository, truck, [1.0, 0.0])
    _save_skill(repository, daylight_skill, [0.94, 0.06])

    provider = StaticEmbeddingProvider({query: [1.0, 0.0]})
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=provider,
        threshold=0.55,
        max_matches=3,
    )
    result = await matcher.match(
        SkillMatchRequest(
            case_id="fog-case",
            sensor_context="front_camera_fog_degraded",
            weather_tags=["fog", "night"],
            top_k_labels=[
                "Critical_Unknown_Obstacle",
                "Construction_Debris",
                "Traffic_Sign",
            ],
            entropy=0.8,
            trigger_text=query,
        )
    )

    assert result.matches
    assert result.matches[0].skill_id == "truck"


@pytest.mark.asyncio
async def test_same_family_variants_collapse_to_single_match(tmp_path: Path) -> None:
    repository = SkillRepository(tmp_path / "skills")
    query = (
        "front_camera_fog_degraded;fog,night;"
        "Critical_Unknown_Obstacle,Construction_Debris,Traffic_Sign;"
        "What is the most likely hazardous obstacle ahead, and what should the ego vehicle do?"
    )
    truck_detection = SkillManifest(
        skill_id="truck-detection",
        name="Overturned_Truck_Hazard_Detection",
        trigger_tags=["fog", "night", "lane_blockage", "reflective"],
        trigger_embedding_text="night fog overturned truck reflective metallic debris blocking lane obstacle",
        focus_region="lower_center_lane",
        output_constraints=["label_must_match: Overturned_Truck"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-3",
    )
    truck_classification = SkillManifest(
        skill_id="truck-classification",
        name="Overturned_Truck_Hazard_Classification",
        trigger_tags=["fog", "night", "lane_blockage", "debris"],
        trigger_embedding_text="night fog overturned truck lane blockage debris classify truck hazard",
        focus_region="lower_center_lane",
        output_constraints=["label_must_match: Overturned_Truck"],
        fallback_label="Critical_Unknown_Obstacle",
        source_case_id="case-4",
    )
    _save_skill(repository, truck_detection, [1.0, 0.0])
    _save_skill(repository, truck_classification, [0.99, 0.01])

    provider = StaticEmbeddingProvider({query: [1.0, 0.0]})
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=provider,
        threshold=0.55,
        max_matches=3,
    )
    result = await matcher.match(
        SkillMatchRequest(
            case_id="fog-case",
            sensor_context="front_camera_fog_degraded",
            weather_tags=["fog", "night"],
            top_k_labels=[
                "Critical_Unknown_Obstacle",
                "Construction_Debris",
                "Traffic_Sign",
            ],
            entropy=0.8,
            trigger_text=query,
        )
    )

    assert [match.skill_id for match in result.matches] == ["truck-detection"]
