"""Synthetic demo cases for real SiliconFlow runs."""

from __future__ import annotations

from pathlib import Path

from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.scene_graph import SceneGraphTriplet


def build_siliconflow_demo_cases(asset_path: Path) -> list[AnomalyCase]:
    ground_truth = [
        SceneGraphTriplet(subject="Ego-vehicle", relation="Yield_to", object="Overturned_Truck_30m_Ahead")
    ]
    return [
        AnomalyCase(
            case_id="siliconflow-demo-1",
            frame_id="frame-001",
            image_path=asset_path,
            question="What is the most likely hazardous obstacle ahead, and what should the ego vehicle do?",
            ground_truth_answer="Overturned_Truck",
            sensor_context="front_camera_fog_degraded",
            weather_tags=["fog", "night"],
            ground_truth_triplets=ground_truth,
            metadata={
                "force_reflection": True,
                "scene_hint": (
                    "Nighttime dense fog. Lower center lane has a large metallic obstacle with reflective tilted stripes, "
                    "scattered debris nearby, partial truck chassis contour, and lane blockage risk around 30 meters ahead."
                ),
            },
        ),
        AnomalyCase(
            case_id="siliconflow-demo-2",
            frame_id="frame-002",
            image_path=asset_path,
            question="Identify the obstacle pattern ahead and provide the safest recommendation.",
            ground_truth_answer="Overturned_Truck",
            sensor_context="front_camera_fog_degraded",
            weather_tags=["fog", "night"],
            ground_truth_triplets=ground_truth,
            metadata={
                "scene_hint": (
                    "Night fog scene. Lower center contains reflective diagonal metal stripes, bulky vehicle body fragments, "
                    "and debris cluster consistent with an overturned truck occupying the lane."
                ),
            },
        ),
    ]


def build_experiment_image_cases(image_path: Path) -> list[AnomalyCase]:
    ground_truth = [
        SceneGraphTriplet(subject="ego_vehicle", relation="follows", object="lead_vehicle_ahead"),
        SceneGraphTriplet(subject="pedestrians", relation="are", object="roadside"),
        SceneGraphTriplet(subject="ego_vehicle", relation="should", object="slow_down_and_yield"),
    ]
    return [
        AnomalyCase(
            case_id="image-experiment-1",
            frame_id="image-frame-001",
            image_path=image_path,
            question="Identify the main driving-relevant entities ahead and provide the safest ego-vehicle action.",
            ground_truth_answer="Lead_Vehicle",
            sensor_context="front_camera_daylight_urban_internal_road",
            weather_tags=["daylight", "clear"],
            ground_truth_triplets=ground_truth,
            metadata={
                "force_reflection": True,
            },
        ),
        AnomalyCase(
            case_id="image-experiment-2",
            frame_id="image-frame-002",
            image_path=image_path,
            question="What traffic interaction pattern is visible in this lane, and what is the safest recommendation?",
            ground_truth_answer="Lead_Vehicle",
            sensor_context="front_camera_daylight_urban_internal_road",
            weather_tags=["daylight", "clear"],
            ground_truth_triplets=ground_truth,
            metadata={},
        ),
    ]
