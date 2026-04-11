#!/usr/bin/env python3
"""Demo of the Universal Hierarchical Perception Framework.

This demo shows how the framework can be used across different benchmarks
(DTPQA, DriveLM, CODA-LM) without code changes.
"""

import asyncio
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ad_cornercase.universal_framework import (
    PerceptionTask,
    TaskSchema,
    TaskType,
    InputModality,
    TASK_REGISTRY,
    EdgeCloudOrchestrator,
    OrchestratorConfig,
    SkillLibrary,
    ReflectionPolicy,
    ReflectionTrigger,
    POLICY_DISTANCE_AWARE,
)


async def demo_dtpqa_task():
    """Demo: DTPQA pedestrian detection task."""
    print("\n" + "="*60)
    print("DEMO 1: DTPQA Pedestrian Detection")
    print("="*60)

    # Create task using registered schema
    schema = TASK_REGISTRY.get("dtpqa_pedestrian_detection")

    # Far-range pedestrian (difficult case)
    far_task = PerceptionTask(
        schema=schema,
        inputs={
            "image_path": "data/images/scene_001.jpg",
            "question": "Are there any pedestrians crossing the road?",
        },
        ground_truth={"answer": "Yes", "explanation": "Pedestrian at 45m"},
        metadata={
            "distance_meters": 45.0,
            "distance_group": "far",
            "lighting_condition": "daylight",
            "weather_condition": "clear",
            "scene_type": "urban_road",
        },
    )

    print(f"Task ID: {far_task.task_id}")
    print(f"Category: {far_task.task_category}")
    print(f"Difficulty: {far_task.difficulty_score:.2f}")
    print(f"Distance: {far_task.metadata['distance_meters']}m")

    # Near-range pedestrian (easy case)
    near_task = PerceptionTask(
        schema=schema,
        inputs={
            "image_path": "data/images/scene_002.jpg",
            "question": "Are there any pedestrians crossing the road?",
        },
        ground_truth={"answer": "Yes", "explanation": "Pedestrian at 10m"},
        metadata={
            "distance_meters": 10.0,
            "distance_group": "near",
            "lighting_condition": "daylight",
            "weather_condition": "clear",
            "scene_type": "urban_road",
        },
    )

    print(f"\nNear Task Difficulty: {near_task.difficulty_score:.2f}")
    print(f"Near Distance: {near_task.metadata['distance_meters']}m")

    return far_task, near_task


async def demo_reflection_policies():
    """Demo: Different reflection policies."""
    print("\n" + "="*60)
    print("DEMO 2: Reflection Policies")
    print("="*60)

    # Policy 1: Conservative (high entropy threshold)
    conservative = ReflectionPolicy(
        name="conservative",
        trigger=ReflectionTrigger.ENTROPY_THRESHOLD,
        entropy_threshold=1.2,
    )

    # Policy 2: Distance-aware (different thresholds per distance)
    distance_aware = POLICY_DISTANCE_AWARE

    # Policy 3: Custom far-range focused
    far_focused = ReflectionPolicy(
        name="far_focused",
        trigger=ReflectionTrigger.DISTANCE_BASED,
        distance_ranges={
            "far": {"min": 30, "max": float("inf"), "force_reflect": True},
            "near": {"min": 0, "max": 30, "force_reflect": False},
        },
    )

    # Test with a far-range task
    test_task = PerceptionTask(
        metadata={"distance_meters": 45.0},
        ground_truth={},
        inputs={},
    )

    edge_output = {"entropy": 0.8, "confidence": 0.6}

    print("\nTesting with far-range task (45m):")
    for policy in [conservative, distance_aware, far_focused]:
        should_reflect, reason = policy.should_reflect(test_task, edge_output, [])
        print(f"  {policy.name:20s}: {'YES' if should_reflect else 'NO'} ({reason})")


async def demo_orchestrator():
    """Demo: Full orchestrator with skill learning."""
    print("\n" + "="*60)
    print("DEMO 3: Edge-Cloud Orchestrator")
    print("="*60)

    # Initialize skill library
    skill_lib = SkillLibrary(storage_dir=Path("/tmp/demo_skills"))

    # Configure orchestrator with distance-aware policy
    config = OrchestratorConfig(
        reflection_policy=POLICY_DISTANCE_AWARE,
        skill_library=skill_lib,
        enable_skill_learning=True,
        enable_skill_application=True,
    )

    orchestrator = EdgeCloudOrchestrator(config)

    # Mock inference functions
    def mock_edge_inference(task: PerceptionTask) -> dict:
        """Simulate edge model inference."""
        # Simulate lower confidence for far-range
        distance = task.metadata.get("distance_meters", 20)
        if distance > 30:
            return {
                "answer": "No",
                "confidence": 0.5,
                "entropy": 0.7,
                "reasoning": "No visible pedestrians",
            }
        return {
            "answer": "Yes",
            "confidence": 0.9,
            "entropy": 0.3,
            "reasoning": "Pedestrian visible",
        }

    def mock_cloud_inference(task: PerceptionTask, edge_output: dict) -> dict:
        """Simulate cloud model inference."""
        # Cloud model is more accurate
        return {
            "answer": task.ground_truth.get("answer", "Yes"),
            "confidence": 0.95,
            "reasoning": "Cloud analysis with higher resolution",
        }

    # Process tasks
    tasks = [
        PerceptionTask(
            schema=TASK_REGISTRY.get("dtpqa_pedestrian_detection"),
            inputs={"image_path": "img1.jpg", "question": "Any pedestrians?"},
            ground_truth={"answer": "Yes"},
            metadata={"distance_meters": 45.0, "distance_group": "far"},
        ),
        PerceptionTask(
            schema=TASK_REGISTRY.get("dtpqa_pedestrian_detection"),
            inputs={"image_path": "img2.jpg", "question": "Any pedestrians?"},
            ground_truth={"answer": "No"},
            metadata={"distance_meters": 15.0, "distance_group": "near"},
        ),
        PerceptionTask(
            schema=TASK_REGISTRY.get("dtpqa_pedestrian_detection"),
            inputs={"image_path": "img3.jpg", "question": "Any pedestrians?"},
            ground_truth={"answer": "Yes"},
            metadata={"distance_meters": 50.0, "distance_group": "far"},
        ),
    ]

    print("\nProcessing tasks:")
    for i, task in enumerate(tasks, 1):
        result = await orchestrator.process(
            task,
            mock_edge_inference,
            mock_cloud_inference,
        )
        print(f"\nTask {i} ({task.metadata['distance_group']}):")
        print(f"  Edge answer: {result['edge_output']['answer']}")
        print(f"  Reflection triggered: {result['reflection_triggered']}")
        print(f"  Reason: {result['reflection_reason']}")
        if result['reflection_triggered']:
            print(f"  Cloud answer: {result['cloud_output']['answer']}")
            print(f"  New skill: {result.get('new_skill_id', 'None')}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")

    # Print stats
    stats = orchestrator.get_stats()
    print("\n" + "-"*40)
    print("Final Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Edge-only: {stats['edge_only']} ({stats['edge_only_ratio']:.1%})")
    print(f"  With reflection: {stats['with_reflection']} ({stats['reflection_ratio']:.1%})")
    print(f"  Skills applied: {stats['skills_applied']}")
    print(f"  New skills learned: {stats['new_skills_learned']}")


async def demo_custom_benchmark():
    """Demo: Adding a custom benchmark without code changes."""
    print("\n" + "="*60)
    print("DEMO 4: Custom Benchmark Registration")
    print("="*60)

    # Register a new benchmark schema at runtime
    from ad_cornercase.universal_framework.core import TaskSchema

    custom_schema = TaskSchema(
        name="my_benchmark_object_counting",
        description="Count objects in traffic scenes",
        task_type=TaskType.OBJECT_DETECTION,
        input_modality=InputModality.IMAGE,
        required_inputs=["image_path", "object_type"],
        output_schema={
            "count": "int",
            "locations": "list[box]",
        },
        benchmark_source="my_custom_benchmark",
    )

    TASK_REGISTRY.register(custom_schema)

    # Create a task using the new schema
    custom_task = PerceptionTask(
        schema=custom_schema,
        inputs={
            "image_path": "data/my_benchmark/scene_001.jpg",
            "object_type": "vehicles",
        },
        ground_truth={"count": 5},
        metadata={
            "scene_complexity": "high",
            "lighting": "night",
        },
    )

    print(f"\nCustom task created:")
    print(f"  Task ID: {custom_task.task_id}")
    print(f"  Category: {custom_task.task_category}")
    print(f"  Schema: {custom_task.schema.name}")
    print(f"  Benchmark: {custom_task.schema.benchmark_source}")

    print("\nRegistered schemas:")
    for schema_name in TASK_REGISTRY.list_schemas():
        print(f"  - {schema_name}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Universal Hierarchical Perception Framework Demo")
    print("="*60)
    print("\nThis demo shows a generalizable system that works")
    print("across different benchmarks without code changes.")

    await demo_dtpqa_task()
    await demo_reflection_policies()
    await demo_orchestrator()
    await demo_custom_benchmark()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Features:")
    print("  ✓ Universal task representation")
    print("  ✓ Configurable reflection policies")
    print("  ✓ Skill-based learning and application")
    print("  ✓ Runtime benchmark registration")
    print("  ✓ Cross-benchmark compatibility")


if __name__ == "__main__":
    asyncio.run(main())
