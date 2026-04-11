#!/usr/bin/env uv run python3
"""Extract skills from completed reflection results and save to skill directory.

Usage:
    uv run python extract_skills_periodically.py --run-id dtpqa_200_final_20260401_214638 --every 20

This script monitors a predictions.jsonl file and extracts skills from completed
reflection results, compiling and saving them to a local skill directory.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from ad_cornercase.bootstrap import build_cloud_service, load_settings
from ad_cornercase.cloud.reflector import ReflectionLLMOutput
from ad_cornercase.cloud.skill_compiler import SkillCompiler
from ad_cornercase.config import RuntimeSettings
from ad_cornercase.logging import configure_logging


async def compile_and_save_skill(
    runtime_settings: RuntimeSettings,
    reflection_data: dict,
    case_id: str,
    output_dir: Path,
) -> str | None:
    """Compile a skill from reflection data and save to disk."""
    from ad_cornercase.prompts.renderer import PromptRenderer
    from ad_cornercase.providers.siliconflow import SiliconFlowProvider

    # Create providers
    provider = SiliconFlowProvider(api_key=runtime_settings.siliconflow_api_key)
    prompt_renderer = PromptRenderer()
    skill_compiler = SkillCompiler(prompt_renderer=prompt_renderer)

    # Build reflection LLM output
    try:
        llm_output = ReflectionLLMOutput.model_validate(reflection_data)
    except Exception as e:
        print(f"  [WARN] Failed to validate reflection data for {case_id}: {e}")
        return None

    if not llm_output.should_persist_skill:
        return None

    # Compile skill
    try:
        compile_instructions = prompt_renderer.load("skill_compile.md")
        compile_prompt = json.dumps(llm_output.model_dump(mode="json"), ensure_ascii=False)
        compile_response = await provider.generate_structured(
            model=runtime_settings.cloud_model,
            instructions=compile_instructions,
            prompt=compile_prompt,
            response_model=SkillCompileOutput,
            metadata={"case_id": case_id, "stage": "skill_compile"},
        )
        bundle = skill_compiler.compile_bundle(
            case_id=case_id,
            output=compile_response.parsed,
            reflection_summary=llm_output.reflection_summary,
        )

        # Save skill
        skill_id = bundle.manifest.skill_id
        skill_file = output_dir / f"{skill_id}.json"
        skill_content = {
            "manifest": bundle.manifest.model_dump(mode="json"),
            "skill_markdown": bundle.skill_markdown,
        }
        with skill_file.open("w", encoding="utf-8") as f:
            json.dump(skill_content, f, ensure_ascii=False, indent=2)
        return skill_id
    except Exception as e:
        print(f"  [ERROR] Failed to compile skill for {case_id}: {e}")
        return None


async def extract_skills_from_predictions(
    predictions_path: Path,
    output_dir: Path,
    runtime_settings: RuntimeSettings,
    processed_ids: set[str],
) -> tuple[set[str], int]:
    """Extract skills from predictions file and save to output directory.

    Returns:
        Tuple of (newly_processed_ids, skills_extracted)
    """
    if not predictions_path.exists():
        return processed_ids, 0

    new_processed = set(processed_ids)
    skills_extracted = 0

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = record.get("case_id", "unknown")
            if case_id in new_processed:
                continue

            reflection_result = record.get("reflection_result")
            if not reflection_result:
                continue

            # Check if should persist
            if not reflection_result.get("should_persist_skill"):
                continue

            # Skip if already has new_skill (already compiled)
            if reflection_result.get("new_skill"):
                continue

            # Compile and save
            skill_id = await compile_and_save_skill(
                runtime_settings,
                reflection_result,
                case_id,
                output_dir,
            )
            if skill_id:
                print(f"  [SAVED] {skill_id} from {case_id}")
                skills_extracted += 1
                new_processed.add(case_id)

    return new_processed, skills_extracted


async def main(run_id: str, every: int, output_dir: str | None):
    # Setup
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)

    predictions_path = Path(runtime_settings.artifacts_dir) / run_id / "predictions.jsonl"
    output_path = Path(output_dir) if output_dir else Path(f"/tmp/skills_extracted_{run_id}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring: {predictions_path}")
    print(f"Output dir: {output_path}")
    print(f"Extract every: {every} samples")
    print()

    processed_ids: set[str] = set()
    last_count = 0

    while True:
        try:
            # Count current predictions
            if predictions_path.exists():
                with predictions_path.open("r", encoding="utf-8") as f:
                    current_count = sum(1 for line in f if line.strip())
            else:
                current_count = 0

            print(f"[{time.strftime('%H:%M:%S')}] Progress: {current_count} samples, checking for skills...")

            # Extract skills if we've crossed a multiple of 'every'
            if current_count >= last_count + every or current_count % every == 0:
                if current_count > last_count:
                    _, extracted = await extract_skills_from_predictions(
                        predictions_path,
                        output_path,
                        runtime_settings,
                        processed_ids,
                    )
                    if extracted > 0:
                        print(f"  -> Extracted {extracted} new skills")
                    last_count = current_count

            # Check if experiment seems complete (no new samples for a while)
            await asyncio.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\nStopping skill extraction.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract skills from completed reflections")
    parser.add_argument("--run-id", required=True, help="Run ID to monitor")
    parser.add_argument("--every", type=int, default=20, help="Extract every N samples")
    parser.add_argument("--output-dir", help="Output directory for skills")
    args = parser.parse_args()

    asyncio.run(main(args.run_id, args.every, args.output_dir))
