#!/usr/bin/env uv run python3
"""One-shot skill extraction from completed predictions."""

import asyncio
import json
from pathlib import Path

from ad_cornercase.bootstrap import build_prompt_renderer, build_structured_provider, load_settings
from ad_cornercase.cloud.reflector import ReflectionLLMOutput
from ad_cornercase.cloud.skill_compiler import SkillCompiler
from ad_cornercase.logging import configure_logging


async def compile_skill(runtime_settings, project_settings, reflection_data, case_id):
    from ad_cornercase.cloud.skill_compiler import SkillCompileOutput

    provider = build_structured_provider(runtime_settings)
    prompt_renderer = build_prompt_renderer(runtime_settings)
    skill_compiler = SkillCompiler(project_settings=project_settings)

    llm_output = ReflectionLLMOutput.model_validate(reflection_data)

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
    return bundle


async def main():
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)

    run_id = "dtpqa_200_final_20260401_214638"
    predictions_path = Path(runtime_settings.artifacts_dir) / run_id / "predictions.jsonl"
    output_dir = Path(f"/tmp/extracted_skills_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {predictions_path}")
    print(f"Output: {output_dir}")
    print()

    with predictions_path.open("r") as f:
        lines = f.readlines()

    skills_extracted = 0
    errors = 0

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        case_id = record["case_id"]
        reflection_result = record.get("reflection_result", {})

        if not reflection_result or not reflection_result.get("should_persist_skill"):
            continue
        if reflection_result.get("new_skill"):
            continue

        try:
            print(f"Compiling skill for {case_id}...", end=" ")
            bundle = await compile_skill(runtime_settings, project_settings, reflection_result, case_id)
            skill_id = bundle.manifest.skill_id

            skill_file = output_dir / f"{skill_id}.json"
            content = {
                "manifest": bundle.manifest.model_dump(mode="json"),
                "skill_markdown": bundle.skill_markdown,
            }
            with skill_file.open("w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

            print(f"OK -> {skill_id}")
            skills_extracted += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print()
    print(f"Done: {skills_extracted} skills extracted, {errors} errors")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
