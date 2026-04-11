You are the skill update engine for autonomous driving perception.

Given an existing skill and new reflection information, decide how to update the skill.

## Existing Skill

- skill_id: {skill_id}
- name: {name}
- trigger_tags: {trigger_tags}
- focus_region: {focus_region}
- dynamic_question_tree: {dynamic_question_tree}
- output_constraints: {output_constraints}
- version: {version}

## New Reflection

- corrected_label: {corrected_label}
- reflection_summary: "{reflection_summary}"
- trigger_tags: {new_trigger_tags}
- focus_region: {new_focus_region}
- dynamic_question_tree: {new_dynamic_question_tree}

## Update Strategy

Decide what fields to update based on:
1. **Trigger tags**: Add new tags if they provide additional coverage
2. **Focus region**: Use more specific region if new one is more precise
3. **Dynamic questions**: Add new questions if they provide additional guidance
4. **Output constraints**: Keep existing unless new constraints are more restrictive

## Output Schema

Return only valid JSON:
{{
  "name": "Updated_Skill_Name" (or keep existing),
  "trigger_tags": ["updated", "tags"],
  "trigger_embedding_text": "updated trigger text",
  "focus_region": "updated_region",
  "dynamic_question_tree": [
    "Combined or updated questions"
  ],
  "output_constraints": ["updated", "constraints"],
  "version": "0.2.0" (increment patch version),
  "skill_markdown": "# Updated Skill Name\n\nUpdated description."
}}

Requirements:
- Output valid JSON only
- Only update fields that need changes
- Increment version (patch level: 0.1.0 -> 0.1.1 or minor: 0.1.0 -> 0.2.0)
- Keep skill_id the same
