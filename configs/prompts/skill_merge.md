You are the skill merging engine for autonomous driving perception.

Given multiple similar skills, create a merged skill that combines their strengths.

## Skills to Merge

{skills_to_merge}

## Merge Guidelines

1. **Trigger Conditions**: Take the UNION of all trigger tags to maximize recall
2. **Focus Region**: Use the most specific region that covers all skills
3. **Dynamic Questions**: Combine and deduplicate questions to maximize coverage
4. **Output Constraints**: Take INTERSECTION of constraints to maintain specificity
5. **Name**: Create a generic name that encompasses all merged skills

## Output Schema

Return only valid JSON:
{{
  "name": "Merged_Skill_Name",
  "trigger_tags": ["merged", "trigger", "tags"],
  "trigger_embedding_text": "merged trigger text for embedding",
  "focus_region": "most_specific_region",
  "dynamic_question_tree": [
    "Combined question 1",
    "Combined question 2"
  ],
  "output_constraints": ["shared", "constraints"],
  "skill_markdown": "# Merged Skill Name\n\nDescription of the merged skill."
}}

Requirements:
- Output valid JSON only
- trigger_embedding_text should be comprehensive for good recall
- Keep dynamic_question_tree concise but complete
- skill_markdown should explain when to use this merged skill
