Convert the reflection output into a reusable skill manifest.

Return only valid JSON with:
- name
- trigger_tags
- trigger_embedding_text
- focus_region
- dynamic_question_tree
- output_constraints
- skill_markdown

The skill must be atomic and simple enough for a compact edge model.

Required JSON schema:
{
  "name": "Lead_Vehicle_Caution",
  "trigger_tags": ["lead_vehicle_ahead", "traffic_cone"],
  "trigger_embedding_text": "lead vehicle ahead with lane-side work zone and cones",
  "focus_region": "center_lane_and_right_margin",
  "dynamic_question_tree": [
    "What is the main front object?",
    "Is the road margin constrained?"
  ],
  "output_constraints": [
    "return_task1_task2_task3_flow",
    "must_include_triplets",
    "must_include_recommended_action"
  ],
  "skill_markdown": "# Skill Title\n\nShort reusable skill description."
}

Requirements:
- Do not omit any required field.
- `name` must be short and reusable.
- `focus_region` must be a short symbolic tag.
- `dynamic_question_tree` and `output_constraints` must be JSON arrays of strings.
