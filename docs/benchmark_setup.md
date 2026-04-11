# Benchmark Setup

## CODA-LM support in v1

The prototype supports the original CODA-LM VQA annotation format:

```text
<root>/
├── Mini/
│   └── vqa_anno/
│       ├── general_perception.jsonl
│       ├── region_perception.jsonl
│       └── driving_suggestion.jsonl
├── Val/
└── Test/
```

Each row must include at least:

- `question_id`
- `image`
- `question`
- `answer`

Optional fields used by this prototype:

- `sensor_context`
- `weather_tags`
- `crop_bbox`
- `ground_truth_triplets`
- `metadata`

## Metrics in v1

- `judge_score_mean`
- `regional_triplet_recall`
- `skill_success_rate`
- `latency_delta_ms`
- `vision_token_delta`

## DTPQA support

The prototype now supports DTPQA through the same replay/evaluation loop used by
CODA-LM. The loader is tolerant to common release variants and normalizes them
into `AnomalyCase`.

Supported annotation shapes:

- direct JSONL rows with image/question/answer fields
- nested JSON files with container keys such as `data`, `samples`, `records`,
  or `items`
- per-image QA collections under `qa_pairs`, `qas`, or `question_answers`

Recognized DTPQA fields:

- image: `image`, `image_path`, `img_path`, `img_name`, `file_name`
- question: `question`, `query`, `prompt`
- answer: `answer`, `gt_answer`, `correct_answer`, `label`
- options: `options`, `choices`, `candidate_answers`, `answers`
- distance: `distance_meters`, `distance`, `object_distance`, `range_meters`
- distance bin: `distance_bin`, `range_label`, `distance_range`

Persisted replay metadata:

- `benchmark`
- `subset`
- `question_type`
- `distance_meters`
- `distance_bin`
- `distance_group`
- `answer_options`

## DTPQA metrics

In addition to the shared aggregate metrics, DTPQA evaluation reports:

- `exact_match_accuracy`
- `distance_bin_accuracy`
- `distance_bin_counts`
- `distance_group_accuracy`
- `distance_group_judge_score_mean`

## DriveLM status

`DriveLM` remains scaffolded and still requires a separate implementation cycle.
