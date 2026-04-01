# Few-Shot Exemplar Samples

Optional feature that sends labeled samples as reference examples alongside
every inference call, helping the model align its output with real examples.

## Overview

When enabled, the plugin prepends user/assistant message pairs (exemplar
image + expected output JSON) before the target image in every API call.
The system prompt gains a few-shot preamble instructing the model to treat
these as reference only.

## Message Structure

```
1. system:    few-shot preamble + task instructions
2. user:      [exemplar_image_1] + "REFERENCE EXAMPLE — ..."
3. assistant: '{"label": "cat"}'
   ... (more exemplar pairs)
N. user:      [target_image] + task prompt
→ model response
```

All messages go in a **single API call per target sample**. The exemplar
images are encoded once and reused across all inference calls.

## Exemplar Source Selection

Four methods to select exemplar samples (only the active radio choice is
used during execution):

| Source | Description |
|--------|-------------|
| Saved view | Select a saved view by name |
| Sample IDs | Comma-separated sample IDs |
| Tag | Match samples by tag name |
| Field | Match samples by field value (boolean or string) |

## Exemplar Label Field

The label field must be compatible with the selected task:

| Task | Compatible Field Types |
|------|----------------------|
| classify | Classification |
| tag | Classifications, Classification |
| caption, vqa, ocr | Classification |
| detect | Detections |

Labels are serialized using the task's Pydantic response model
(`model_dump_json()`), producing JSON that matches the exact format the
structured output parser expects.

## Cost Impact

Exemplar images are included in **every** inference call. Each exemplar
adds approximately:

- ~85 tokens at "low" detail
- ~765 tokens at "high" or "auto" detail
- ~30 tokens for framing text and JSON

For example, 5 exemplars at "auto" detail add ~3,975 tokens per call.
Over 1,000 samples, that's ~3.975M extra input tokens.

The cost summary (always visible at the bottom of the form) shows a
breakdown of inference cost vs. exemplar overhead.

## Configuration

| Setting | Default | Persisted |
|---------|---------|-----------|
| `exemplars_enabled` | `False` | Yes |
| `exemplar_source` | `"saved_view"` | Yes |
| `exemplar_view_name` | `""` | Yes |
| `exemplar_sample_ids` | `""` | Yes |
| `exemplar_tag` | `""` | Yes |
| `exemplar_field_name` | `""` | Yes |
| `exemplar_field_value` | `""` | Yes |
| `exemplar_label_field` | `""` | Yes |

The cost warning threshold defaults to $5.00 and is configurable via the
`FIFTYONE_OPENAI_COST_WARN` environment variable.

## Error Handling

All exemplar errors halt execution with actionable messages:

- Missing saved view → error listing available views
- Invalid sample IDs → error listing which IDs not found
- Empty tag/field match → error with verification guidance
- Missing or empty label field → error identifying the sample
- Image encoding failure → error with filepath and details

Partial exemplar sets are never used. All configured exemplars must
succeed or the entire run halts.

## Detection Exemplars

For detection tasks, bounding boxes from FiftyOne's native format
(normalized [0,1] `[x, y, w, h]`) are converted to the configured
coordinate format and box format. The default format (normalized_1, xyxy)
requires only an xywh→xyxy conversion; non-default formats use
`_fo_to_run_format()`.

## Run Summary

When exemplars are used, the run summary in
`dataset.info["openai_runs"][field]` includes:

- `exemplars_enabled: True`
- `exemplar_count: int`
- `exemplar_source: str`
- `exemplar_label_field: str`
