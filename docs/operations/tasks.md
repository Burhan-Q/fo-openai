# Tasks

The plugin supports 6 vision inference tasks.

## Task Reference

| Task | Output Type | FiftyOne Label | Default Field | Default Temp |
|------|------------|----------------|---------------|-------------|
| `caption` | `TextResponse` | `fo.Classification` | `caption` | 0.2 |
| `classify` | `ClassifyResponse` | `fo.Classification` | `classification` | 0.0 |
| `tag` | `TagResponse` | `fo.Classifications` | `tags` | 0.0 |
| `detect` | `DetectResponse` | `fo.Detections` | `detections` | 0.0 |
| `vqa` | `VQAResponse` | `fo.Classification` | `vqa_answer` | 0.2 |
| `ocr` | `TextResponse` | `fo.Classification` | `ocr_text` | 0.0 |

## Class Labels

Tasks that accept classes: `classify`, `tag`, `detect`.

Three input sources (selected via radio group in the UI):

1. **From dataset field** — picks an existing label field, extracts unique labels via `dataset.distinct()`. Supports `Classification`, `Classifications`, and `Detections` fields.
2. **Custom list** — comma-separated text input
3. **Open-ended** — no constraint; model generates freely

When classes are provided, a dynamically constrained Pydantic model with `Literal` types is used, producing `enum` constraints in the JSON schema. See [Structured Output](../architecture/structured-output.md).

## Prompt Structure

Each request sends via `client.responses.parse()`:
1. **Instructions** — task-specific system prompt (via the `instructions` parameter)
2. **Input array** — optional exemplar pairs + user message with image content + text prompt

The `detect` task dynamically builds its system and user prompts based on the coordinate format and box format selected.

Tasks with open-ended vs constrained variants (`classify`, `tag`) use different system prompts depending on whether classes are provided.

## Custom Prompt Override

Users can override the default user prompt via a collapsible "Custom prompt" toggle. The override replaces the entire user-facing prompt text while keeping the system prompt and image content intact.

## Output Fields

Results are written to `openai_infer_{task}` (e.g., `openai_infer_classify`). If the field exists, it auto-increments: `openai_infer_classify1`, `openai_infer_classify2`, etc. The "Overwrite last result" toggle reuses the highest existing field instead.

Errors are written to `{field_name}_error` with `[API]` or `[Parse]` prefixes.
