# Structured Output

How the plugin ensures the model returns well-formed, parseable responses.

## OpenAI SDK Parse API

The plugin uses `client.beta.chat.completions.parse()` (not `client.chat.completions.create()`). This method:

- Accepts a Pydantic `BaseModel` class as `response_format`
- Converts the model's JSON schema to OpenAI's `response_format` parameter automatically
- Returns `message.parsed` as the validated Pydantic instance (or `None` on failure)
- Returns `message.refusal` as a string if the model refuses (content policy)

The engine raises `ValueError` for refusals or empty parsed responses — these are caught as `[API]` errors in the batch loop.

## Pydantic Response Models

Six static models in `tasks.py`:

| Model | Fields | Used by |
|-------|--------|---------|
| `TextResponse` | `text: str` | caption, ocr |
| `ClassifyResponse` | `label: str` | classify (open-ended) |
| `TagResponse` | `labels: list[str]` | tag (open-ended) |
| `VQAResponse` | `answer: str` | vqa |
| `DetectionItem` | `label: str`, `box: list[float]` | nested in DetectResponse |
| `DetectResponse` | `detections: list[DetectionItem]` | detect (open-ended) |

## Dynamic Constrained Models

When the user provides class labels, the plugin builds Pydantic models with `Literal` constraints at runtime using `pydantic.create_model()`. This produces JSON schemas with `enum` arrays, restricting the model's output vocabulary.

```python
# For classify with classes=["cat", "dog"]:
Literal["cat", "dog"]  →  {"enum": ["cat", "dog"]} in JSON schema
```

Three builder functions:
- `_constrained_classify_model(classes)` — `label: Literal[...]`
- `_constrained_tag_model(classes)` — `labels: list[Literal[...]]`
- `_constrained_detect_model(classes)` — nested: detection items with `label: Literal[...]`

## Response Flow

```
OpenAI API → message.parsed (Pydantic instance)
    ↓
task.parse_response(parsed) → FiftyOne label
```

Since the OpenAI SDK handles JSON parsing and Pydantic validation, `parse_response` only does type conversion — it never touches raw JSON strings. Parse errors at this stage would indicate a mismatch between the Pydantic model and the FiftyOne label conversion logic, not a malformed API response.

## Key Design Choice

Only user-specified completion kwargs (`temperature`, `max_completion_tokens`, `top_p`, `seed`) are sent to the API. Omitted params use the model's own defaults. This avoids `BadRequestError` from sending parameters that newer models don't support (e.g., `max_tokens` was replaced by `max_completion_tokens`). See [Design Decisions](../development/decisions.md).
