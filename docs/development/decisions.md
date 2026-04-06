# Design Decisions

Key choices made during development, with rationale.

## OpenAI Responses API

**Decision:** Use the OpenAI Responses API (`client.responses.parse()`) for all inference.

**Why:** The Responses API is OpenAI's recommended endpoint, replacing the older Chat Completions beta parse method. Key advantages:
- Native `instructions` parameter separates system prompt from the input array
- `text_format` accepts Pydantic models directly for structured output
- `output_parsed` returns validated Pydantic instances
- `store=False` avoids unnecessary server-side response persistence
- Automatic prompt prefix caching with 50-75% input cost discount
- Stable GA endpoint (not `beta`)

## OpenAI SDK over LiteLLM

**Decision:** Use the OpenAI Python SDK directly instead of LiteLLM.

**Why:** LiteLLM was originally chosen for provider routing and cost estimation. In practice:
- LiteLLM's `response_format=PydanticModel` did not parse responses into Pydantic instances — it just sent the JSON schema and returned raw strings, requiring manual `model_validate_json()` parsing
- LiteLLM had no per-request `timeout` that worked reliably in async context
- LiteLLM added ~1MB of dependencies and complexity for what amounted to a thin wrapper
- A supply chain attack on LiteLLM required pinning to a specific version

**Cost estimation** was the only LiteLLM feature retained — implemented by fetching the community pricing JSON from GitHub directly. See [Cost Estimation](../operations/cost-estimation.md).

## Only send user-specified SDK parameters

**Decision:** `completion_kwargs` only contains values the user explicitly set in the form. Unset parameters are omitted entirely.

**Why:** Newer OpenAI models reject unexpected parameters. By omitting unset params, the model uses its own defaults and new parameters don't break old models.

## `_DEFAULTS` only for plugin-level settings

**Decision:** `_DEFAULTS` contains only batch/concurrency/format settings (`batch_size`, `max_concurrent`, `max_workers`, `coordinate_format`, `box_format`, `image_detail`). No SDK parameters (`temperature`, `max_output_tokens`, `top_p`).

**Why:** Plugin-level settings (how the plugin operates) have sensible defaults. SDK parameters (how the model behaves) should be left to the model unless the user explicitly overrides them.

## FiftyOne secrets for API key resolution

**Decision:** Declare both `FIFTYONE_OPENAI_API_KEY` and `OPENAI_API_KEY` in `fiftyone.yml` and access via `ctx.secrets[]`.

**Why:** FiftyOne's `SecretsDictionary` reads from environment variables automatically. Declaring both names means users with either env var convention are supported without manual `os.environ.get()` in code. The `SecretsDictionary` returns `""` (not `None`) for declared-but-unset secrets, so the `or` chain works correctly.

## Pixel coordinates as the default for detection

**Decision:** The `pixel` coordinate format is the default and recommended format for detection tasks.

**Why:** Research (arXiv:2406.13208) and industry practice (Gemini, Qwen-VL, OpenAI fine-tuning) show that integer coordinates produce more accurate bounding boxes from LLMs. Integers align with tokenizer distributions — a float like ``0.3741`` costs multiple tokens and is rare in training data, while ``374`` is 1-2 tokens and common. The original image dimensions are included in every detection prompt ("Original image: WxH pixels (landscape)") to give the model concrete grounding. Requires `compute_metadata()` which only reads file headers (fast). Both `pixel` and `normalized_1000` use integers; `normalized_1` (0-1 floats) remains available but is expected to be less accurate.

## `TextFieldView` for prompt override (not `TextView`)

**Decision:** Use `types.TextFieldView()` for the custom prompt input.

**Why:** `types.TextView` is display-only (renders text, not an input). `types.TextFieldView` is the actual text input. The `multiline=True` kwarg is accepted by the Python side but not honored by the FiftyOne frontend — the field renders as single-line regardless. `CodeView` would render multi-line but has code-editor styling inappropriate for natural language prompts. The single-line `TextFieldView` is the best available option; it scrolls horizontally for long prompts.

## Cached constrained model builders

**Decision:** The `_constrained_*_model()` functions use `@lru_cache` with tuple keys.

**Why:** Without caching, `create_model()` was called per-exemplar during serialization and per-batch during inference. For a run with N exemplars, the same model was rebuilt N+B times. Caching with `maxsize=8` covers all practical class list variants with negligible memory cost.
