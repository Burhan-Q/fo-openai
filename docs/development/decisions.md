# Design Decisions

Key choices made during development, with rationale.

## OpenAI SDK over LiteLLM

**Decision:** Use the OpenAI Python SDK directly instead of LiteLLM.

**Why:** LiteLLM was originally chosen for provider routing and cost estimation. In practice:
- LiteLLM's `response_format=PydanticModel` did not parse responses into Pydantic instances — it just sent the JSON schema and returned raw strings, requiring manual `model_validate_json()` parsing
- LiteLLM had no per-request `timeout` that worked reliably in async context
- LiteLLM added ~1MB of dependencies and complexity for what amounted to a thin wrapper
- A supply chain attack on LiteLLM required pinning to a specific version

The OpenAI SDK's `client.beta.chat.completions.parse()` handles Pydantic parsing natively (`message.parsed`), surfaces refusals (`message.refusal`), and supports `timeout` as a constructor parameter.

**Cost estimation** was the only LiteLLM feature retained — implemented by fetching the community pricing JSON from GitHub directly. See [Cost Estimation](../operations/cost-estimation.md).

## Only send user-specified SDK parameters

**Decision:** `completion_kwargs` only contains values the user explicitly set in the form. Unset parameters are omitted entirely.

**Why:** Newer OpenAI models reject the `max_tokens` parameter (replaced by `max_completion_tokens`). Sending hardcoded defaults caused 100% failure rates (`BadRequestError: Unsupported parameter: 'max_tokens'`). By omitting unset params, the model uses its own defaults and new parameters don't break old models.

## `_DEFAULTS` only for plugin-level settings

**Decision:** `_DEFAULTS` contains only batch/concurrency/format settings (`batch_size`, `max_concurrent`, `max_workers`, `coordinate_format`, `box_format`, `image_detail`). No SDK parameters (`temperature`, `max_completion_tokens`, `top_p`, `seed`).

**Why:** Plugin-level settings (how the plugin operates) have sensible defaults. SDK parameters (how the model behaves) should be left to the model unless the user explicitly overrides them.

## FiftyOne secrets for API key resolution

**Decision:** Declare both `FIFTYONE_OPENAI_API_KEY` and `OPENAI_API_KEY` in `fiftyone.yml` and access via `ctx.secrets[]`.

**Why:** FiftyOne's `SecretsDictionary` reads from environment variables automatically. Declaring both names means users with either env var convention are supported without manual `os.environ.get()` in code. The `SecretsDictionary` returns `""` (not `None`) for declared-but-unset secrets, so the `or` chain works correctly.

## Pixel coordinates removed from UI

**Decision:** The `pixel` coordinate format is not exposed in the detection format dropdown.

**Why:** OpenAI resizes images internally before vision processing. Bounding box coordinates returned by the model are relative to the resized image, not the original. There is no reliable way to reverse this scaling. Normalized coordinates (0-1 or 0-1000) work universally regardless of internal resizing. The code still supports pixel format for potential future use.

## `TextFieldView` for prompt override (not `TextView`)

**Decision:** Use `types.TextFieldView()` for the custom prompt input.

**Why:** `types.TextView` is display-only (renders text, not an input). `types.TextFieldView` is the actual text input. The `multiline=True` kwarg is accepted by the Python side but not honored by the FiftyOne frontend — the field renders as single-line regardless. `CodeView` would render multi-line but has code-editor styling inappropriate for natural language prompts. The single-line `TextFieldView` is the best available option; it scrolls horizontally for long prompts.
