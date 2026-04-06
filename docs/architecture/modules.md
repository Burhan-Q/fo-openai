# Module Map

## Dependency Graph

```
__init__.py
  └── operators.py
        ├── engine.py
        │     └── _log.py
        ├── tasks.py
        ├── utils.py
        ├── _log.py
        └── _pricing.py
              └── _log.py
```

## Module Responsibilities

### `operators.py`
The largest module. Contains both FiftyOne operators and all UI helper functions.

**Operators:**
- `OpenAIInference` — dynamic form (`resolve_input`), batch execution loop (`execute`), config export (`resolve_output`)
- `CheckOpenAIStatus` — `on_dataset_open` listener that subscribes to MongoDB change-stream for delegated job completion

**Helper categories:**
- Config resolution: `_resolve_config`, `_create_engine`, `_create_task`
- Class label resolution: `_resolve_classes_from_field`, `_get_field_classes`, `_find_label_fields`
- Field management: `_resolve_field_name`, `_write_batch_results`
- UI form builders: `_model_selector`, `_task_selector`, `_task_settings`, `_class_source_selector`, `_field_picker`, `_cost_summary`, `_output_settings`, `_logging_settings`, `_advanced_settings`, `_image_detail_selector`, `_base_url_input`, `_detection_format_selectors`, `_exemplar_tab`, `_exemplar_saved_view_picker`, `_exemplar_field_picker`, `_exemplar_label_field_picker`, `_exemplar_preview`
- Formatting: `_fmt_usd`, `_error`

### `engine.py`
Thin wrapper around `AsyncOpenAI`. Single class: `OpenAIEngine`.

- Constructor takes `model`, `api_key`, `base_url`, `max_concurrent`, `timeout`, and `**completion_kwargs`
- `completion_kwargs` are forwarded directly to `client.responses.parse()` — only user-specified values, no defaults
- `infer_batch(instructions, inputs, response_model)` → `_async_infer_batch()` → returns `list[BaseModel | Exception]`
- `_run_async()` helper handles FiftyOne's existing event loop (runs async in a thread)

### `tasks.py`
Task definitions, Pydantic response models, and response-to-FiftyOne-label conversion.

- 6 static Pydantic models: `TextResponse`, `ClassifyResponse`, `TagResponse`, `VQAResponse`, `DetectionItem`, `DetectResponse`
- 3 dynamic constrained model builders using `Literal` + `create_model` (cached via `@lru_cache`)
- `TaskConfig` class: prompt construction, response model selection, label conversion
- `get_instructions()` returns the system instructions (with optional few-shot preamble)
- `build_input()` builds the Responses API `input` array (exemplar pairs + user message)
- `_FEWSHOT_PREAMBLE` and `_TASK_VERBS` constants for few-shot system prompt framing
- `_convert_box()`: bounding box format conversion (xyxy/xywh/cxcywh → FiftyOne [x,y,w,h] in [0,1])

### `exemplars.py`
Few-shot exemplar resolution, serialization, and message building.

- `resolve_exemplars()`: 4 resolution strategies (saved view, sample IDs, tag, field)
- `serialize_exemplar()`: FiftyOne label → task Pydantic response model JSON via `model_dump_json()`
- `_fo_to_run_format()`: FiftyOne bbox [x,y,w,h] in [0,1] → run coordinate/box format (reverse of `_convert_box`)
- `build_exemplar_messages()`: constructs alternating user/assistant message pairs with explicit framing

### `utils.py`
Stateless utilities.

- Image encoding: `build_image_contents()` — parallel base64 encoding, URL passthrough, `detail` level
- Config persistence: `get_global_config()`, `save_global_config()`, `clear_global_config()` via `ExecutionStore`
- Parameter filtering: `pick_params()`, `normalize_classes()`, `parse_config_json()`

### `_log.py`
Plugin-scoped logging under the `fo_openai` namespace.

- `get_logger(name)` — returns namespaced logger
- `configure(enabled, level, log_file)` — sets up handlers per run
- `_resolve_log_path(raw)` — directory → timestamped file, existing file → auto-increment
- `summarise_errors()` — builds run summary dict for `dataset.info`
- `truncate()` — shortens text for log messages

### `_pricing.py`
Model pricing data from the LiteLLM community-maintained JSON on GitHub.

- Fetches from `raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`
- Cached to `~/.fiftyone/plugins/cache/fo-openai/model_prices.json` (24-hour TTL)
- In-memory cache per process
- Fallback chain: memory → fresh disk → remote fetch → stale disk → empty dict
- `get_model_info(model)` and `estimate_cost(model, ...)` are the public API
