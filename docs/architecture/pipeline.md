# Inference Pipeline

Step-by-step execution flow in `OpenAIInference.execute()`.

## Phase 1: Configuration

1. **Mode dispatch** — `config_mode` is `manual`, `json`, or `reset`
   - `reset`: clears global + dataset config, reloads, returns
   - `json`: parses JSON, merges into `params`, validates required keys
   - `manual`: proceeds with form values
2. **Configure logging** — `configure_logging(enabled, level, log_file)`
3. **Resolve classes from field** — if `class_source == "field"`, extracts unique labels from the selected dataset field via `dataset.distinct()`
4. **Create engine** — `_create_engine(params, ctx.secrets)` resolves API key (params → `FIFTYONE_OPENAI_API_KEY` → `OPENAI_API_KEY`), builds `completion_kwargs` from user-set values only
5. **Create task** — `_create_task(params)` builds `TaskConfig` with prompts and classes

## Phase 2: Preparation

6. **Get target view** — `ctx.target_view()` for sample IDs and filepaths
7. **Resolve exemplars** — if `exemplars_enabled`, lazy-import `exemplars.py`, call `resolve_exemplars()` then `build_exemplar_messages()` to build user/assistant message pairs (encoded once, reused for every sample)
8. **Resolve output field** — `openai_infer_{task}`, auto-incremented or overwritten
9. **Compute instructions** — `task.get_instructions(exemplar_messages)` once (shared across all samples)
10. **Build metadata** — if `log_metadata`, capture prompt and inference config
11. **Clear stale errors** — if overwriting, null out the prior `_error` field
12. **Image dimensions** — if detect task, `compute_metadata()` for widths/heights (used in prompt and coordinate conversion)

## Phase 3: Batch Loop

For each batch of `batch_size` samples:

13. **Encode images** — `build_image_contents()` in parallel threads, with `image_detail` level
14. **Build input** — `task.build_input(image_content, exemplar_messages, image_width, image_height)` per sample (exemplar pairs + user message with image + text)
15. **Infer** — `engine.infer_batch(instructions, inputs, response_model)` → `asyncio.gather` of `client.responses.parse()` calls under semaphore
16. **Process responses** — for each `BaseModel | Exception`:
    - `Exception` → `[API]` error string, logged, counted
    - `BaseModel` → `task.parse_response(parsed)` converts to FiftyOne label
    - Parse failure → `[Parse]` error string, logged, counted
17. **Write results** — `dataset.set_values()` for results + errors (dynamic fields)
18. **Report progress** — `set_progress` via trigger (immediate) or `ctx.set_progress` (delegated); throttled to ≥3 s between updates

## Phase 4: Finalization

19. **Run summary** — always written to `dataset.info["openai_runs"][field_name]` with error counts, first N error samples, and exemplar metadata (if enabled)
20. **Persist config** — `save_global_config(params)` + dataset-scoped `ExecutionStore` (excludes `api_key`)
21. **Notify** — delegated: final `ctx.set_progress(1.0)` (status tracked by FiftyOne delegation infrastructure); immediate: `reload_dataset`

## Data Flow Diagram

```
User form params
    ↓
_create_engine() → OpenAIEngine(model, api_key, **completion_kwargs)
_create_task()   → TaskConfig(task, classes, prompts)
    ↓
target_view → [ids, filepaths]
exemplar_messages = build_exemplar_messages()  (if enabled, built once)
instructions = task.get_instructions(exemplar_messages)  (computed once)
    ↓
┌─ Batch loop ─────────────────────────────────────────────────────┐
│ filepaths → build_image_contents() → image_dicts                 │
│ image_dicts → task.build_input(img, exemplar_messages) → inputs  │
│ (instructions, inputs) → engine.infer_batch()                    │
│     → [BaseModel|Exc]                                            │
│ BaseModel → task.parse_response() → fo.Label                     │
│ results + errors → dataset.set_values()                          │
└──────────────────────────────────────────────────────────────────┘
    ↓
dataset.info["openai_runs"] ← run summary (+ exemplar metadata)
ExecutionStore("openai_config") ← persisted config (dataset-scoped)
```
