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
7. **Resolve output field** — `openai_infer_{task}`, auto-incremented or overwritten
8. **Compute instructions** — `task.get_instructions(exemplar_messages)` once (shared across all samples)
9. **Build metadata** — if `log_metadata`, capture prompt and inference config
10. **Clear stale errors** — if overwriting, null out the prior `_error` field
11. **Image dimensions** — if detect + pixel coordinates, `compute_metadata()` for widths/heights

## Phase 3: Batch Loop

For each batch of `batch_size` samples:

12. **Encode images** — `build_image_contents()` in parallel threads, with `image_detail` level
13. **Build input** — `task.build_input(image_content)` per sample (exemplar pairs + user message with image + text)
14. **Infer** — `engine.infer_batch(instructions, inputs, response_model)` → `asyncio.gather` of `client.responses.parse()` calls under semaphore
15. **Process responses** — for each `BaseModel | Exception`:
    - `Exception` → `[API]` error string, logged, counted
    - `BaseModel` → `task.parse_response(parsed)` converts to FiftyOne label
    - Parse failure → `[Parse]` error string, logged, counted
16. **Write results** — `dataset.set_values()` for results + errors (dynamic fields)
17. **Report progress** — `set_progress` via trigger (immediate) or `ctx.set_progress` (delegated)

## Phase 4: Finalization

18. **Run summary** — always written to `dataset.info["openai_runs"][field_name]` with error counts and first N error samples
19. **Persist config** — `save_global_config(params)` + `dataset.info["_openai_config"]` (excludes `api_key`)
20. **Notify** — delegated: signal `openai_status` store; immediate: `reload_dataset`

## Data Flow Diagram

```
User form params
    ↓
_create_engine() → OpenAIEngine(model, api_key, **completion_kwargs)
_create_task()   → TaskConfig(task, classes, prompts)
    ↓
target_view → [ids, filepaths]
instructions = task.get_instructions()  (computed once)
    ↓
┌─ Batch loop ─────────────────────────────────────┐
│ filepaths → build_image_contents() → image_dicts │
│ image_dicts → task.build_input() → input_msgs    │
│ (instructions, inputs) → engine.infer_batch()    │
│     → [BaseModel|Exc]                            │
│ BaseModel → task.parse_response() → fo.Label     │
│ results + errors → dataset.set_values()          │
└──────────────────────────────────────────────────┘
    ↓
dataset.info["openai_runs"] ← run summary
dataset.info["_openai_config"] ← persisted config
```
