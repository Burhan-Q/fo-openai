# Few-Shot Exemplar Samples — Design Spec

**Date:** 2026-03-31
**Status:** Implemented (some details changed during development — see operational docs for current behavior)

## Context

The `@Burhan-Q/fo-openai` plugin currently sends each image to OpenAI with a
system prompt and a single user message (image + task prompt). The model has no
reference for what "good output" looks like beyond the text instructions.

Users often have a handful of already-labeled samples that demonstrate the
desired output quality and format. By including these as **few-shot exemplars**
in every inference call, the model can align its responses with real examples
rather than relying solely on text descriptions. This is especially valuable
for subjective tasks (captioning style, classification edge cases) and complex
tasks (detection with specific labeling conventions).

**Goal:** Allow users to optionally provide labeled samples as few-shot
reference examples, sent alongside every inference call, to improve model
output quality and consistency.

---

## 1. Message Architecture

### Current Message Structure (no exemplars)

```
1. system:    task-specific instructions
2. user:      [target_image] + task prompt
→ model response (structured output)
```

### New Message Structure (with exemplars)

```
1. system:    few-shot preamble + task-specific instructions
2. user:      [exemplar_image_1] + "REFERENCE EXAMPLE — ..."
3. assistant: '{"label": "cat"}'
4. user:      [exemplar_image_2] + "REFERENCE EXAMPLE — ..."
5. assistant: '{"labels": ["red", "round"]}'
   ... (repeat for each exemplar)
N-1. user:    [target_image] + task prompt
→ model response (structured output)
```

All messages are sent in a **single API call** per target sample. The
user/assistant pairs are conversation context, not separate requests.

### System Prompt Preamble

When exemplars are present, the system prompt is prepended with explicit
few-shot framing:

```
You will first be shown REFERENCE EXAMPLES. These examples demonstrate the
EXACT expected output format and labeling quality. DO NOT describe or analyze
the example images. They are provided ONLY as reference for how you should
respond to the FINAL image. After the examples, you will receive one image
to {task_verb}. Apply the same approach shown in the examples.
```

This preamble is auto-generated and prepended to whatever system prompt the
user has configured (default or custom). The user is informed in the Advanced
tab that the preamble is added automatically when exemplars are active.

### Exemplar User Message Framing

Each exemplar user message contains:
- The exemplar image (base64 or URL, same encoding as target images)
- Explicit text: `"REFERENCE EXAMPLE — This image has already been labeled.
  The correct output for this image is shown in the next message. DO NOT
  analyze this image."`

### Exemplar Assistant Message Content

Each assistant message contains the expected output serialized as the task's
Pydantic response model JSON, produced via `ResponseModel.model_dump_json()`.
This ensures the exemplar output format exactly matches what the structured
output parser expects.

Examples by task:
- **classify:** `'{"label": "cat"}'`
- **tag:** `'{"labels": ["red", "round", "shiny"]}'`
- **caption:** `'{"text": "A tabby cat sleeping on a windowsill"}'`
- **detect:** `'{"detections": [{"label": "person", "box": [0.1, 0.2, 0.5, 0.8]}]}'`
- **vqa:** `'{"answer": "There are three people in the image"}'`
- **ocr:** `'{"text": "STOP"}'`

For **detection** exemplars, the default coordinate/box format is normalized
[0,1] which matches FiftyOne's native storage format — no conversion needed
in the default case. Conversion via `_fo_to_run_format()` in `exemplars.py`
is only required when the user has configured a non-default format (e.g.,
`normalized_1000` or `cxcywh`). This is the **reverse** of what
`tasks.py:_convert_boxes()` does at parse time (which converts model output
back to FiftyOne format).

---

## 2. Module Design: `exemplars.py`

New module (~200 lines) with three responsibilities:

### 2.1 Exemplar Resolution

Converts user's selection into a FiftyOne view of exemplar samples.

```python
def resolve_exemplars(
    dataset: fo.Dataset,
    source: str,           # "saved_view" | "sample_ids" | "tag" | "field"
    view_name: str | None,
    sample_ids: str | None,
    tag: str | None,
    field_name: str | None,
    field_value: Any | None,
) -> fo.DatasetView:
    """Resolve exemplar samples from the user's selected source."""
```

Four resolution strategies:
- **saved_view:** `dataset.load_saved_view(view_name)`
- **sample_ids:** `dataset.select(id_list)` (comma-separated string parsed)
- **tag:** `dataset.match_tags(tag)`
- **field:** `dataset.match(F(field_name) == field_value)` or
  `dataset.match(F(field_name) == True)` for boolean fields

Returns a `DatasetView` so callers can use `.values()` for bulk access.

### 2.2 Exemplar Serialization

Converts a sample's label field value into the task's Pydantic response model
JSON string.

```python
def serialize_exemplar(
    sample: fo.Sample,
    label_field: str,
    task_config: TaskConfig,
) -> str:
    """Serialize a sample's label field to the task's response model JSON."""
```

Reads the sample's field value and constructs the appropriate Pydantic model:
- **Classification** field → `ClassifyResponse` or constrained variant
- **Classifications** field → `TagResponse` or constrained variant
- **Detections** field → `DetectResponse` with coordinate conversion
- **String** field → `TextResponse` (caption/ocr) or `VQAResponse`

Uses the same constrained model builders from `tasks.py` when classes are
configured, ensuring exemplar JSON matches the exact schema the model will be
forced to produce.

### 2.3 Exemplar Message Building

Constructs the full list of user/assistant message pairs.

```python
def build_exemplar_messages(
    exemplar_view: fo.DatasetView,
    label_field: str,
    task_config: TaskConfig,
    image_detail: str = "auto",
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Build user/assistant message pairs for all exemplar samples."""
```

For each exemplar sample:
1. Encode image via `utils.build_image_contents([filepath], ...)`
2. Serialize label field via `serialize_exemplar()`
3. Construct user message: image + "REFERENCE EXAMPLE ..." framing text
4. Construct assistant message: serialized JSON string

Returns a flat list of message dicts (alternating user/assistant).

---

## 3. Changes to `tasks.py`

### 3.1 `TaskConfig.build_messages()` Extension

```python
def build_messages(
    self,
    image_content: dict[str, Any],
    exemplar_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
```

When `exemplar_messages` is provided:
1. System prompt gains the few-shot preamble (prepended to existing prompt)
2. Exemplar messages are inserted after system prompt, before user query
3. Final user message (target image + task prompt) remains unchanged

When `exemplar_messages` is `None` (default): behavior is identical to today.

### 3.2 Few-Shot Preamble Constant

A new constant `_FEWSHOT_PREAMBLE` template string in `tasks.py`, parameterized
by task verb (e.g., "classify", "caption", "detect objects in").

---

## 4. Tabbed UI Restructuring

### 4.1 Five-Tab Layout

> **Implementation note:** The final implementation uses 5 tabs — Logging was promoted to its own tab instead of being placed outside the tab layout.

The operator form is restructured using `types.TabsView()`:

```
┌─────────┬────────┬────────────┬──────────┬──────────┐
│  Model  │  Task  │ Exemplars  │ Logging  │ Advanced │
└─────────┴────────┴────────────┴──────────┴──────────┘
```

Implemented via:
```python
tabs = types.TabsView()
tabs.add_choice("model", label="Model")
tabs.add_choice("task", label="Task")
tabs.add_choice("exemplars", label="Exemplars")
tabs.add_choice("advanced", label="Advanced")

inputs.enum("active_tab", tabs.values(), default="model", view=tabs)
```

Tab selection is stored in `ctx.params["active_tab"]` and used for
conditional rendering of each tab's fields.

### 4.2 Tab 1: Model

Contents (from current form):
- Config mode radio (Manual / JSON / Reset)
- Model name text input
- Base URL (optional)
- Cost preview (dynamic, reflects task + exemplar selections)

### 4.3 Tab 2: Task

Contents (from current form):
- Task selector dropdown
- Task-specific fields (classes, question, custom prompt)
- Class source selector (from field / custom list / open-ended)
- Output field name + overwrite toggle

### 4.4 Tab 3: Exemplars (NEW)

**When exemplars are disabled (default):**
- Enable toggle (OFF)
- Notice: "Provide labeled samples as few-shot examples to guide model
  output quality. Toggle on to configure."

**When exemplars are enabled:**
- Enable toggle (ON)
- **Exemplar source** — RadioGroup with 4 options:
  - "Saved view" → dropdown of `dataset.list_saved_views()`
  - "Sample IDs" → text input for comma-separated sample IDs
  - "Tag" → text input for tag name
  - "Field" → dropdown of dataset fields + value input (for boolean
    fields, the value input is auto-hidden and matches `True`; for
    string/enum fields, user provides the match value)
- **Exemplar label field** — dropdown listing:
  - All label fields (Classification, Classifications, Detections)
  - String fields (for caption/ocr/vqa tasks)
  - Filtered to show only fields compatible with the selected task
- **Exemplar preview** section:
  - Count of resolved exemplar samples
  - Per-exemplar token estimate
  - Total exemplar token overhead per inference call
  - Estimated exemplar cost (separate from inference cost)
  - Warning if exemplar count × image tokens is large
  - Tip: "Exemplar images are included in every inference call. Each
    exemplar adds ~N tokens to every request."

**Radio selection behavior:** Only the fields for the **active** radio
selection are used during execution. If a user enters sample IDs, then
switches to "Saved view", the sample IDs remain in the form (for user
convenience if they switch back) but are **ignored** — only the saved view
is resolved. The `resolve_exemplars()` function dispatches solely on the
`source` parameter value, never reading params for inactive sources.

### 4.5 Tab 4: Advanced

Contents (from current form):
- Temperature, max_tokens, top_p, seed
- Batch size, max concurrent, max workers
- Image detail level
- Coordinate format, box format
- Timeout
- System prompt override (with note about auto-prepended preamble)

### 4.6 Logging (Always Visible, Outside Tabs)

Logging controls are placed **outside** the tab-conditional blocks (alongside
the cost summary) so they are always visible regardless of which tab is active.
This avoids burying an important operational toggle in the Advanced tab.

- **Enable logging** toggle (prominent, always visible)
- When enabled: log level dropdown + log file path input
- Placed above the cost summary, below the tabs

### 4.7 Pre-Execute Cost Summary

A cost summary is rendered **outside** the tab-conditional blocks, at the
bottom of the form (always visible regardless of active tab). This is the
last thing the user sees before the execute button:

```
┌─────────────────────────────────────────────────┐
│  Estimated Cost Summary                         │
│                                                 │
│  Inference:  $X.XX  (N samples × $Y.YY/sample)│
│  Exemplars:  $Z.ZZ  (M exemplars × N samples)  │
│  ─────────────────────────────────────────────  │
│  Total:      $T.TT                              │
│                                                 │
│  ⚠ 5 exemplars add ~4,000 tokens per call      │
└─────────────────────────────────────────────────┘
                                        [Execute]
```

When exemplars are disabled, the exemplar line is omitted and the display
matches the current behavior.

The cost summary uses `Notice` (< $5 total) or `Warning` (>= $5 total).
The $5 warning threshold is configurable via the `FIFTYONE_OPENAI_COST_WARN`
environment variable (parsed as float, defaults to `5.0`). This allows teams
to set their own cost tolerance.

---

## 5. Cost Estimation

### 5.1 Per-Call Token Breakdown

```
base_input_tokens  = PROMPT_TEXT_TOKENS + target_image_tokens
exemplar_tokens    = N_exemplars × (image_tokens + EXEMPLAR_TEXT_TOKENS)
total_input_tokens = base_input_tokens + exemplar_tokens
output_tokens      = OUTPUT_TOKEN_ESTIMATES[task]  (unchanged)
```

Where:
- `PROMPT_TEXT_TOKENS = 50` (existing constant, may increase slightly with
  few-shot preamble — bump to ~70 when exemplars active)
- `image_tokens` = `IMAGE_TOKEN_COUNTS[image_detail]` (85/765/765)
- `EXEMPLAR_TEXT_TOKENS ≈ 30` (framing text + serialized JSON per exemplar)

### 5.2 Total Cost Formula

```
per_sample_cost = (total_input_tokens × input_cpt) + (output_tokens × output_cpt)
total_cost = per_sample_cost × num_samples
exemplar_overhead = (exemplar_tokens × input_cpt) × num_samples
```

### 5.3 Display

**Exemplars tab:** contextual estimate showing per-call overhead.
**Pre-execute summary:** full breakdown (inference + exemplar + total).
Both are dynamically updated as exemplar count and task change.

---

## 6. Configuration Persistence

### 6.1 New Persistable Keys

Added to `utils.py` `PERSISTABLE_KEYS`:

| Key | Type | Default |
|-----|------|---------|
| `exemplars_enabled` | `bool` | `False` |
| `exemplar_source` | `str` | `"saved_view"` |
| `exemplar_view_name` | `str` | `""` |
| `exemplar_sample_ids` | `str` | `""` |
| `exemplar_tag` | `str` | `""` |
| `exemplar_field_name` | `str` | `""` |
| `exemplar_field_value` | `str` | `""` |
| `exemplar_label_field` | `str` | `""` |

### 6.2 Persistence Behavior

- Follows existing three-level merge: defaults → global → dataset-specific
- Exemplar settings are dataset-specific by nature (saved views, tags, fields
  are per-dataset), so dataset-level persistence is most relevant
- JSON config export includes exemplar settings when enabled
- JSON config import restores exemplar settings (validated against current
  dataset — warns if saved view or field doesn't exist)

---

## 7. Execution Flow Changes

### 7.1 In `operators.py` `execute()`

After existing configuration phase, before the batch loop:

```python
# Resolve exemplars (if enabled)
exemplar_messages = None
if params.get("exemplars_enabled"):
    from .exemplars import resolve_exemplars, build_exemplar_messages

    exemplar_view = resolve_exemplars(
        dataset=ctx.dataset,
        source=params["exemplar_source"],
        view_name=params.get("exemplar_view_name"),
        sample_ids=params.get("exemplar_sample_ids"),
        tag=params.get("exemplar_tag"),
        field_name=params.get("exemplar_field_name"),
        field_value=params.get("exemplar_field_value"),
    )

    exemplar_messages = build_exemplar_messages(
        exemplar_view=exemplar_view,
        label_field=params["exemplar_label_field"],
        task_config=task,
        image_detail=params.get("image_detail", "auto"),
        max_workers=params.get("max_workers", 4),
    )
```

Then in the batch loop, pass to `build_messages()`:

```python
messages = task.build_messages(img, exemplar_messages=exemplar_messages)
```

Exemplar images are encoded **once** before the batch loop and reused across
all target samples (they're the same for every call).

### 7.2 Exemplar Exclusion from Inference Targets

If exemplar samples overlap with the target view (user selected "all samples"
but some are exemplars), they should still be processed as inference targets.
Exemplars are reference data, not exclusion criteria. The user controls what
gets processed via their view/selection.

---

## 8. Error Handling

**Core principle:** NEVER skip, squash, or swallow errors. Every error must
either be surfaced to the user with actionable information or allowed to
propagate. No silent failures. No `try/except` blocks without a meaningful
recovery strategy.

### 8.1 Resolution Errors (halt execution, display error)

- Saved view not found → `Error` in UI with message: "Saved view '{name}'
  does not exist. Available views: {list}." Halt.
- No samples match tag/field → `Error` in UI: "No samples found matching
  tag '{tag}' (or field '{field}={value}'). Verify the tag/field exists
  and has matching samples." Halt.
- Invalid sample IDs → `Error` in UI: "Sample IDs not found in dataset:
  {invalid_ids}." Halt. Do NOT silently filter to valid IDs.

### 8.2 Serialization Errors (halt execution, display error)

- Field doesn't exist on exemplar sample → `Error`: "Exemplar sample
  {id} does not have field '{field}'. Ensure all exemplar samples have
  the specified label field populated." Halt.
- Field type incompatible with task → `Error` in UI during form validation:
  "Field '{field}' is type {type}, incompatible with task '{task}'. See
  compatible types in the Exemplars tab." Halt.
- Empty/None field value on exemplar → `Error`: "Exemplar sample {id}
  has empty value for field '{field}'. All exemplar samples must have
  the label field populated." Halt.

### 8.3 Encoding Errors (halt execution, display error)

- Exemplar image file not found → `Error`: "Exemplar sample {id}: file
  not found at '{filepath}'. Verify the file exists." Halt.
- Exemplar image encoding failure → `Error`: "Failed to encode exemplar
  image for sample {id}: {error}. Check file format and permissions." Halt.
- NEVER skip exemplar samples. Partial exemplar sets produce inconsistent
  results and waste tokens. All configured exemplars must succeed or the
  entire run must halt with a clear error.

---

## 9. Validation Rules

### 9.1 Field-Task Compatibility

| Task | Compatible Field Types |
|------|----------------------|
| classify | Classification, string |
| tag | Classifications, Classification (single → list), string (comma-split) |
| caption | string, Classification.label |
| detect | Detections |
| vqa | string, Classification.label |
| ocr | string, Classification.label |

The Exemplars tab dynamically filters the label field dropdown to show only
compatible fields for the selected task.

### 9.2 Cross-Tab Validation

- If task changes and the selected exemplar label field becomes incompatible,
  show a `Warning` on the Exemplars tab
- If exemplars are enabled but no source is configured, show `Error`
- If exemplars are enabled but label field is not selected, show `Error`

---

## 10. Logging

### 10.1 New Log Messages

- **INFO:** "Resolved N exemplar samples from {source}"
- **INFO:** "Built N exemplar message pairs (M tokens estimated)"
- **INFO:** "Exemplar overhead: ~N tokens per inference call"
- **DEBUG:** per-exemplar serialization details (sample ID, field value,
  serialized JSON)
- **DEBUG:** per-exemplar image encoding details (filepath, encoding method)

### 10.2 Run Summary

`dataset.info["openai_runs"][field_name]` gains:
- `exemplars_enabled: bool`
- `exemplar_count: int`
- `exemplar_source: str`
- `exemplar_label_field: str`
- `exemplar_token_overhead: int` (estimated total extra tokens)

---

## 11. Files Summary

| File | Action | Estimated Lines Changed |
|------|--------|------------------------|
| `exemplars.py` | NEW | ~200 lines |
| `tasks.py` | MODIFY | ~30 lines (build_messages extension, preamble constant) |
| `operators.py` | MODIFY | ~250 lines (tab restructuring, exemplar tab UI, cost updates, execute changes) |
| `utils.py` | MODIFY | ~10 lines (new persistable keys) |
| `fiftyone.yml` | NO CHANGE | — |
| `docs/` | MODIFY | New exemplar docs, update pipeline/config/cost docs |

---

## 12. Verification

### 12.1 Unit Tests

Tests use real FiftyOne datasets (in-memory or temporary) — no mocking
unless absolutely impossible to avoid.

- `exemplars.py`: test each resolution strategy with real FiftyOne datasets
  containing labeled samples
- `exemplars.py`: test serialization for all 6 task types using real
  FiftyOne label objects (Classification, Detections, etc.)
- `exemplars.py`: test message building produces correct message structure
  (role, content format, alternating user/assistant)
- `tasks.py`: test `build_messages()` with and without exemplar messages
- `exemplars.py`: test error cases raise with actionable messages (missing
  field, empty field, invalid sample IDs, missing saved view)

### 12.2 Integration Tests

- End-to-end: create dataset with labeled samples, configure exemplars,
  run inference, verify messages sent to API contain exemplar pairs
- Cost estimation: verify exemplar overhead is correctly calculated
- Persistence: save config with exemplars, reload, verify settings restored

### 12.3 Manual Testing

- Open operator in FiftyOne App
- Verify 4-tab layout renders correctly
- Test each exemplar source type
- Verify cost summary updates dynamically
- Run inference with exemplars enabled, check results quality
- Run inference with exemplars disabled, confirm no behavior change
- Test with large exemplar count, verify warning appears

---

## 13. Implementation Guidelines

These apply to all code written for this feature:

1. **No silent errors.** Never use `try/except` without a meaningful recovery
   strategy. If there is no fallback, let the error propagate. Every error
   must surface to the user with actionable information.

2. **No mocking in unit tests.** Use real FiftyOne datasets and label objects.
   Mocking is only acceptable when testing against external APIs in
   integration tests and even then should be minimized.

3. **Docstrings are mandatory** when a function has more than two non-obvious
   arguments. All arguments must be documented. "Obvious" means the name
   alone (e.g., `text_str`, `integers`) fully describes it.

4. **Use `model_dump_json()`** for serializing Pydantic models to JSON
   strings. Do not use `json.dumps(model.model_dump())`.

5. **Use for-loops for UI elements** where possible (e.g., adding choices
   to a RadioGroup/Dropdown from a list) instead of repetitive per-line calls.

6. **Active radio selection only.** When resolving exemplar source, dispatch
   solely on the active radio value. Never read params for inactive sources.
   Inactive source values are retained in the form for user convenience but
   ignored during execution.

7. **Logging throughout.** All new functions must use the `fo_openai` logger.
   Log at INFO for high-level operations, DEBUG for per-sample details.

8. **Low complexity, concise, optimized.** Avoid unnecessary abstractions.
   Prefer direct, readable code. Minimize nesting depth.

9. **Coordinate conversion only when needed.** Default format (normalized
   [0,1] xyxy) matches FiftyOne's native format — skip conversion in the
   default case.

10. **Cost warning threshold is configurable.** Read from
    `FIFTYONE_OPENAI_COST_WARN` environment variable (default: `5.0`).
