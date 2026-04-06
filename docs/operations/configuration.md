# Configuration

## Three Configuration Modes

The operator form offers a radio selector:

1. **Configure manually** — interactive form with all fields
2. **Paste JSON config** — import a JSON config exported from a prior run
3. **Reset to defaults** — clears all stored settings (global + dataset)

## Persistence (2-tier)

Settings persist across app restarts at two levels:

| Tier | Storage | Scope |
|------|---------|-------|
| Global | `ExecutionStore("openai_config")` | Cross-dataset |
| Dataset | `dataset.info["_openai_config"]` | Per-dataset |

**Merge order:** hardcoded `_DEFAULTS` → global config → dataset config → form params.

The `api_key` is excluded from dataset-level persistence for security.

## Secrets

Declared in `fiftyone.yml`:
```yaml
secrets:
  - FIFTYONE_OPENAI_API_KEY
  - OPENAI_API_KEY
```

FiftyOne's `SecretsDictionary` reads these from environment variables automatically. Behavior:
- Env var set → returns the value string
- Env var not set → returns `""` (empty string, not `None`)
- Key not declared in `fiftyone.yml` → returns `None`

**API key resolution order in `_create_engine`:**
1. `params["api_key"]` (from advanced settings form field)
2. `ctx.secrets["FIFTYONE_OPENAI_API_KEY"]`
3. `ctx.secrets["OPENAI_API_KEY"]`
4. Raise `ValueError` if all empty

## Persisted Keys

All keys in `utils._PERSIST_KEYS`:

```
model, base_url, api_key, task, classes, question, prompt,
system_prompt, prompt_override, temperature, max_output_tokens,
top_p, batch_size, max_concurrent, max_workers, timeout,
image_detail, coordinate_format, box_format, log_metadata,
enable_logging, log_level, log_file, exemplars_enabled,
exemplar_source, exemplar_view_name, exemplar_sample_ids,
exemplar_tag, exemplar_field_name, exemplar_field_value,
exemplar_label_field
```

## SDK Parameter Handling

Only user-explicitly-set values are sent to the OpenAI API. Parameters left empty in the form are omitted entirely from `completion_kwargs`, letting the SDK/model use its own defaults. This prevents errors like `BadRequestError: Unsupported parameter: 'max_tokens'` on newer models.

See [Design Decisions](../development/decisions.md) for rationale.

## JSON Config Import/Export

After a run, `resolve_output` displays an exportable JSON config (with `api_key` excluded). This can be pasted into the "Paste JSON config" mode for reproducible runs. Only keys in `_PERSIST_KEYS` are accepted; unknown keys are silently dropped.
