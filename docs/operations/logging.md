# Logging

Opt-in, plugin-scoped logging under the `fo_openai` namespace.

## Enabling

Toggle "Enable logging" in the **Logging** tab of the operator form. When enabled, two additional controls appear:
- **Log Level** — `INFO` (default), `DEBUG`, or `WARNING`
- **Log File** — optional file path; leave empty for stderr only

All three settings persist across app restarts via the config persistence system.

## Log Levels

| Level | What's logged |
|-------|--------------|
| `INFO` | Run start (model, samples, batch_size, field), batch completion counts, run completion summary |
| `DEBUG` | All of INFO plus per-batch image encoding and request counts |
| `WARNING` | Per-sample errors only (always emitted even when logging is disabled, since WARNING is the floor) |

## Log File Path Resolution

The `_resolve_log_path()` function in `_log.py` handles three cases:

| Input | Behavior | Example |
|-------|----------|---------|
| Directory (or ends with `/`) | ISO-timestamped filename | `~/logs/` → `~/logs/2026-03-31T14-59-11.log` |
| Existing file | Auto-increment suffix | `run.log` → `run2.log` → `run3.log` |
| New file path | Used as-is | `~/my-project/inference.log` |

Parent directories are created automatically.

## Error Classification

Per-sample errors are prefixed to distinguish their origin:

| Prefix | Source | Example |
|--------|--------|---------|
| `[API]` | Exception from `client.responses.parse()` | Rate limit, timeout, content policy, bad parameter |
| `[Parse]` | Exception from `task.parse_response()` | Coordinate conversion failure, unexpected model structure |

## Run Summary

Always written to `dataset.info["openai_runs"][field_name]["summary"]`, regardless of logging settings:

```json
{
  "total": 200,
  "succeeded": 176,
  "api_errors": 20,
  "parse_errors": 4,
  "first_errors": [
    {"id": "abc123", "stage": "api", "error": "[API] BadRequestError: ..."},
    {"id": "def456", "stage": "parse", "error": "[Parse] ValueError: ..."}
  ]
}
```

The `first_errors` list captures up to 10 samples for post-mortem diagnosis. This data survives even if the app crashes, since it's persisted to MongoDB via `dataset.info`.

## Logger Isolation

All loggers are children of `fo_openai` (e.g., `fo_openai.operators`, `fo_openai.engine`). `root.propagate = False` ensures no output leaks into FiftyOne's own logging. Prior-run handlers are cleaned up on each new run to prevent duplication.
