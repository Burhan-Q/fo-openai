# Known Issues

## FiftyOne UI Constraints

### `TextFieldView(multiline=True)` not honored
The FiftyOne frontend does not render `TextFieldView` as a multi-line text area even when `multiline=True` is passed. The kwarg is serialized to JSON correctly but the React component ignores it. Long prompts scroll horizontally in a single-line field.

### `TextView` is display-only
`types.TextView` renders static text, not an input field. It cannot be used for user input despite its name suggesting otherwise. Use `types.TextFieldView` for text inputs.

### App may become non-responsive during long inference runs
When running inference on many samples in immediate (non-delegated) mode, the FiftyOne App UI can become unresponsive. This appears to be caused by rapid `set_progress` triggers and/or large numbers of `set_values` writes. The GraphQL server (strawberry) may throw errors during this state. **Mitigation:** Use delegated execution for datasets larger than ~50 samples.

## Detection Coordinates

### Pixel coordinates unreliable with OpenAI models
OpenAI resizes images internally before vision processing. Returned bounding box coordinates are relative to the internal resolution, which is undocumented and may change. The `pixel` coordinate format is retained in code but removed from the UI dropdown. Only `normalized_1` (0-1) and `normalized_1000` (0-1000) are exposed.

### 0-1000 normalized coordinates untested
The `normalized_1000` coordinate format is available in the UI but has not been validated with OpenAI models. It may or may not produce accurate results depending on how the model interprets the coordinate instruction in the system prompt.

## Async Event Loop

### FiftyOne App event loop conflict
FiftyOne's Uvicorn server runs its own asyncio event loop. Calling `asyncio.run()` from within operator execution raises `RuntimeError`. The `_run_async()` helper in `engine.py` detects this and runs the coroutine in a dedicated `ThreadPoolExecutor` thread. This works but adds thread overhead.

## Cost Estimation

### Pricing data may be stale
The LiteLLM pricing JSON is cached for 24 hours. New models released within that window won't have pricing data available until the cache refreshes. The cost preview shows "unavailable" in this case.

### Token estimates are heuristic
Image token counts are estimates (85 for low detail, 765 for high/auto). Actual token usage depends on image dimensions and OpenAI's internal tiling algorithm. Output token estimates are rough heuristics per task type. Cost previews should be treated as approximations.
