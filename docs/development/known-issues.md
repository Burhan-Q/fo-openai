# Known Issues

## FiftyOne UI Constraints

### `TextFieldView(multiline=True)` not honored
The FiftyOne frontend does not render `TextFieldView` as a multi-line text area even when `multiline=True` is passed. The kwarg is serialized to JSON correctly but the React component ignores it. Long prompts scroll horizontally in a single-line field.

### `TextView` is display-only
`types.TextView` renders static text, not an input field. It cannot be used for user input despite its name suggesting otherwise. Use `types.TextFieldView` for text inputs.

### App may become non-responsive during long inference runs
When running inference on many samples in immediate (non-delegated) mode, the FiftyOne App UI can become unresponsive. This appears to be caused by rapid `set_progress` triggers and/or large numbers of `set_values` writes. The GraphQL server (strawberry) may throw errors during this state. **Mitigation:** Use delegated execution for datasets larger than ~50 samples.

## Detection Coordinates

### General-purpose LLMs have limited detection accuracy
Base (non-fine-tuned) OpenAI models like GPT-4o, GPT-4.1, and GPT-5 are
not architecturally designed for precise spatial localization. On detection
benchmarks (RF100-VL, CVPR 2025), GPT-5 scored mAP50:95 of only 1.5.
Bounding box positions may be offset, especially on non-square images.
For precision-critical detection, consider fine-tuned models or dedicated
detectors (YOLO, GroundingDINO) instead.

### Pixel coordinates are the default (with image dimensions in prompt)
The default coordinate format is ``pixel`` — integer pixel coordinates
relative to the original image dimensions, which are included in every
detection prompt ("Original image: WxH pixels (landscape)"). This gives
the model concrete grounding to the actual image dimensions and produces
integer values that align well with LLM tokenizers. Requires
``compute_metadata()`` (reads file headers only, fast).

### Integer coordinates preferred over 0-1 floats
Research (arXiv:2406.13208) and industry practice (Gemini, Qwen-VL, OpenAI
fine-tuning) show that integer coordinates produce more accurate bounding
boxes from LLMs. Integers align with tokenizer distributions — a float
like ``0.3741`` costs multiple tokens and is rare in training data, while
``374`` is 1-2 tokens and common. Both ``pixel`` and ``normalized_1000``
use integers; ``normalized_1`` (0-1 floats) remains available but is
expected to be less accurate.

## Async Event Loop

### FiftyOne App event loop conflict
FiftyOne's Uvicorn server runs its own asyncio event loop. Calling `asyncio.run()` from within operator execution raises `RuntimeError`. The `_run_async()` helper in `engine.py` detects this and runs the coroutine in a dedicated `ThreadPoolExecutor` thread. This works but adds thread overhead.

## Cost Estimation

### Pricing data may be stale
The LiteLLM pricing JSON is cached for 24 hours. New models released within that window won't have pricing data available until the cache refreshes. The cost preview shows "unavailable" in this case.

### Token estimates are heuristic
Image token counts are estimates (85 for low detail, 765 for high/auto). Actual token usage depends on image dimensions and OpenAI's internal tiling algorithm. Output token estimates are rough heuristics per task type. Cost previews should be treated as approximations.
