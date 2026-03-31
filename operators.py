"""FiftyOne operator for OpenAI inference: UI, batching, progress, and result
storage."""

from __future__ import annotations

import json
from typing import Any, Generator

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from ._log import configure as configure_logging
from ._log import get_logger, summarise_errors
from ._pricing import estimate_cost, get_model_info
from .engine import OpenAIEngine
from .tasks import (
    IMAGE_TOKEN_COUNTS,
    OUTPUT_TOKEN_ESTIMATES,
    PROMPT_TEXT_TOKENS,
    TaskConfig,
)
from .utils import (
    build_image_contents,
    clear_global_config,
    get_global_config,
    normalize_classes,
    parse_config_json,
    pick_params,
    save_global_config,
)

logger = get_logger(__name__)

_DEFAULTS: dict[str, Any] = {
    "batch_size": 8,
    "max_concurrent": 16,
    "max_workers": 4,
    "coordinate_format": "normalized_1",
    "box_format": "xyxy",
    "image_detail": "auto",
}


class OpenAIInference(foo.Operator):
    """Send images to OpenAI vision models for labeling via LiteLLM."""

    @property
    def config(self) -> foo.OperatorConfig:
        """Return the operator configuration."""
        return foo.OperatorConfig(
            name="run_openai_inference",
            label="Run OpenAI Inference",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx: Any) -> types.Property:
        """Build the dynamic input form shown to the user."""
        inputs = types.Object()

        if not ctx.dataset:
            inputs.view("error", types.Error(label="No dataset loaded"))
            return types.Property(inputs)

        stored = _resolve_config(ctx)

        # Config mode selector
        mode_radio = types.RadioGroup(orientation="horizontal")
        mode_radio.add_choice("manual", label="Configure manually")
        mode_radio.add_choice("json", label="Paste JSON config")
        mode_radio.add_choice("reset", label="Reset to defaults")
        inputs.enum(
            "config_mode",
            mode_radio.values(),
            default="manual",
            label="Configuration",
            view=mode_radio,
        )
        config_mode: str = ctx.params.get("config_mode", "manual")

        if config_mode == "json":
            _json_config_mode(ctx, inputs)
        elif config_mode == "reset":
            inputs.view(
                "reset_notice",
                types.Notice(
                    label=(
                        "All stored settings (global and dataset) will be"
                        " cleared and defaults restored."
                    )
                ),
            )
        else:
            _model_selector(ctx, inputs, stored)
            task = _task_selector(ctx, inputs, stored)
            _task_settings(ctx, inputs, task, stored)
            _cost_preview(ctx, inputs, task)
            _output_settings(ctx, inputs, task, stored)
            _advanced_settings(ctx, inputs, stored)

        if config_mode != "reset":
            inputs.view_target(ctx)

        return types.Property(
            inputs, view=types.View(label="OpenAI Inference")
        )

    def resolve_delegation(self, ctx: Any) -> bool | None:
        """Let the user choose immediate vs. delegated execution."""
        return ctx.params.get("delegate", None)

    def execute(self, ctx: Any) -> Generator[Any, None, None]:
        """Run inference over the target view, yielding progress."""
        params: dict[str, Any] = ctx.params
        config_mode: str = params.get("config_mode", "manual")

        # Handle reset mode
        if config_mode == "reset":
            clear_global_config()
            ctx.dataset.info.pop("_openai_config", None)
            ctx.dataset.save()
            if not ctx.delegated:
                yield ctx.trigger("reload_dataset")
            return

        # Handle JSON paste mode
        if config_mode == "json":
            raw = params.get("config_json") or ""
            cfg, err = parse_config_json(raw)
            if err:
                yield _error(ctx, f"Config import failed: {err}")
                return
            params.update(cfg)
            if not params.get("model") or not params.get("task"):
                yield _error(
                    ctx, "Config missing required 'model' or 'task'"
                )
                return

        # -- Configure plugin logging --
        configure_logging(
            enabled=params.get("enable_logging", False),
            level=params.get("log_level", "INFO"),
            log_file=params.get("log_file", ""),
        )

        # Resolve classes from field picker if applicable
        _resolve_classes_from_field(ctx, params)

        try:
            engine, api_key = _create_engine(params, ctx.secrets)
            task = _create_task(params)
        except Exception as e:
            yield _error(ctx, str(e))
            return

        batch_size: int = params.get("batch_size", _DEFAULTS["batch_size"])
        image_detail: str = params.get(
            "image_detail", _DEFAULTS["image_detail"]
        )

        # Get target samples and resolve output field
        view = ctx.target_view()
        ids: list[str] = view.values("id")
        filepaths: list[str] = view.values("filepath")
        total: int = len(ids)
        max_workers: int = params.get("max_workers", _DEFAULTS["max_workers"])

        response_model = task.get_response_model()

        field_name = _resolve_field_name(
            ctx.dataset, params["task"], params.get("overwrite_last", False)
        )

        logger.info(
            "Starting %s inference: model=%s, samples=%d, "
            "batch_size=%d, field=%s",
            params["task"],
            params["model"],
            total,
            batch_size,
            field_name,
        )

        # Build metadata for optional per-label logging
        log_metadata: bool = params.get("log_metadata", False)
        full_prompt = ""
        infer_cfg: dict[str, Any] = {}
        if log_metadata:
            if task.system_prompt:
                full_prompt += f"[system] {task.system_prompt}\n"
            full_prompt += f"[user] {task.prompt}"
            infer_cfg = {
                **engine.completion_kwargs,
                "batch_size": batch_size,
                "coordinate_format": task.coordinate_format,
                "box_format": task.box_format,
                "max_concurrent": engine.max_concurrent,
            }

        # Clear stale error fields when overwriting
        if params.get("overwrite_last", False):
            error_field = f"{field_name}_error"
            if error_field in ctx.dataset.get_field_schema(flat=True):
                ctx.dataset.set_values(
                    error_field,
                    {sid: None for sid in ids},
                    key_field="id",
                )

        # Collect image dimensions for pixel coordinate normalisation
        need_dims = (
            task.task == "detect" and task.coordinate_format == "pixel"
        )
        if need_dims:
            view.compute_metadata()
            widths: list[float | None] = view.values("metadata.width")
            heights: list[float | None] = view.values("metadata.height")
        else:
            widths = [None] * total
            heights = [None] * total

        # Process in batches
        processed = 0
        api_errors = 0
        parse_errors = 0
        # Collect first N error samples for the run summary
        error_samples: list[dict[str, str]] = []
        max_error_samples = 10

        for i in range(0, total, batch_size):
            end = i + batch_size
            batch_ids = ids[i:end]
            batch_paths = filepaths[i:end]
            batch_widths = widths[i:end]
            batch_heights = heights[i:end]
            batch_num = i // batch_size + 1

            logger.debug("Batch %d: encoding %d images", batch_num, len(batch_ids))
            image_contents = build_image_contents(
                batch_paths,
                max_workers=max_workers,
                image_detail=image_detail,
            )
            batch_messages = [
                task.build_messages(img) for img in image_contents
            ]
            logger.debug("Batch %d: sending %d requests", batch_num, len(batch_messages))
            responses = engine.infer_batch(
                batch_messages, response_model=response_model
            )

            results: dict[str, Any] = {}
            errors: dict[str, str] = {}
            for sid, resp, img_w, img_h in zip(
                batch_ids, responses, batch_widths, batch_heights
            ):
                if isinstance(resp, Exception):
                    err_msg = f"[API] {type(resp).__name__}: {resp}"
                    errors[sid] = err_msg
                    api_errors += 1
                    logger.warning("Sample %s API error: %s", sid, err_msg)
                    if len(error_samples) < max_error_samples:
                        error_samples.append(
                            {"id": sid, "stage": "api", "error": err_msg}
                        )
                    continue
                try:
                    label = task.parse_response(
                        resp, image_width=img_w, image_height=img_h
                    )
                    if log_metadata:
                        label.model_name = params["model"]
                        label.prompt = full_prompt
                        label.infer_cfg = infer_cfg
                    results[sid] = label
                except Exception as e:
                    err_msg = f"[Parse] {type(e).__name__}: {e}"
                    errors[sid] = err_msg
                    parse_errors += 1
                    logger.warning("Sample %s parse error: %s", sid, err_msg)
                    if len(error_samples) < max_error_samples:
                        error_samples.append(
                            {"id": sid, "stage": "parse", "error": err_msg}
                        )

            _write_batch_results(ctx.dataset, field_name, results, errors)

            processed += len(batch_ids)
            total_err = api_errors + parse_errors
            progress_label = f"{processed}/{total} samples"
            if total_err:
                progress_label += f" ({total_err} errors)"

            logger.info(
                "Batch %d complete: %d ok, %d errors",
                batch_num,
                len(results),
                len(errors),
            )

            if ctx.delegated:
                ctx.set_progress(
                    progress=processed / total, label=progress_label
                )
            else:
                yield ctx.trigger(
                    "set_progress",
                    dict(progress=processed / total, label=progress_label),
                )

        # -- Run summary (always written) --
        total_err = api_errors + parse_errors
        summary = summarise_errors(
            api_errors=api_errors,
            parse_errors=parse_errors,
            total=total,
            error_samples=error_samples,
        )
        runs: dict[str, Any] = ctx.dataset.info.get("openai_runs", {})
        run_entry: dict[str, Any] = {"summary": summary}
        if log_metadata:
            run_entry["model_name"] = params["model"]
            run_entry["prompt"] = full_prompt
            run_entry["infer_cfg"] = infer_cfg
        runs[field_name] = run_entry
        ctx.dataset.info["openai_runs"] = runs

        logger.info(
            "Run complete: %d/%d succeeded, %d API errors, %d parse errors",
            total - total_err,
            total,
            api_errors,
            parse_errors,
        )

        save_global_config(params)
        ctx.dataset.info["_openai_config"] = pick_params(
            params, exclude=("api_key",)
        )
        ctx.dataset.save()

        if ctx.delegated:
            ctx.store("openai_status").set("done", True)
        else:
            yield ctx.trigger("reload_dataset")

    def resolve_output(self, ctx: Any) -> types.Property:
        """Display an exportable JSON config after execution."""
        outputs = types.Object()
        outputs.str("summary", label="Summary")

        if ctx.params.get("config_mode") != "reset":
            cfg_json = json.dumps(
                pick_params(ctx.params, exclude=("api_key",)), indent=2
            )
            outputs.str(
                "config_export",
                label="Exportable Config (copy to reuse)",
                default=cfg_json,
                view=types.CodeView(language="json", read_only=True),
            )

        return types.Property(outputs, view=types.View(label="Complete"))


class CheckOpenAIStatus(foo.Operator):
    """Auto-subscribe to delegated-job completion via MongoDB change-stream.

    Fires a toast and reloads the dataset when the worker signals done.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        """Return the operator configuration."""
        return foo.OperatorConfig(
            name="check_openai_status",
            label="Check OpenAI Status",
            on_dataset_open=True,
            execute_as_generator=True,
            unlisted=True,
        )

    async def execute(self, ctx: Any) -> Any:
        """Wait for a completion signal, then notify and reload."""
        import asyncio

        from fiftyone.operators.store.notification_service import (
            default_notification_service,
        )

        if not ctx.dataset:
            return

        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        def _on_change(_message: Any) -> None:
            """Set the event from the notification callback thread."""
            loop.call_soon_threadsafe(event.set)

        sub_id = default_notification_service.subscribe(
            "openai_status",
            callback=_on_change,
            dataset_id=str(ctx.dataset._doc.id),
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=600)
            ctx.store("openai_status").delete("done")
            yield ctx.trigger(
                "notify",
                params={
                    "message": "OpenAI inference complete",
                    "variant": "success",
                },
            )
            yield ctx.trigger("reload_dataset")
        except asyncio.TimeoutError:
            pass
        finally:
            default_notification_service.unsubscribe(sub_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(ctx: Any, message: str) -> Any:
    """Yield-safe error via ``set_progress``."""
    label = f"Error: {message}"
    if ctx.delegated:
        ctx.set_progress(progress=0, label=label)
        return None
    return ctx.trigger("set_progress", {"progress": 0, "label": label})


def _fmt_usd(value: float) -> str:
    """Format a dollar amount with dynamic precision.

    Uses just enough decimal places to show two significant digits in
    the fractional part, clamped to 2-6 decimals.  For example::

        0.0000012  -> "$0.0000012"
        0.0015     -> "$0.0015"
        0.19       -> "$0.19"
        3.5        -> "$3.50"
    """
    if value < 0.01:
        if value == 0:
            return "$0.00"
        # Show enough decimals to reveal at least 2 significant figures
        # e.g. 0.00097 -> 5 decimals, 0.0000012 -> 7 decimals
        s = f"{value:.10f}"
        dot = s.index(".")
        first_sig = next(
            (i for i in range(dot + 1, len(s)) if s[i] != "0"), len(s) - 1
        )
        decimals = max(first_sig - dot + 1, 4)
        return f"${value:.{decimals}f}"
    if value < 1.0:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _resolve_config(ctx: Any) -> dict[str, Any]:
    """Merge dataset config > global config > ``_DEFAULTS``."""
    merged = dict(_DEFAULTS)
    for cfg in (
        get_global_config(),
        ctx.dataset.info.get("_openai_config") or {},
    ):
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _create_engine(
    params: dict[str, Any], secrets: Any
) -> tuple[OpenAIEngine, str]:
    """Build an ``OpenAIEngine`` from operator params and secrets.

    Resolution order for the API key:

    1. Explicit ``api_key`` in operator params (advanced settings).
    2. ``FIFTYONE_OPENAI_API_KEY`` from FiftyOne plugin secrets / env.
    3. ``OPENAI_API_KEY`` from FiftyOne plugin secrets / env.

    Both secret names are declared in ``fiftyone.yml`` so FiftyOne
    resolves them from environment variables automatically via
    ``ctx.secrets``.  An unset secret returns an empty string, so we
    treat empty strings as missing.

    Returns ``(engine, api_key)``.

    Raises:
        ValueError: If no API key can be resolved.
    """
    api_key: str = (
        params.get("api_key")
        or secrets["FIFTYONE_OPENAI_API_KEY"]
        or secrets["OPENAI_API_KEY"]
        or ""
    )
    if not api_key:
        raise ValueError(
            "No API key configured. Set FIFTYONE_OPENAI_API_KEY or"
            " OPENAI_API_KEY as an environment variable, or provide"
            " an API key in the advanced settings."
        )

    # Build completion kwargs — only include values the user explicitly set.
    # Omitted params let the OpenAI SDK / model use their own defaults,
    # avoiding errors like "Unsupported parameter: 'max_tokens'".
    completion_kwargs: dict[str, Any] = {}
    if params.get("temperature") is not None:
        completion_kwargs["temperature"] = params["temperature"]
    if params.get("max_completion_tokens") is not None:
        completion_kwargs["max_completion_tokens"] = params[
            "max_completion_tokens"
        ]
    if params.get("top_p") is not None:
        completion_kwargs["top_p"] = params["top_p"]
    if params.get("seed") is not None:
        completion_kwargs["seed"] = params["seed"]

    timeout_val = params.get("timeout")
    engine = OpenAIEngine(
        model=params["model"],
        api_key=api_key,
        base_url=params.get("base_url") or None,
        max_concurrent=params.get(
            "max_concurrent", _DEFAULTS["max_concurrent"]
        ),
        timeout=float(timeout_val) if timeout_val else None,
        **completion_kwargs,
    )
    return engine, api_key


def _create_task(params: dict[str, Any]) -> TaskConfig:
    """Build a ``TaskConfig`` from operator params."""
    return TaskConfig(
        task=params["task"],
        prompt=params.get("prompt_override") or params.get("prompt"),
        system_prompt=params.get("system_prompt"),
        classes=normalize_classes(params.get("classes")),
        coordinate_format=params.get(
            "coordinate_format", _DEFAULTS["coordinate_format"]
        ),
        box_format=params.get("box_format", _DEFAULTS["box_format"]),
        question=params.get("question", ""),
    )


def _resolve_classes_from_field(
    ctx: Any, params: dict[str, Any]
) -> None:
    """When the user chose *field* as the class source, extract unique
    labels from the selected dataset field and write them into *params*."""
    if params.get("class_source") != "field":
        return

    source_field: str | None = params.get("source_field")
    if not source_field or not ctx.dataset:
        return

    label_classes = _get_field_classes(ctx.dataset, source_field)
    if label_classes:
        params["classes"] = ", ".join(sorted(label_classes))


_LABEL_PATH_SUFFIXES: list[str] = [
    ".label",                    # Classification
    ".classifications.label",    # Classifications
    ".detections.label",         # Detections
]


def _get_field_classes(dataset: fo.Dataset, field_name: str) -> list[str]:
    """Extract unique class labels from a dataset label field.

    Handles ``Classification``, ``Classifications``, and ``Detections``
    fields by probing known sub-field paths.
    """
    schema = dataset.get_field_schema(flat=True)
    for suffix in _LABEL_PATH_SUFFIXES:
        path = f"{field_name}{suffix}"
        if path in schema:
            return dataset.distinct(path)
    return []


def _resolve_field_name(
    dataset: fo.Dataset, task_name: str, overwrite: bool = False
) -> str:
    """Resolve the output field name for a task run.

    Produces ``openai_infer_{task}``, ``openai_infer_{task}1``, etc.
    When *overwrite* is ``True`` the highest existing field is reused.
    """
    schema = dataset.get_field_schema(flat=True)
    base = f"openai_infer_{task_name}"

    if base not in schema:
        return base

    n = 1
    while f"{base}{n}" in schema:
        n += 1

    if overwrite:
        return f"{base}{n - 1}" if n > 1 else base
    return f"{base}{n}"


def _write_batch_results(
    dataset: fo.Dataset,
    field_name: str,
    results: dict[str, Any],
    errors: dict[str, str],
) -> None:
    """Bulk-write a batch of results and errors as flat sample fields."""
    if results:
        dataset.set_values(
            field_name, results, key_field="id", dynamic=True
        )
    if errors:
        dataset.set_values(
            f"{field_name}_error", errors, key_field="id", dynamic=True
        )


# ---------------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------------


def _model_selector(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add the model text-input field to the form."""
    inputs.str(
        "model",
        label="Model",
        required=True,
        default=stored.get("model", ""),
        description="LiteLLM model ID (e.g., gpt-4o, gpt-4o-mini)",
    )


def _task_selector(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> str | None:
    """Add the task dropdown and return the currently selected task."""
    task_dropdown = types.Dropdown(label="Task")
    task_dropdown.add_choice(
        "caption",
        label="Caption",
        description="Generate a text description of the image",
    )
    task_dropdown.add_choice(
        "classify",
        label="Classify",
        description="Assign a single class label",
    )
    task_dropdown.add_choice(
        "tag",
        label="Tag",
        description="Assign multiple labels",
    )
    task_dropdown.add_choice(
        "detect",
        label="Detect",
        description="Detect objects with bounding boxes",
    )
    task_dropdown.add_choice(
        "vqa",
        label="VQA",
        description="Answer a question about the image",
    )
    task_dropdown.add_choice(
        "ocr",
        label="OCR",
        description="Extract text visible in the image",
    )
    inputs.enum(
        "task",
        task_dropdown.values(),
        required=True,
        default=stored.get("task"),
        label="Task",
        view=task_dropdown,
    )
    return ctx.params.get("task", None)


def _json_config_mode(ctx: Any, inputs: types.Object) -> None:
    """Render the JSON-paste configuration sub-form."""
    inputs.str(
        "config_json",
        label="Paste JSON Config",
        required=True,
        description="Paste a config exported from a previous run",
        view=types.CodeView(language="json"),
    )
    inputs.bool(
        "show_params",
        label="Show accepted parameters",
        default=False,
        view=types.SwitchView(),
    )
    if ctx.params.get("show_params"):
        inputs.md(
            "**Model:** `model`, `base_url`\n\n"
            "**Task:** `task`, `classes`, `question`, `prompt`, "
            "`system_prompt`, `prompt_override`\n\n"
            "**Advanced:** `temperature`, `max_completion_tokens`, `top_p`, "
            "`seed`, `batch_size`, `max_concurrent`, `max_workers`, "
            "`image_detail`, `coordinate_format`, `box_format`",
            name="params_ref",
        )
    raw: str | None = ctx.params.get("config_json")
    if not raw:
        return

    cfg, err = parse_config_json(raw)
    if err:
        inputs.view("json_err", types.Error(label=err))
        return

    missing = [k for k in ("model", "task") if not cfg.get(k)]
    if cfg.get("task") == "vqa" and not cfg.get("question"):
        missing.append("question")

    if missing:
        inputs.view(
            "json_warn",
            types.Warning(
                label="Missing required: " + ", ".join(missing)
            ),
        )
    else:
        inputs.view(
            "json_ok",
            types.Notice(
                label=f"Valid: {cfg['task']} task with {cfg['model']}"
            ),
        )


def _task_settings(
    ctx: Any,
    inputs: types.Object,
    task: str | None,
    stored: dict[str, Any],
) -> None:
    """Add task-specific settings (classes, question, prompt override)."""
    if task is None:
        return

    inputs.view(
        "task_header",
        types.Header(label="Task Settings", divider=True),
    )

    if task in ("classify", "tag", "detect"):
        _class_source_selector(ctx, inputs, stored)

    if task == "vqa":
        inputs.str(
            "question",
            label="Question",
            required=True,
            default=stored.get("question", ""),
            description="Question to ask about each image",
        )

    inputs.bool(
        "show_prompt_override",
        label="Custom prompt",
        default=bool(stored.get("prompt_override")),
        view=types.SwitchView(),
    )
    if ctx.params.get("show_prompt_override", False):
        inputs.str(
            "prompt_override",
            label="Prompt Override",
            required=False,
            default=stored.get("prompt_override", ""),
            description="Override the default prompt for this task",
            view=types.TextFieldView(multiline=True),
        )


def _class_source_selector(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add the 3-source class-label input (dataset field / custom / open)."""
    source_radio = types.RadioGroup(orientation="horizontal")
    source_radio.add_choice("field", label="From dataset field")
    source_radio.add_choice("custom", label="Custom list")
    source_radio.add_choice("open", label="Open-ended")
    inputs.enum(
        "class_source",
        source_radio.values(),
        default="custom",
        label="Class Source",
        view=source_radio,
    )

    class_source: str = ctx.params.get("class_source", "custom")

    if class_source == "field":
        _field_picker(ctx, inputs)
    elif class_source == "custom":
        stored_classes = stored.get("classes") or ""
        if isinstance(stored_classes, list):
            stored_classes = ", ".join(stored_classes)
        inputs.str(
            "classes",
            label="Classes",
            required=False,
            default=stored_classes,
            description="Comma-separated class names",
        )
    # "open" needs no additional input


def _field_picker(ctx: Any, inputs: types.Object) -> None:
    """Add a dropdown of existing label fields and preview unique labels."""
    if not ctx.dataset:
        return

    label_fields = _find_label_fields(ctx.dataset)
    if not label_fields:
        inputs.view(
            "no_fields",
            types.Warning(label="No label fields found in this dataset"),
        )
        return

    field_dropdown = types.Dropdown(label="Source Field")
    for field_name, field_type in label_fields:
        field_dropdown.add_choice(
            field_name, label=f"{field_name} ({field_type})"
        )

    inputs.enum(
        "source_field",
        field_dropdown.values(),
        label="Source Field",
        view=field_dropdown,
    )

    source_field: str | None = ctx.params.get("source_field")
    if not source_field:
        return

    label_classes = _get_field_classes(ctx.dataset, source_field)
    if not label_classes:
        inputs.view(
            "no_classes",
            types.Warning(
                label=f"No labels found in field '{source_field}'"
            ),
        )
        return

    sorted_classes = sorted(label_classes)
    preview = ", ".join(sorted_classes[:20])
    if len(sorted_classes) > 20:
        preview += f", ... (+{len(sorted_classes) - 20} more)"
    inputs.view(
        "field_classes",
        types.Notice(
            label=f"Found {len(sorted_classes)} classes: {preview}"
        ),
    )


_LABEL_TYPES: set[str] = {"Classification", "Classifications", "Detections"}


def _find_label_fields(
    dataset: fo.Dataset,
) -> list[tuple[str, str]]:
    """Return ``(field_name, type_name)`` for every label field in *dataset*."""
    return [
        (name, doc_type.__name__)
        for name, field in dataset.get_field_schema().items()
        if (doc_type := getattr(field, "document_type", None)) is not None
        and doc_type.__name__ in _LABEL_TYPES
    ]


def _cost_preview(
    ctx: Any, inputs: types.Object, task: str | None
) -> None:
    """Show an inline cost estimate based on model, task, and view size."""
    model: str | None = ctx.params.get("model")
    if not model or not task:
        return

    if get_model_info(model) is None:
        inputs.view(
            "cost_notice",
            types.Notice(
                label=(
                    f"Cost preview unavailable: '{model}' not found in"
                    " pricing data"
                )
            ),
        )
        return

    image_detail: str = ctx.params.get("image_detail", "auto")
    input_tokens = PROMPT_TEXT_TOKENS + IMAGE_TOKEN_COUNTS.get(
        image_detail, 765
    )
    output_tokens = OUTPUT_TOKEN_ESTIMATES.get(task, 60)

    try:
        num_samples = len(ctx.target_view())
    except Exception:
        num_samples = ctx.dataset.count() if ctx.dataset else 0

    est = estimate_cost(
        model=model,
        num_samples=num_samples,
        est_input_tokens=input_tokens,
        est_output_tokens=output_tokens,
    )
    if not est:
        return

    per_img = est["per_image_cost"]
    total = est["total_cost"]
    label = (
        f"Estimated cost: {_fmt_usd(per_img)}/image,"
        f" {_fmt_usd(total)} total for {num_samples} samples"
    )
    view_type = types.Warning if total > 1.0 else types.Notice
    inputs.view("cost_estimate", view_type(label=label))


def _output_settings(
    ctx: Any,
    inputs: types.Object,
    task: str | None,
    stored: dict[str, Any],
) -> None:
    """Add output-field, metadata-logging, and logging settings."""
    if not task or not ctx.dataset:
        return

    inputs.view(
        "output_header",
        types.Header(label="Output Settings", divider=True),
    )

    schema = ctx.dataset.get_field_schema(flat=True)
    base_field = f"openai_infer_{task}"
    has_existing = base_field in schema

    if has_existing:
        inputs.bool(
            "overwrite_last",
            label="Overwrite last result",
            default=False,
            view=types.SwitchView(),
        )
        overwrite = ctx.params.get("overwrite_last", False)
        resolved = _resolve_field_name(ctx.dataset, task, overwrite)
        prefix = "Overwriting" if overwrite else "Writing to"
        inputs.view(
            "field_info", types.Notice(label=f"{prefix}: {resolved}")
        )
    else:
        inputs.view(
            "field_info", types.Notice(label=f"Writing to: {base_field}")
        )

    inputs.bool(
        "log_metadata",
        label="Log run metadata",
        default=False,
        view=types.SwitchView(),
        description=(
            "Store model name, prompt, and inference config"
            " on each result label and in dataset info"
        ),
    )

    _logging_settings(ctx, inputs, stored)


def _advanced_settings(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add collapsible advanced settings to the form."""
    inputs.view(
        "adv_header",
        types.Header(label="Advanced Settings", divider=True),
    )
    inputs.bool(
        "show_advanced",
        label="Show advanced settings",
        default=False,
        view=types.SwitchView(),
    )
    if not ctx.params.get("show_advanced", False):
        return

    inputs.float(
        "temperature",
        label="Temperature",
        default=stored.get("temperature"),
        min=0.0,
        max=2.0,
        description=(
            "Sampling temperature (leave empty for model default)"
        ),
    )
    inputs.int(
        "max_completion_tokens",
        label="Max Completion Tokens",
        default=stored.get("max_completion_tokens"),
        min=1,
        max=16384,
        description=(
            "Maximum tokens to generate per sample"
            " (leave empty for model default)"
        ),
    )
    inputs.float(
        "top_p",
        label="Top P",
        default=stored.get("top_p"),
        min=0.0,
        max=1.0,
        description="Nucleus sampling (leave empty for model default)",
    )
    inputs.int(
        "seed",
        label="Seed",
        default=stored.get("seed"),
        description=(
            "Random seed for reproducible results"
            " (leave empty for non-deterministic)"
        ),
    )
    inputs.int(
        "batch_size",
        label="Batch Size",
        default=stored.get("batch_size"),
        min=1,
        max=512,
        description="Number of samples per inference batch",
    )
    inputs.int(
        "max_concurrent",
        label="Max Concurrent Requests",
        default=stored.get("max_concurrent"),
        min=1,
        max=256,
        description="Maximum parallel API requests",
    )
    inputs.int(
        "max_workers",
        label="Image Loading Workers",
        default=stored.get("max_workers"),
        min=1,
        max=32,
        description="Thread pool size for parallel image loading/encoding",
    )
    inputs.int(
        "timeout",
        label="Request Timeout (seconds)",
        default=stored.get("timeout"),
        min=10,
        max=600,
        description=(
            "Per-request timeout in seconds"
            " (leave empty for SDK default)"
        ),
    )

    _image_detail_selector(inputs, stored)
    _base_url_input(inputs, stored)

    if ctx.params.get("task") == "detect":
        _detection_format_selectors(inputs, stored)


def _image_detail_selector(
    inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add the OpenAI image-detail dropdown."""
    detail_dropdown = types.Dropdown()
    detail_dropdown.add_choice(
        "auto",
        label="Auto (Recommended)",
        description="Let the model decide detail level",
    )
    detail_dropdown.add_choice(
        "low",
        label="Low",
        description="85 tokens per image, faster and cheaper",
    )
    detail_dropdown.add_choice(
        "high",
        label="High",
        description="~765 tokens per image, more detailed analysis",
    )
    inputs.enum(
        "image_detail",
        detail_dropdown.values(),
        default=stored.get("image_detail"),
        label="Image Detail",
        view=detail_dropdown,
    )


def _base_url_input(
    inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add the optional base-URL text input for Azure / proxy setups."""
    inputs.str(
        "base_url",
        label="Base URL (Optional)",
        default=stored.get("base_url", ""),
        description=(
            "Custom API base URL for Azure OpenAI or proxy setups"
            " (leave empty for standard OpenAI)"
        ),
    )


def _logging_settings(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add opt-in logging controls to the form.

    Settings are persisted so they survive app restarts.
    """
    inputs.bool(
        "enable_logging",
        label="Enable logging",
        default=stored.get("enable_logging", False),
        view=types.SwitchView(),
        description="Log inference progress and errors to stderr and optionally a file",
    )
    if not ctx.params.get("enable_logging", False):
        return

    level_dropdown = types.Dropdown()
    level_dropdown.add_choice(
        "INFO",
        label="INFO (Recommended)",
        description="Run progress and error summaries",
    )
    level_dropdown.add_choice(
        "DEBUG",
        label="DEBUG",
        description="Verbose per-batch and per-sample details",
    )
    level_dropdown.add_choice(
        "WARNING",
        label="WARNING",
        description="Only warnings and errors",
    )
    inputs.enum(
        "log_level",
        level_dropdown.values(),
        default=stored.get("log_level", "INFO"),
        label="Log Level",
        view=level_dropdown,
    )
    inputs.str(
        "log_file",
        label="Log File (Optional)",
        default=stored.get("log_file", ""),
        description=(
            "File path or directory for log output."
            " Directories get a timestamped filename;"
            " existing files auto-increment (run.log → run2.log)."
            " Leave empty for stderr only."
        ),
    )


def _detection_format_selectors(
    inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add coordinate-format and box-format dropdowns for detection tasks."""
    coord_dropdown = types.Dropdown()
    coord_dropdown.add_choice("normalized_1", label="0-1 (normalized)")
    coord_dropdown.add_choice(
        "normalized_1000", label="0-1000 (normalized)"
    )
    inputs.enum(
        "coordinate_format",
        coord_dropdown.values(),
        default=stored.get("coordinate_format"),
        label="Coordinate Format",
        view=coord_dropdown,
        description="Bounding box coordinate convention used by the model",
    )

    box_dropdown = types.Dropdown()
    box_dropdown.add_choice("xyxy", label="xyxy — corners")
    box_dropdown.add_choice("xywh", label="xywh — origin + size")
    box_dropdown.add_choice("cxcywh", label="cxcywh — center + size")
    inputs.enum(
        "box_format",
        box_dropdown.values(),
        default=stored.get("box_format"),
        label="Box Format",
        view=box_dropdown,
        description="Bounding box format produced by the model",
    )
