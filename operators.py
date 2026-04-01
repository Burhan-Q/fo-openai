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
    EXEMPLAR_TEXT_TOKENS,
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

        # Config mode selector (always visible)
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
            # -- Cost summary ABOVE tabs (persistent banner) --
            task = ctx.params.get("task")
            _cost_summary(ctx, inputs, task)

            # -- 5-tab layout --
            tabs = types.TabsView()
            for key, label in [
                ("model", "Model"),
                ("task", "Task"),
                ("exemplars", "Exemplars"),
                ("logging", "Logging"),
                ("advanced", "Advanced"),
            ]:
                tabs.add_choice(key, label=label)

            inputs.enum(
                "active_tab",
                tabs.values(),
                default="model",
                label="Settings",
                view=tabs,
            )
            active_tab: str = ctx.params.get("active_tab", "model")

            if active_tab == "model":
                _model_selector(ctx, inputs, stored)
                _base_url_input(inputs, stored)
            elif active_tab == "task":
                task = _task_selector(ctx, inputs, stored)
                _task_settings(ctx, inputs, task, stored)
                _output_settings(ctx, inputs, task, stored)
            elif active_tab == "exemplars":
                _exemplar_tab(ctx, inputs, stored)
            elif active_tab == "logging":
                _logging_settings(ctx, inputs, stored)
            elif active_tab == "advanced":
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

        # -- Resolve exemplars (if enabled) --
        exemplar_messages: list[dict[str, Any]] | None = None
        if params.get("exemplars_enabled"):
            from .exemplars import build_exemplar_messages, resolve_exemplars

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
                task=params["task"],
                classes=normalize_classes(params.get("classes")),
                coordinate_format=params.get(
                    "coordinate_format", _DEFAULTS["coordinate_format"]
                ),
                box_format=params.get(
                    "box_format", _DEFAULTS["box_format"]
                ),
                image_detail=image_detail,
                max_workers=max_workers,
            )
            logger.info(
                "Exemplars: %d message pairs built",
                len(exemplar_messages) // 2,
            )

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
                task.build_messages(img, exemplar_messages=exemplar_messages)
                for img in image_contents
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
        if params.get("exemplars_enabled") and exemplar_messages:
            run_entry["exemplars_enabled"] = True
            run_entry["exemplar_count"] = len(exemplar_messages) // 2
            run_entry["exemplar_source"] = params.get("exemplar_source")
            run_entry["exemplar_label_field"] = params.get(
                "exemplar_label_field"
            )
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


def _advanced_settings(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add advanced settings fields to the form."""
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

    if ctx.params.get("task") == "detect":
        _detection_format_selectors(inputs, stored)

    inputs.str(
        "system_prompt",
        label="System Prompt Override",
        default=stored.get("system_prompt", ""),
        description=(
            "Custom system prompt. When exemplars are active, a"
            " few-shot preamble is automatically prepended."
        ),
        view=types.TextFieldView(multiline=True),
    )


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


# ---------------------------------------------------------------------------
# Exemplar tab
# ---------------------------------------------------------------------------

# Compatible field types per task for exemplar label field selection
_EXEMPLAR_FIELD_TYPES: dict[str, set[str]] = {
    "classify": {"Classification"},
    "tag": {"Classifications", "Classification"},
    "caption": {"Classification"},
    "detect": {"Detections"},
    "vqa": {"Classification"},
    "ocr": {"Classification"},
}


def _exemplar_tab(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Render the Exemplars tab contents.

    Shows an enable toggle (default OFF). When enabled, presents the
    exemplar source selector, label field picker, and a live preview
    of resolved exemplar count and token cost.

    Args:
        ctx: Operator execution context.
        inputs: Form object to add fields to.
        stored: Merged configuration from persistence layers.
    """
    inputs.bool(
        "exemplars_enabled",
        label="Enable few-shot exemplars",
        default=stored.get("exemplars_enabled", False),
        view=types.SwitchView(),
    )

    if not ctx.params.get("exemplars_enabled", False):
        inputs.view(
            "exemplar_info",
            types.Notice(
                label=(
                    "Provide labeled samples as few-shot examples to"
                    " guide model output quality. Toggle on to configure."
                )
            ),
        )
        return

    # Exemplar source selector
    source_radio = types.RadioGroup()
    for key, label in [
        ("saved_view", "Saved view"),
        ("sample_ids", "Sample IDs"),
        ("tag", "Tag"),
        ("field", "Field"),
    ]:
        source_radio.add_choice(key, label=label)

    inputs.enum(
        "exemplar_source",
        source_radio.values(),
        default=stored.get("exemplar_source", "saved_view"),
        label="Exemplar Source",
        view=source_radio,
    )

    source: str = ctx.params.get("exemplar_source", "saved_view")

    # Source-specific fields (only show the active source's input)
    if source == "saved_view":
        _exemplar_saved_view_picker(ctx, inputs, stored)
    elif source == "sample_ids":
        inputs.str(
            "exemplar_sample_ids",
            label="Sample IDs",
            default=stored.get("exemplar_sample_ids", ""),
            description="Comma-separated sample IDs to use as exemplars",
        )
    elif source == "tag":
        inputs.str(
            "exemplar_tag",
            label="Tag",
            default=stored.get("exemplar_tag", ""),
            description="Tag name to select exemplar samples",
        )
    elif source == "field":
        _exemplar_field_picker(ctx, inputs, stored)

    # Exemplar label field picker
    _exemplar_label_field_picker(ctx, inputs, stored)

    # Exemplar preview (count + cost estimate)
    _exemplar_preview(ctx, inputs)


def _exemplar_saved_view_picker(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add a dropdown of saved views for exemplar source selection."""
    if not ctx.dataset:
        return
    saved_views = ctx.dataset.list_saved_views()
    if not saved_views:
        inputs.view(
            "no_saved_views",
            types.Warning(label="No saved views found in this dataset"),
        )
        return

    view_dropdown = types.Dropdown()
    for name in saved_views:
        view_dropdown.add_choice(name, label=name)

    inputs.enum(
        "exemplar_view_name",
        view_dropdown.values(),
        default=stored.get("exemplar_view_name", ""),
        label="Saved View",
        view=view_dropdown,
    )


def _exemplar_field_picker(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add field + value inputs for field-based exemplar source."""
    if not ctx.dataset:
        return
    schema = ctx.dataset.get_field_schema()
    field_dropdown = types.Dropdown()
    for name in sorted(schema.keys()):
        field_dropdown.add_choice(name, label=name)

    inputs.enum(
        "exemplar_field_name",
        field_dropdown.values(),
        default=stored.get("exemplar_field_name", ""),
        label="Field Name",
        view=field_dropdown,
    )

    field_name = ctx.params.get("exemplar_field_name")
    if not field_name:
        return

    # Auto-detect boolean fields — skip value input
    field_obj = schema.get(field_name)
    if field_obj is not None and getattr(field_obj, "ftype", None) is not None:
        from fiftyone.core.fields import BooleanField
        if isinstance(field_obj, BooleanField):
            return

    inputs.str(
        "exemplar_field_value",
        label="Match Value",
        default=stored.get("exemplar_field_value", ""),
        description="Value to match (e.g., 'true', 'exemplar')",
    )


def _exemplar_label_field_picker(
    ctx: Any, inputs: types.Object, stored: dict[str, Any]
) -> None:
    """Add a dropdown of label fields compatible with the selected task.

    Args:
        ctx: Operator execution context.
        inputs: Form object to add fields to.
        stored: Merged configuration from persistence layers.
    """
    if not ctx.dataset:
        return

    task: str | None = ctx.params.get("task")
    compat_types = _EXEMPLAR_FIELD_TYPES.get(task or "", set())

    label_fields = _find_label_fields(ctx.dataset)
    # Filter to task-compatible types (show all if task not yet selected)
    if compat_types:
        label_fields = [
            (name, ftype) for name, ftype in label_fields
            if ftype in compat_types
        ]

    if not label_fields:
        msg = "No compatible label fields found"
        if task:
            msg += f" for task '{task}'"
        inputs.view("no_exemplar_fields", types.Warning(label=msg))
        return

    field_dropdown = types.Dropdown()
    for name, ftype in label_fields:
        field_dropdown.add_choice(name, label=f"{name} ({ftype})")

    inputs.enum(
        "exemplar_label_field",
        field_dropdown.values(),
        default=stored.get("exemplar_label_field", ""),
        label="Exemplar Label Field",
        view=field_dropdown,
        description="Field containing the expected output for each exemplar",
    )


def _exemplar_preview(ctx: Any, inputs: types.Object) -> None:
    """Show a preview of resolved exemplar count and token estimate.

    Args:
        ctx: Operator execution context.
        inputs: Form object to add fields to.
    """
    source: str = ctx.params.get("exemplar_source", "saved_view")
    label_field: str | None = ctx.params.get("exemplar_label_field")

    if not label_field:
        inputs.view(
            "exemplar_warn",
            types.Warning(label="Select an exemplar label field to continue"),
        )
        return

    # Attempt to count exemplar samples
    count = _count_exemplar_samples(ctx, source)
    if count is None:
        return
    if count == 0:
        inputs.view(
            "exemplar_empty",
            types.Warning(label="No exemplar samples resolved from this source"),
        )
        return

    image_detail: str = ctx.params.get("image_detail", "auto")
    img_tokens = IMAGE_TOKEN_COUNTS.get(image_detail, 765)
    per_exemplar = img_tokens + EXEMPLAR_TEXT_TOKENS
    total_overhead = count * per_exemplar

    inputs.view(
        "exemplar_count",
        types.Notice(
            label=(
                f"{count} exemplar samples resolved. "
                f"Each adds ~{per_exemplar} tokens per inference call "
                f"(~{total_overhead} total tokens overhead per call)."
            )
        ),
    )

    if total_overhead > 5000:
        inputs.view(
            "exemplar_cost_warn",
            types.Warning(
                label=(
                    f"High exemplar token overhead ({total_overhead} tokens/call). "
                    f"Exemplar images are included in EVERY inference call. "
                    f"Consider reducing exemplar count if cost is a concern."
                )
            ),
        )


def _count_exemplar_samples(ctx: Any, source: str) -> int | None:
    """Attempt to count exemplar samples based on the current source config.

    Returns count or ``None`` if source configuration is incomplete.
    """
    if not ctx.dataset:
        return None
    try:
        if source == "saved_view":
            name = ctx.params.get("exemplar_view_name")
            if not name:
                return None
            return len(ctx.dataset.load_saved_view(name))
        if source == "sample_ids":
            ids_str = ctx.params.get("exemplar_sample_ids", "")
            ids = [s.strip() for s in ids_str.split(",") if s.strip()]
            return len(ids) if ids else None
        if source == "tag":
            tag = ctx.params.get("exemplar_tag", "").strip()
            if not tag:
                return None
            return len(ctx.dataset.match_tags(tag))
        if source == "field":
            field_name = ctx.params.get("exemplar_field_name")
            if not field_name:
                return None
            field_value = ctx.params.get("exemplar_field_value")
            from fiftyone import ViewField as F
            if not field_value:
                return len(ctx.dataset.match(F(field_name) == True))  # noqa: E712
            return len(ctx.dataset.match(F(field_name) == field_value))
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Cost summary (always visible, outside tabs)
# ---------------------------------------------------------------------------

# Default cost warning threshold in USD
_DEFAULT_COST_WARN: float = 5.0


def _get_cost_warn_threshold() -> float:
    """Read the cost warning threshold from env, defaulting to $5."""
    import os
    try:
        return float(os.environ.get("FIFTYONE_OPENAI_COST_WARN", _DEFAULT_COST_WARN))
    except (ValueError, TypeError):
        return _DEFAULT_COST_WARN


def _estimate_prompt_tokens(ctx: Any, task: str | None) -> int:
    """Estimate text tokens for system prompt + user prompt.

    Uses a rough 4-chars-per-token heuristic. Accounts for custom
    prompts and long class lists, which can significantly exceed the
    default ``PROMPT_TEXT_TOKENS`` constant.

    Args:
        ctx: Operator execution context (reads params).
        task: Currently selected task, or ``None``.

    Returns:
        Estimated token count for prompt text (no image tokens).
    """
    text_len = 0
    # System prompt (default or custom)
    system = ctx.params.get("system_prompt", "")
    if system:
        text_len += len(system)
    else:
        # Default system prompts are ~60-120 chars
        text_len += 100
    # User prompt (default or custom override)
    prompt = ctx.params.get("prompt_override") or ""
    if prompt:
        text_len += len(prompt)
    else:
        text_len += 60  # default task prompts
    # Classes list (can be very long)
    classes = ctx.params.get("classes", "")
    if isinstance(classes, str) and classes:
        text_len += len(classes)
    elif isinstance(classes, list):
        text_len += sum(len(str(c)) for c in classes)
    # Question (VQA)
    if task == "vqa":
        text_len += len(ctx.params.get("question", ""))
    # Rough token estimate: ~4 chars per token, with a floor
    return max(PROMPT_TEXT_TOKENS, text_len // 4)


def _fmt_tokens(n: int) -> str:
    """Format a token count with comma separators for readability."""
    return f"{n:,}"


def _cost_summary(
    ctx: Any, inputs: types.Object, task: str | None
) -> None:
    """Render a cost summary above the tabs as a persistent banner.

    Inference and exemplar costs are computed independently so the
    table rows are additive and the total is their sum. Includes a
    Tokens/Call and Total Tokens column for usage visibility.

    Args:
        ctx: Operator execution context.
        inputs: Form object to add fields to.
        task: Currently selected task, or ``None``.
    """
    model: str | None = ctx.params.get("model")
    if not model or not task:
        inputs.view(
            "cost_pending",
            types.Notice(label="Select a model and task to see cost estimate"),
        )
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
    img_tokens = IMAGE_TOKEN_COUNTS.get(image_detail, 765)
    output_tokens = OUTPUT_TOKEN_ESTIMATES.get(task, 60)
    prompt_tokens = _estimate_prompt_tokens(ctx, task)

    try:
        num_samples = len(ctx.target_view())
    except Exception:
        num_samples = ctx.dataset.count() if ctx.dataset else 0

    # -- Inference tokens & cost (without exemplars) --
    infer_input_per_call = prompt_tokens + img_tokens
    infer_tokens_per_call = infer_input_per_call + output_tokens
    inference_est = estimate_cost(
        model=model,
        num_samples=num_samples,
        est_input_tokens=infer_input_per_call,
        est_output_tokens=output_tokens,
    )
    if not inference_est:
        return

    infer_per_sample = inference_est["per_image_cost"]
    infer_total = inference_est["total_cost"]
    input_cpt = inference_est.get("input_cost_per_token", 0)

    # -- Exemplar tokens & cost (independent, input tokens only) --
    exemplars_enabled = ctx.params.get("exemplars_enabled", False)
    exemplar_count = 0
    exemplar_tokens_per_call = 0
    exemplar_per_sample = 0.0
    exemplar_total = 0.0
    if exemplars_enabled:
        source = ctx.params.get("exemplar_source", "saved_view")
        exemplar_count = _count_exemplar_samples(ctx, source) or 0
        exemplar_tokens_per_call = exemplar_count * (img_tokens + EXEMPLAR_TEXT_TOKENS)
        exemplar_per_sample = exemplar_tokens_per_call * input_cpt
        exemplar_total = exemplar_per_sample * num_samples

    # -- Combined totals --
    grand_per_sample = infer_per_sample + exemplar_per_sample
    grand_total = infer_total + exemplar_total
    grand_tokens_per_call = infer_tokens_per_call + exemplar_tokens_per_call
    grand_tokens_total = grand_tokens_per_call * num_samples

    # Build markdown table with token columns
    rows = [
        "| | Tokens/Call | Total Tokens | Cost/Sample | Total Cost |",
        "|---|--:|--:|--:|--:|",
        (
            f"| **Prompt** | {_fmt_tokens(prompt_tokens)} | "
            f"{_fmt_tokens(prompt_tokens * num_samples)} | | |"
        ),
        (
            f"| **Image** | {_fmt_tokens(img_tokens)} | "
            f"{_fmt_tokens(img_tokens * num_samples)} | | |"
        ),
        (
            f"| **Output** | {_fmt_tokens(output_tokens)} | "
            f"{_fmt_tokens(output_tokens * num_samples)} | | |"
        ),
        (
            f"| **Inference** | {_fmt_tokens(infer_tokens_per_call)} | "
            f"{_fmt_tokens(infer_tokens_per_call * num_samples)} | "
            f"{_fmt_usd(infer_per_sample)} | {_fmt_usd(infer_total)} |"
        ),
    ]

    if exemplars_enabled and exemplar_count > 0:
        rows.append(
            f"| **Exemplars** ({exemplar_count}) | "
            f"{_fmt_tokens(exemplar_tokens_per_call)} | "
            f"{_fmt_tokens(exemplar_tokens_per_call * num_samples)} | "
            f"{_fmt_usd(exemplar_per_sample)} | {_fmt_usd(exemplar_total)} |"
        )

    rows.append(
        f"| **Total** | {_fmt_tokens(grand_tokens_per_call)} | "
        f"**{_fmt_tokens(grand_tokens_total)}** | "
        f"{_fmt_usd(grand_per_sample)} | **{_fmt_usd(grand_total)}** |"
    )

    rows.append(f"\n*{num_samples} samples*")

    table_md = "\n".join(rows)

    warn_threshold = _get_cost_warn_threshold()
    if grand_total >= warn_threshold:
        inputs.view(
            "cost_warn",
            types.Warning(
                label=(
                    f"Estimated total cost: {_fmt_usd(grand_total)}"
                    f" (above ${warn_threshold:.0f} threshold)"
                )
            ),
        )
    inputs.md(table_md, name="cost_table")
