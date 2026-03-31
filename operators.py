"""FiftyOne operator for OpenAI inference: UI, batching, progress, and result
storage."""

import json
import os

import fiftyone.operators as foo
from fiftyone.operators import types

from .engine import OpenAIEngine
from .tasks import TaskConfig
from .utils import (
    build_image_contents,
    clear_global_config,
    get_global_config,
    normalize_classes,
    parse_config_json,
    pick_params,
    save_global_config,
)

_DEFAULTS = {
    "batch_size": 8,
    "max_tokens": 512,
    "top_p": 1.0,
    "max_concurrent": 16,
    "max_workers": 4,
    "coordinate_format": "normalized_1",
    "box_format": "xyxy",
    "image_detail": "auto",
}


class OpenAIInference(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_openai_inference",
            label="Run OpenAI Inference",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
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
        config_mode = ctx.params.get("config_mode", "manual")

        if config_mode == "json":
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
                    "**Advanced:** `temperature`, `max_tokens`, `top_p`, "
                    "`seed`, `batch_size`, `max_concurrent`, `max_workers`, "
                    "`image_detail`, `coordinate_format`, `box_format`",
                    name="params_ref",
                )
            raw = ctx.params.get("config_json")
            if raw:
                cfg, err = parse_config_json(raw)
                if err:
                    inputs.view("json_err", types.Error(label=err))
                else:
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
                                label=(f"Valid: {cfg['task']} task with {cfg['model']}")
                            ),
                        )
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
            _output_settings(ctx, inputs, task)
            _advanced_settings(ctx, inputs, stored)

        if config_mode != "reset":
            inputs.view_target(ctx)

        return types.Property(inputs, view=types.View(label="OpenAI Inference"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", None)

    def execute(self, ctx):
        params = ctx.params
        config_mode = params.get("config_mode", "manual")

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
                yield _error(ctx, "Config missing required 'model' or 'task'")
                return

        # Resolve classes from field picker if applicable
        _resolve_classes_from_field(ctx, params)

        try:
            engine, api_key = _create_engine(params, ctx.secrets)
            task = _create_task(params)
        except Exception as e:
            yield _error(ctx, str(e))
            return

        if params.get("temperature") is None:
            engine.temperature = task.default_temperature

        batch_size = params.get("batch_size", _DEFAULTS["batch_size"])
        image_detail = params.get("image_detail", _DEFAULTS["image_detail"])

        # Get target samples and resolve output field
        view = ctx.target_view()
        ids = view.values("id")
        filepaths = view.values("filepath")
        total = len(ids)
        max_workers = params.get("max_workers", _DEFAULTS["max_workers"])

        response_model = task.get_response_model()

        # Resolve output field name
        field_name = _resolve_field_name(
            ctx.dataset,
            params["task"],
            params.get("overwrite_last", False),
        )

        # Build metadata for optional per-label logging
        log_metadata = params.get("log_metadata", False)
        if log_metadata:
            full_prompt = ""
            if task.system_prompt:
                full_prompt += f"[system] {task.system_prompt}\n"
            full_prompt += f"[user] {task.prompt}"

            infer_cfg = {
                "temperature": engine.temperature,
                "max_tokens": engine.max_tokens,
                "top_p": engine.top_p,
                "batch_size": batch_size,
                "coordinate_format": task.coordinate_format,
                "box_format": task.box_format,
                "max_concurrent": engine.max_concurrent,
            }

        # Clear stale error fields when overwriting
        if params.get("overwrite_last", False):
            error_field = f"{field_name}_error"
            schema = ctx.dataset.get_field_schema(flat=True)
            if error_field in schema:
                ctx.dataset.set_values(
                    error_field,
                    {sid: None for sid in ids},
                    key_field="id",
                )

        # Collect image dimensions for pixel coordinate normalization
        need_dims = task.task == "detect" and task.coordinate_format == "pixel"
        if need_dims:
            view.compute_metadata()
            widths = view.values("metadata.width")
            heights = view.values("metadata.height")
        else:
            widths = [None] * total
            heights = [None] * total

        # Process in batches
        processed = 0
        total_errors = 0

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_paths = filepaths[i : i + batch_size]
            batch_widths = widths[i : i + batch_size]
            batch_heights = heights[i : i + batch_size]

            # Parallel image content construction
            image_contents = build_image_contents(
                batch_paths,
                max_workers=max_workers,
                image_detail=image_detail,
            )

            # Build messages for each image
            batch_messages = [task.build_messages(img) for img in image_contents]

            # Batch inference with structured output
            responses = engine.infer_batch(
                batch_messages,
                response_model=response_model,
            )

            # Parse responses with per-sample error handling
            results = {}
            errors = {}
            for sid, resp, img_w, img_h in zip(
                batch_ids, responses, batch_widths, batch_heights
            ):
                if isinstance(resp, Exception):
                    errors[sid] = f"{type(resp).__name__}: {resp}"
                    total_errors += 1
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
                    errors[sid] = f"{type(e).__name__}: {e}"
                    total_errors += 1

            # Bulk-write results and errors
            _write_batch_results(
                ctx.dataset,
                field_name,
                results,
                errors,
            )

            # Progress
            processed += len(batch_ids)
            progress_label = f"{processed}/{total} samples"
            if total_errors:
                progress_label += f" ({total_errors} errors)"

            if ctx.delegated:
                ctx.set_progress(progress=processed / total, label=progress_label)
            else:
                yield ctx.trigger(
                    "set_progress",
                    dict(progress=processed / total, label=progress_label),
                )

        # Persist run metadata and settings
        if log_metadata:
            runs = ctx.dataset.info.get("openai_runs", {})
            runs[field_name] = {
                "model_name": params["model"],
                "prompt": full_prompt,
                "infer_cfg": infer_cfg,
            }
            ctx.dataset.info["openai_runs"] = runs

        save_global_config(params)
        ctx.dataset.info["_openai_config"] = pick_params(params, exclude=("api_key",))
        ctx.dataset.save()

        # Notify / reload
        if ctx.delegated:
            store = ctx.store("openai_status")
            store.set("done", True)
        else:
            yield ctx.trigger("reload_dataset")

    def resolve_output(self, ctx):
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
    """Subscribes to the ``openai_status`` execution store via MongoDB
    change-stream notifications. When the delegated worker writes a
    completion signal, this operator fires a toast and reloads the dataset.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="check_openai_status",
            label="Check OpenAI Status",
            on_dataset_open=True,
            execute_as_generator=True,
            unlisted=True,
        )

    async def execute(self, ctx):
        import asyncio

        from fiftyone.operators.store.notification_service import (
            default_notification_service,
        )

        if not ctx.dataset:
            return

        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        def _on_change(_message):
            loop.call_soon_threadsafe(event.set)

        sub_id = default_notification_service.subscribe(
            "openai_status",
            callback=_on_change,
            dataset_id=str(ctx.dataset._doc.id),
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=600)

            store = ctx.store("openai_status")
            store.delete("done")

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


# -- Helpers --


def _error(ctx, message):
    """Yield-safe error via set_progress."""
    label = f"Error: {message}"
    if ctx.delegated:
        ctx.set_progress(progress=0, label=label)
        return None
    return ctx.trigger("set_progress", {"progress": 0, "label": label})


def _resolve_config(ctx):
    """Merge dataset config > global config > _DEFAULTS."""
    merged = dict(_DEFAULTS)
    for cfg in (
        get_global_config(),
        ctx.dataset.info.get("_openai_config") or {},
    ):
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _create_engine(params, secrets):
    """Build an OpenAIEngine from operator params and secrets.

    Returns (engine, api_key).
    """
    api_key = (
        params.get("api_key")
        or secrets.get("FIFTYONE_OPENAI_API_KEY", None)
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No API key configured. Set FIFTYONE_OPENAI_API_KEY in"
            " FiftyOne secrets, OPENAI_API_KEY in your environment,"
            " or provide an API key in the advanced settings."
        )
    base_url = params.get("base_url") or None

    engine = OpenAIEngine(
        model=params["model"],
        api_key=api_key,
        base_url=base_url,
        max_concurrent=params.get("max_concurrent", _DEFAULTS["max_concurrent"]),
        temperature=params.get("temperature", None),
        max_tokens=params.get("max_tokens", _DEFAULTS["max_tokens"]),
        top_p=params.get("top_p", _DEFAULTS["top_p"]),
        seed=params.get("seed", None),
    )

    return engine, api_key


def _create_task(params):
    """Build a TaskConfig from operator params."""
    classes = normalize_classes(params.get("classes"))

    return TaskConfig(
        task=params["task"],
        prompt=params.get("prompt_override") or params.get("prompt"),
        system_prompt=params.get("system_prompt"),
        classes=classes,
        coordinate_format=params.get(
            "coordinate_format", _DEFAULTS["coordinate_format"]
        ),
        box_format=params.get("box_format", _DEFAULTS["box_format"]),
        question=params.get("question", ""),
    )


def _resolve_classes_from_field(ctx, params):
    """If user selected a dataset field as class source, extract unique
    labels and set them in params."""
    class_source = params.get("class_source", "custom")
    if class_source != "field":
        return

    source_field = params.get("source_field")
    if not source_field or not ctx.dataset:
        return

    label_classes = _get_field_classes(ctx.dataset, source_field)
    if label_classes:
        params["classes"] = ", ".join(sorted(label_classes))


def _get_field_classes(dataset, field_name):
    """Extract unique class labels from a dataset label field.

    Handles Classification, Classifications, and Detections fields.
    """
    schema = dataset.get_field_schema(flat=True)

    # Try Classification path: field.label
    label_path = f"{field_name}.label"
    if label_path in schema:
        return dataset.distinct(label_path)

    # Try Classifications path: field.classifications.label
    cls_path = f"{field_name}.classifications.label"
    if cls_path in schema:
        return dataset.distinct(cls_path)

    # Try Detections path: field.detections.label
    det_path = f"{field_name}.detections.label"
    if det_path in schema:
        return dataset.distinct(det_path)

    return []


def _resolve_field_name(dataset, task_name, overwrite=False):
    """Resolve the output field name for a task run.

    Produces: openai_infer_task_name, openai_infer_task_name1, ...
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


def _write_batch_results(dataset, field_name, results, errors):
    """Write a batch of results and errors as flat sample fields."""
    if results:
        dataset.set_values(
            field_name,
            results,
            key_field="id",
            dynamic=True,
        )
    if errors:
        dataset.set_values(
            f"{field_name}_error",
            errors,
            key_field="id",
            dynamic=True,
        )


# -- UI helper functions --


def _model_selector(ctx, inputs, stored):
    """Add model selection field."""
    inputs.str(
        "model",
        label="Model",
        required=True,
        default=stored.get("model", ""),
        description="LiteLLM model ID (e.g., gpt-4o, gpt-4o-mini)",
    )


def _task_selector(ctx, inputs, stored):
    """Add task dropdown."""
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


def _task_settings(ctx, inputs, task, stored):
    """Add task-specific settings with enhanced class label input."""
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

    inputs.str(
        "prompt_override",
        label="Prompt Override",
        required=False,
        default=stored.get("prompt_override", ""),
        description="Override the default prompt for this task",
    )


def _class_source_selector(ctx, inputs, stored):
    """Add 3-source class label input: from dataset field, custom, or
    open-ended."""
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

    class_source = ctx.params.get("class_source", "custom")

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


def _field_picker(ctx, inputs):
    """Add dropdown of existing label fields and show unique labels."""
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
            field_name,
            label=f"{field_name} ({field_type})",
        )

    inputs.enum(
        "source_field",
        field_dropdown.values(),
        label="Source Field",
        view=field_dropdown,
    )

    source_field = ctx.params.get("source_field")
    if source_field:
        label_classes = _get_field_classes(ctx.dataset, source_field)
        if label_classes:
            sorted_classes = sorted(label_classes)
            preview = ", ".join(sorted_classes[:20])
            if len(sorted_classes) > 20:
                preview += f", ... (+{len(sorted_classes) - 20} more)"
            inputs.view(
                "field_classes",
                types.Notice(label=(f"Found {len(sorted_classes)} classes: {preview}")),
            )
        else:
            inputs.view(
                "no_classes",
                types.Warning(label=f"No labels found in field '{source_field}'"),
            )


def _find_label_fields(dataset):
    """Find all label fields (Classification, Classifications, Detections)
    in the dataset.

    Returns list of (field_name, type_name) tuples.
    """
    result = []
    schema = dataset.get_field_schema()
    for field_name, field in schema.items():
        doc_type = getattr(field, "document_type", None)
        if doc_type is None:
            continue
        type_name = doc_type.__name__
        if type_name in ("Classification", "Classifications", "Detections"):
            result.append((field_name, type_name))
    return result


def _cost_preview(ctx, inputs, task):
    """Show inline cost estimate based on model, task, and target view."""
    model = ctx.params.get("model")
    if not model or not task:
        return

    cost_info = OpenAIEngine.get_model_info(model)
    if cost_info is None:
        inputs.view(
            "cost_notice",
            types.Notice(
                label=(
                    f"Cost preview unavailable: '{model}' not found in"
                    " LiteLLM pricing data"
                )
            ),
        )
        return

    # Estimate tokens
    image_detail = ctx.params.get("image_detail", "auto")
    image_tokens = {"low": 85, "high": 765, "auto": 765}.get(image_detail, 765)
    prompt_text_tokens = 50
    input_tokens_per_sample = prompt_text_tokens + image_tokens

    task_cfg = TaskConfig(task=task, classes=None)
    output_tokens_per_sample = task_cfg.estimated_output_tokens()

    # Get sample count
    try:
        num_samples = len(ctx.target_view())
    except Exception:
        num_samples = ctx.dataset.count() if ctx.dataset else 0

    estimate = OpenAIEngine.estimate_cost(
        model=model,
        num_samples=num_samples,
        est_input_tokens=input_tokens_per_sample,
        est_output_tokens=output_tokens_per_sample,
    )

    if estimate:
        per_img = estimate["per_image_cost"]
        total = estimate["total_cost"]

        label = (
            f"Estimated cost: ~${per_img:.5f}/image,"
            f" ~${total:.4f} total for {num_samples} samples"
        )

        view_type = types.Warning if total > 1.0 else types.Notice
        inputs.view("cost_estimate", view_type(label=label))


def _output_settings(ctx, inputs, task):
    """Add output field and metadata logging settings."""
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
            "field_info",
            types.Notice(label=f"{prefix}: {resolved}"),
        )
    else:
        inputs.view(
            "field_info",
            types.Notice(label=f"Writing to: {base_field}"),
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


def _advanced_settings(ctx, inputs, stored):
    """Add collapsible advanced settings."""
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
        description="Sampling temperature (leave empty for task default)",
    )
    inputs.int(
        "max_tokens",
        label="Max Tokens",
        default=stored.get("max_tokens"),
        min=1,
        max=4096,
        description="Maximum tokens to generate per sample",
    )
    inputs.float(
        "top_p",
        label="Top P",
        default=stored.get("top_p"),
        min=0.0,
        max=1.0,
    )
    inputs.int(
        "seed",
        label="Seed",
        default=stored.get("seed"),
        description=(
            "Random seed for reproducible results (leave empty for non-deterministic)"
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

    # OpenAI image detail level
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

    # Optional base URL for Azure/proxy setups
    inputs.str(
        "base_url",
        label="Base URL (Optional)",
        default=stored.get("base_url", ""),
        description=(
            "Custom API base URL for Azure OpenAI or proxy setups"
            " (leave empty for standard OpenAI)"
        ),
    )

    task = ctx.params.get("task")
    if task == "detect":
        coord_dropdown = types.Dropdown()
        coord_dropdown.add_choice("normalized_1", label="0-1 (normalized)")
        coord_dropdown.add_choice("normalized_1000", label="0-1000 (normalized)")
        coord_dropdown.add_choice("pixel", label="Pixel coordinates")
        inputs.enum(
            "coordinate_format",
            coord_dropdown.values(),
            default=stored.get("coordinate_format"),
            label="Coordinate Format",
            view=coord_dropdown,
            description=("Bounding box coordinate convention used by the model"),
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
