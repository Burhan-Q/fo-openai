"""Few-shot exemplar resolution, serialization, and message building.

Converts user-selected FiftyOne samples with existing labels into
OpenAI chat message pairs (user image + assistant JSON response) that
serve as reference examples for the model.
"""

from __future__ import annotations

from typing import Any

import fiftyone as fo
from fiftyone import ViewField as F

from ._log import get_logger
from .tasks import (
    ClassifyResponse,
    DetectResponse,
    DetectionItem,
    TagResponse,
    TextResponse,
    VQAResponse,
    _constrained_classify_model,
    _constrained_detect_model,
    _constrained_tag_model,
)
from .utils import build_image_contents

logger = get_logger(__name__)

# Framing text appended to each exemplar user message.
_EXEMPLAR_USER_TEXT: str = (
    "REFERENCE EXAMPLE — This image has already been labeled. "
    "The correct output for this image is shown in the next message. "
    "DO NOT analyze this image."
)


def resolve_exemplars(
    dataset: fo.Dataset,
    source: str,
    view_name: str | None = None,
    sample_ids: str | None = None,
    tag: str | None = None,
    field_name: str | None = None,
    field_value: Any = None,
) -> fo.DatasetView:
    """Resolve exemplar samples from the active source selection.

    Only the parameters matching *source* are used; all others are
    ignored regardless of their value.

    Args:
        dataset: The FiftyOne dataset to query.
        source: Active selection method — one of ``"saved_view"``,
            ``"sample_ids"``, ``"tag"``, or ``"field"``.
        view_name: Name of a saved view (used when *source* is
            ``"saved_view"``).
        sample_ids: Comma-separated sample IDs (used when *source* is
            ``"sample_ids"``).
        tag: Tag name to match (used when *source* is ``"tag"``).
        field_name: Field name to filter on (used when *source* is
            ``"field"``).
        field_value: Value to match against *field_name*. For boolean
            fields pass ``True``; for string/enum fields pass the
            desired value.

    Returns:
        A :class:`fiftyone.core.view.DatasetView` containing the
        resolved exemplar samples.

    Raises:
        ValueError: If the source is invalid or resolution produces
            zero samples.
    """
    if source == "saved_view":
        return _resolve_saved_view(dataset, view_name)
    if source == "sample_ids":
        return _resolve_sample_ids(dataset, sample_ids)
    if source == "tag":
        return _resolve_tag(dataset, tag)
    if source == "field":
        return _resolve_field(dataset, field_name, field_value)
    raise ValueError(
        f"Unknown exemplar source: '{source}'. "
        f"Must be one of: saved_view, sample_ids, tag, field."
    )


def _resolve_saved_view(
    dataset: fo.Dataset, view_name: str | None
) -> fo.DatasetView:
    """Load a saved view by name."""
    if not view_name:
        raise ValueError("Exemplar source is 'saved_view' but no view name was provided.")
    available = dataset.list_saved_views()
    if view_name not in available:
        raise ValueError(
            f"Saved view '{view_name}' does not exist. "
            f"Available views: {', '.join(available) if available else '(none)'}."
        )
    view = dataset.load_saved_view(view_name)
    if not len(view):
        raise ValueError(f"Saved view '{view_name}' contains no samples.")
    logger.info("Resolved %d exemplar samples from saved view '%s'", len(view), view_name)
    return view


def _resolve_sample_ids(
    dataset: fo.Dataset, sample_ids: str | None
) -> fo.DatasetView:
    """Select samples by comma-separated IDs."""
    if not sample_ids or not sample_ids.strip():
        raise ValueError("Exemplar source is 'sample_ids' but no IDs were provided.")
    id_list = [s.strip() for s in sample_ids.split(",") if s.strip()]
    if not id_list:
        raise ValueError("Exemplar source is 'sample_ids' but no valid IDs were provided.")
    view = dataset.select(id_list)
    found_ids = set(view.values("id"))
    missing = [sid for sid in id_list if sid not in found_ids]
    if missing:
        raise ValueError(
            f"Sample IDs not found in dataset: {', '.join(missing)}. "
            f"Verify these IDs exist in '{dataset.name}'."
        )
    logger.info("Resolved %d exemplar samples from sample IDs", len(view))
    return view


def _resolve_tag(
    dataset: fo.Dataset, tag: str | None
) -> fo.DatasetView:
    """Match samples by tag name."""
    if not tag or not tag.strip():
        raise ValueError("Exemplar source is 'tag' but no tag name was provided.")
    tag = tag.strip()
    view = dataset.match_tags(tag)
    if not len(view):
        raise ValueError(
            f"No samples found matching tag '{tag}'. "
            f"Verify the tag exists and has matching samples in '{dataset.name}'."
        )
    logger.info("Resolved %d exemplar samples from tag '%s'", len(view), tag)
    return view


def _resolve_field(
    dataset: fo.Dataset, field_name: str | None, field_value: Any
) -> fo.DatasetView:
    """Match samples by field value."""
    if not field_name:
        raise ValueError("Exemplar source is 'field' but no field name was provided.")
    if field_value is None:
        field_value = True
    view = dataset.match(F(field_name) == field_value)
    if not len(view):
        raise ValueError(
            f"No samples found matching {field_name}={field_value!r}. "
            f"Verify the field exists and has matching samples in '{dataset.name}'."
        )
    logger.info(
        "Resolved %d exemplar samples from field '%s'=%r",
        len(view), field_name, field_value,
    )
    return view


# ---------------------------------------------------------------------------
# Serialization — FiftyOne label → Pydantic response model JSON
# ---------------------------------------------------------------------------

# Mapping: task name → compatible FiftyOne label types
_TASK_COMPAT: dict[str, set[str]] = {
    "classify": {"Classification", "str"},
    "tag": {"Classifications", "Classification", "str"},
    "caption": {"str", "Classification"},
    "detect": {"Detections"},
    "vqa": {"str", "Classification"},
    "ocr": {"str", "Classification"},
}


def serialize_exemplar(
    sample: fo.Sample,
    label_field: str,
    task: str,
    classes: list[str] | None = None,
    coordinate_format: str = "pixel",
    box_format: str = "xyxy",
) -> str:
    """Serialize a sample's label field to the task's response model JSON.

    Args:
        sample: The FiftyOne sample containing the label.
        label_field: Name of the field to read from the sample.
        task: Task identifier (``"classify"``, ``"tag"``, ``"detect"``,
            ``"caption"``, ``"vqa"``, ``"ocr"``).
        classes: Optional class constraint list for classify/tag/detect.
        coordinate_format: Bounding-box coordinate convention for
            detection serialization.
        box_format: Bounding-box layout for detection serialization.

    Returns:
        JSON string matching the task's Pydantic response model.

    Raises:
        ValueError: If the field is missing, empty, or incompatible
            with the task.
    """
    value = sample.get_field(label_field)
    if value is None:
        raise ValueError(
            f"Exemplar sample {sample.id} has empty/None value for "
            f"field '{label_field}'. All exemplar samples must have "
            f"the label field populated."
        )
    logger.debug("Serializing exemplar %s field '%s': %r", sample.id, label_field, type(value).__name__)

    if task in ("caption", "ocr"):
        return _serialize_text(value, sample.id, label_field)
    if task == "vqa":
        return _serialize_vqa(value, sample.id, label_field)
    if task == "classify":
        return _serialize_classify(value, sample.id, label_field, classes)
    if task == "tag":
        return _serialize_tag(value, sample.id, label_field, classes)
    if task == "detect":
        metadata = sample.metadata
        img_w = getattr(metadata, "width", None) if metadata else None
        img_h = getattr(metadata, "height", None) if metadata else None
        return _serialize_detect(
            value, sample.id, label_field, classes,
            coordinate_format, box_format,
            image_width=img_w, image_height=img_h,
        )
    raise ValueError(f"Unknown task: '{task}'")


def _extract_label_text(value: Any, sample_id: str, label_field: str) -> str:
    """Extract a string from a FiftyOne label or raw string value."""
    if isinstance(value, str):
        return value
    if isinstance(value, fo.Classification):
        return value.label
    raise ValueError(
        f"Exemplar sample {sample_id}: field '{label_field}' is type "
        f"'{type(value).__name__}', expected str or Classification."
    )


def _serialize_text(value: Any, sample_id: str, label_field: str) -> str:
    text = _extract_label_text(value, sample_id, label_field)
    return TextResponse(text=text).model_dump_json()


def _serialize_vqa(value: Any, sample_id: str, label_field: str) -> str:
    text = _extract_label_text(value, sample_id, label_field)
    return VQAResponse(answer=text).model_dump_json()


def _serialize_classify(
    value: Any, sample_id: str, label_field: str, classes: list[str] | None
) -> str:
    label = _extract_label_text(value, sample_id, label_field)
    if classes:
        model_cls = _constrained_classify_model(classes)
        return model_cls(label=label).model_dump_json()
    return ClassifyResponse(label=label).model_dump_json()


def _serialize_tag(
    value: Any, sample_id: str, label_field: str, classes: list[str] | None
) -> str:
    if isinstance(value, str):
        labels = [s.strip() for s in value.split(",") if s.strip()]
    elif isinstance(value, fo.Classification):
        labels = [value.label]
    elif isinstance(value, fo.Classifications):
        labels = [c.label for c in value.classifications]
    else:
        raise ValueError(
            f"Exemplar sample {sample_id}: field '{label_field}' is type "
            f"'{type(value).__name__}', expected str, Classification, "
            f"or Classifications."
        )
    if classes:
        model_cls = _constrained_tag_model(classes)
        return model_cls(labels=labels).model_dump_json()
    return TagResponse(labels=labels).model_dump_json()


def _serialize_detect(
    value: Any,
    sample_id: str,
    label_field: str,
    classes: list[str] | None,
    coordinate_format: str,
    box_format: str,
    image_width: float | None = None,
    image_height: float | None = None,
) -> str:
    """Serialize Detections to DetectResponse JSON.

    FiftyOne stores bounding boxes as normalized [0,1] ``[x, y, w, h]``.
    Conversion to the target format is handled by ``_fo_to_run_format``.

    Args:
        value: The FiftyOne Detections label value.
        sample_id: Sample ID for error messages.
        label_field: Field name for error messages.
        classes: Optional class constraint list.
        coordinate_format: Target coordinate convention.
        box_format: Target box layout.
        image_width: Original image width (required for ``"pixel"``
            coordinate format).
        image_height: Original image height (required for ``"pixel"``
            coordinate format).
    """
    if not isinstance(value, fo.Detections):
        raise ValueError(
            f"Exemplar sample {sample_id}: field '{label_field}' is type "
            f"'{type(value).__name__}', expected Detections."
        )
    if coordinate_format == "pixel" and (not image_width or not image_height):
        raise ValueError(
            f"Exemplar sample {sample_id}: pixel coordinate format requires "
            f"image dimensions but metadata is missing. Ensure "
            f"compute_metadata() has been called."
        )
    items = []
    for det in value.detections:
        box = _fo_to_run_format(
            det.bounding_box, coordinate_format, box_format,
            image_width=image_width, image_height=image_height,
        )
        items.append(DetectionItem(label=det.label, box=box))

    if classes:
        model_cls = _constrained_detect_model(classes)
        return model_cls(
            detections=[{"label": it.label, "box": it.box} for it in items]
        ).model_dump_json()
    return DetectResponse(detections=items).model_dump_json()


def _fo_to_run_format(
    bbox: list[float],
    coordinate_format: str,
    box_format: str,
    image_width: float | None = None,
    image_height: float | None = None,
) -> list[float]:
    """Convert FiftyOne ``[x, y, w, h]`` (normalized 0-1) to the run format.

    Args:
        bbox: FiftyOne bounding box as ``[x, y, width, height]`` in
            normalized ``[0, 1]`` coordinates.
        coordinate_format: Target coordinate system
            (``"pixel"``, ``"normalized_1"``, ``"normalized_1000"``).
        box_format: Target box layout
            (``"xyxy"``, ``"xywh"``, ``"cxcywh"``).
        image_width: Original image width in pixels.  Required when
            *coordinate_format* is ``"pixel"``.
        image_height: Original image height in pixels.  Required when
            *coordinate_format* is ``"pixel"``.

    Returns:
        Bounding box in the target format as a list of floats.
    """
    x, y, w, h = bbox

    # Scale to target coordinate system
    if coordinate_format == "pixel":
        x = x * image_width
        y = y * image_height
        w = w * image_width
        h = h * image_height
    elif coordinate_format == "normalized_1000":
        x, y, w, h = x * 1000, y * 1000, w * 1000, h * 1000
    # normalized_1: no scaling needed (already [0, 1])

    # Convert to target box format
    if box_format == "xyxy":
        return [x, y, x + w, y + h]
    if box_format == "xywh":
        return [x, y, w, h]
    if box_format == "cxcywh":
        return [x + w / 2, y + h / 2, w, h]
    return [x, y, x + w, y + h]  # default xyxy


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def build_exemplar_messages(
    exemplar_view: fo.DatasetView,
    label_field: str,
    task: str,
    classes: list[str] | None = None,
    coordinate_format: str = "pixel",
    box_format: str = "xyxy",
    image_detail: str = "auto",
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Build user/assistant message pairs for all exemplar samples.

    Each exemplar produces two messages: a ``user`` message containing
    the image and reference framing text, and an ``assistant`` message
    containing the serialized expected output in the task's response
    model JSON format.

    Args:
        exemplar_view: FiftyOne view of exemplar samples.
        label_field: Name of the label field to serialize from each
            sample.
        task: Task identifier for serialization.
        classes: Optional class constraint list.
        coordinate_format: Bounding-box coordinate convention for
            detection serialization.
        box_format: Bounding-box layout for detection serialization.
        image_detail: OpenAI image detail level (``"low"``, ``"high"``,
            or ``"auto"``).
        max_workers: Thread pool size for parallel image encoding.

    Returns:
        Flat list of message dicts (alternating user/assistant).

    Raises:
        ValueError: If any exemplar sample fails to serialize or encode.
    """
    # Compute metadata for detection pixel coords (reads file headers only)
    if task == "detect" and coordinate_format == "pixel":
        exemplar_view.compute_metadata()

    filepaths: list[str] = exemplar_view.values("filepath")
    samples: list[fo.Sample] = list(exemplar_view)

    # Encode all exemplar images at once
    image_contents = build_image_contents(
        filepaths, max_workers=max_workers, image_detail=image_detail,
    )

    messages: list[dict[str, Any]] = []
    for sample, image_content in zip(samples, image_contents):
        # Serialize the label field to response model JSON
        json_str = serialize_exemplar(
            sample=sample,
            label_field=label_field,
            task=task,
            classes=classes,
            coordinate_format=coordinate_format,
            box_format=box_format,
        )

        # User message: image + reference framing
        messages.append({
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": _EXEMPLAR_USER_TEXT},
            ],
        })
        # Assistant message: expected output JSON
        messages.append({
            "role": "assistant",
            "content": json_str,
        })

    logger.info(
        "Built %d exemplar message pairs (%d samples)",
        len(messages) // 2, len(samples),
    )
    return messages
