"""TaskConfig: Pydantic response models, prompts, structured output schemas,
and post-generation validation for all vision inference tasks.

Uses the OpenAI Responses API (``client.responses.parse()``) for
structured output with Pydantic models.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import fiftyone as fo
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# -- Pydantic response models --


class TextResponse(BaseModel):
    """Response model for caption and OCR tasks."""

    text: str


class ClassifyResponse(BaseModel):
    """Response model for open-ended classification."""

    label: str


class TagResponse(BaseModel):
    """Response model for open-ended tagging."""

    labels: list[str]


class VQAResponse(BaseModel):
    """Response model for visual question answering."""

    answer: str


class DetectionItem(BaseModel):
    """A single detection with label and bounding box."""

    label: str
    box: list[float]


class DetectResponse(BaseModel):
    """Response model for object detection."""

    detections: list[DetectionItem]


# -- Dynamic constrained model builders --


@lru_cache(maxsize=8)
def _constrained_classify_model(classes: tuple[str, ...]) -> type[BaseModel]:
    """Build a Pydantic model with ``label`` constrained to *classes*."""
    from typing import Literal

    literal_type = Literal[classes]  # type: ignore[valid-type]
    return create_model("ClassifyConstrained", label=(literal_type, ...))


@lru_cache(maxsize=8)
def _constrained_tag_model(classes: tuple[str, ...]) -> type[BaseModel]:
    """Build a Pydantic model with ``labels`` constrained to *classes*."""
    from typing import Literal

    literal_type = Literal[classes]  # type: ignore[valid-type]
    return create_model("TagConstrained", labels=(list[literal_type], ...))


@lru_cache(maxsize=8)
def _constrained_detect_model(classes: tuple[str, ...]) -> type[BaseModel]:
    """Build a Pydantic model with detection labels constrained to *classes*."""
    from typing import Literal

    literal_type = Literal[classes]  # type: ignore[valid-type]
    item = create_model(
        "DetItem", label=(literal_type, ...), box=(list[float], ...)
    )
    return create_model("DetectConstrained", detections=(list[item], ...))


# -- Estimated output tokens per task (for cost preview) --

OUTPUT_TOKEN_ESTIMATES: dict[str, int] = {
    "caption": 80,
    "classify": 15,
    "tag": 40,
    "detect": 200,
    "vqa": 60,
    "ocr": 100,
}

# Image token counts by OpenAI detail level
IMAGE_TOKEN_COUNTS: dict[str, int] = {
    "low": 85,
    "high": 765,
    "auto": 765,
}

# Base prompt text token estimate (system + user prompt text, no image)
PROMPT_TEXT_TOKENS: int = 50

# Bumped prompt token estimate when few-shot exemplars are active
# (accounts for the preamble text added to the system prompt)
PROMPT_TEXT_TOKENS_FEWSHOT: int = 70

# Estimated tokens per exemplar (framing text + serialized JSON)
EXEMPLAR_TEXT_TOKENS: int = 30

# Few-shot preamble prepended to the system prompt when exemplars are
# provided.  The ``{task_verb}`` placeholder is replaced with the
# task-specific action (e.g. "classify", "caption", "detect objects in").
_FEWSHOT_PREAMBLE: str = (
    "You will first be shown REFERENCE EXAMPLES. These examples "
    "demonstrate the EXACT expected output format and labeling quality. "
    "DO NOT describe or analyze the example images. They are provided "
    "ONLY as reference for how you should respond to the FINAL image. "
    "After the examples, you will receive one image to {task_verb}. "
    "Apply the same approach shown in the examples.\n\n"
)

_TASK_VERBS: dict[str, str] = {
    "caption": "caption",
    "classify": "classify",
    "tag": "tag",
    "detect": "detect objects in",
    "vqa": "answer a question about",
    "ocr": "extract text from",
}


class TaskConfig:
    """Builds prompts, Pydantic response models, and parses LLM responses."""

    TASKS: dict[str, dict[str, Any]] = {
        "caption": {
            "system": (
                "You are an image captioner. Respond with a JSON object:"
                ' {"text": "your description"}'
            ),
            "prompt": "Describe this image concisely.",
            "output_type": "Classification",
            "default_field": "caption",
            "default_temperature": 0.2,
        },
        "classify": {
            "system": (
                "You are an image classifier. Respond with exactly one"
                " class label."
            ),
            "system_open": (
                "You are an image classifier. Respond with a JSON object:"
                ' {"label": "your label"}'
            ),
            "prompt": "Classify this image. Choose exactly one: {classes}",
            "prompt_open": (
                "Classify this image with the single most appropriate label."
            ),
            "output_type": "Classification",
            "default_field": "classification",
            "default_temperature": 0.0,
        },
        "tag": {
            "system": (
                "You are an image tagger. Respond with a JSON object:"
                ' {"labels": ["tag1", "tag2", ...]}'
            ),
            "prompt": "Tag this image with all applicable labels from: {classes}",
            "prompt_open": "Tag this image with all applicable descriptive labels.",
            "output_type": "Classifications",
            "default_field": "tags",
            "default_temperature": 0.0,
        },
        "detect": {
            "output_type": "Detections",
            "default_field": "detections",
            "default_temperature": 0.0,
        },
        "vqa": {
            "system": (
                "You are a visual question answerer. Respond with a JSON"
                ' object: {"answer": "your answer"}'
            ),
            "prompt": "{question}",
            "output_type": "Classification",
            "default_field": "vqa_answer",
            "default_temperature": 0.2,
        },
        "ocr": {
            "system": (
                "You are an OCR engine. Respond with a JSON object:"
                ' {"text": "extracted text"}'
            ),
            "prompt": "Extract all text visible in this image.",
            "output_type": "Classification",
            "default_field": "ocr_text",
            "default_temperature": 0.0,
        },
    }

    _BOX_FORMATS: dict[str, str] = {
        "xyxy": "[x_min, y_min, x_max, y_max]",
        "xywh": "[x, y, width, height]",
        "cxcywh": "[cx, cy, width, height]",
    }

    _COORD_DESCS: dict[str, str] = {
        "normalized_1000": (
            "integer coordinates on a 0-1000 scale."
            " x-axis: 0=left edge, 1000=right edge."
            " y-axis: 0=top edge, 1000=bottom edge."
            " All values MUST be integers between 0 and 1000"
        ),
        "normalized_1": (
            "floating-point coordinates on a 0.0-1.0 scale."
            " x-axis: 0.0=left edge, 1.0=right edge."
            " y-axis: 0.0=top edge, 1.0=bottom edge."
            " All values MUST be between 0.0 and 1.0"
        ),
        "pixel": (
            "pixel coordinates relative to the ORIGINAL image dimensions."
            " x-axis: 0=left edge, image_width=right edge."
            " y-axis: 0=top edge, image_height=bottom edge."
            " All values MUST be non-negative integers"
        ),
    }

    def __init__(
        self,
        task: str,
        prompt: str | None = None,
        system_prompt: str | None = None,
        classes: list[str] | None = None,
        coordinate_format: str = "pixel",
        box_format: str = "xyxy",
        **template_kwargs: str,
    ) -> None:
        """Initialise a task configuration with prompts and constraints.

        Args:
            task: Task identifier (one of ``TASKS`` keys).
            prompt: Custom user prompt override.  Uses the task default when
                ``None``.
            system_prompt: Custom system prompt override.
            classes: Class labels for classify / tag / detect.  ``None``
                activates the open-ended variant for classify and tag.
            coordinate_format: Bounding-box coordinate convention.
                ``"pixel"`` (default, recommended) uses integer pixel
                coordinates relative to the original image dimensions,
                which are included in each prompt.
                ``"normalized_1000"`` uses 0-1000 integer scale.
                ``"normalized_1"`` uses 0.0-1.0 float coordinates.
            box_format: Bounding-box layout
                (``"xyxy"``, ``"xywh"``, or ``"cxcywh"``).
            **template_kwargs: Extra format kwargs for prompt templates
                (e.g. ``question`` for VQA).
        """
        if task not in self.TASKS:
            raise ValueError(
                f"Unknown task: {task}. Must be one of {list(self.TASKS)}"
            )

        defaults = self.TASKS[task]
        self.task: str = task
        self.classes: list[str] | None = classes
        self.output_type: str = defaults["output_type"]
        self.default_field: str = defaults["default_field"]
        self.default_temperature: float = defaults["default_temperature"]
        self.coordinate_format: str = coordinate_format
        self.box_format: str = box_format

        if task == "detect":
            coord_desc = self._COORD_DESCS.get(
                coordinate_format, self._COORD_DESCS["normalized_1"]
            )
            box_labels = self._BOX_FORMATS.get(
                box_format, self._BOX_FORMATS["xyxy"]
            )

            default_system = (
                "You are a precise object detector."
                " Respond with a JSON object:"
                ' {"detections": [{"label": "...", "box": '
                + box_labels
                + "}, ...]}.\n"
                "Coordinate system: " + coord_desc + "."
                " Coordinates MUST be relative to the ORIGINAL image"
                " dimensions provided with each image, NOT any internally"
                " resized or padded version."
                " Ensure each bounding box tightly fits the detected"
                " object."
            )
            default_prompt = "Detect all objects in this image."
            default_prompt_with_classes = (
                "Detect these objects in this image: {classes}."
                " For each object, return its label and bounding box"
                " as " + box_labels + " in " + coord_desc + "."
            )
        else:
            open_ended = task in ("classify", "tag") and not classes
            default_system = defaults.get(
                "system_open" if open_ended else "system", ""
            )
            default_prompt = defaults.get(
                "prompt_open" if open_ended else "prompt", ""
            )
            default_prompt_with_classes = None

        self.system_prompt: str = (
            system_prompt if system_prompt is not None else default_system
        )

        if prompt is not None:
            raw_prompt = prompt
        elif task == "detect" and classes and default_prompt_with_classes:
            raw_prompt = default_prompt_with_classes
        else:
            raw_prompt = default_prompt

        fmt_kwargs: dict[str, str] = {**template_kwargs}
        if classes:
            fmt_kwargs["classes"] = ", ".join(classes)
        self.prompt: str = (
            raw_prompt.format(**fmt_kwargs) if fmt_kwargs else raw_prompt
        )

    def get_instructions(
        self,
        exemplar_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return the instructions string for the Responses API.

        When *exemplar_messages* are provided the few-shot preamble is
        prepended to the system prompt.  This value maps directly to the
        ``instructions`` parameter of ``client.responses.parse()``.
        """
        system = self.system_prompt
        if exemplar_messages and system:
            verb = _TASK_VERBS.get(self.task, self.task)
            system = _FEWSHOT_PREAMBLE.format(task_verb=verb) + system
        return system

    def build_input(
        self,
        image_content: dict[str, Any],
        exemplar_messages: list[dict[str, Any]] | None = None,
        image_width: float | None = None,
        image_height: float | None = None,
    ) -> list[dict[str, Any]]:
        """Build the ``input`` array for the Responses API.

        Returns a list of message dicts containing optional exemplar
        pairs followed by the final user query.  The system prompt is
        handled separately via :meth:`get_instructions`.

        Args:
            image_content: Image content dict (base64 or URL) for the
                target sample.
            exemplar_messages: Optional list of pre-built user/assistant
                message pairs from :func:`exemplars.build_exemplar_messages`.
            image_width: Original image width in pixels.  Included in
                the user message for detection tasks so the model can
                ground coordinates to the original image dimensions.
            image_height: Original image height in pixels.
        """
        messages: list[dict[str, Any]] = []

        # Exemplar pairs (user image + assistant JSON) if provided
        if exemplar_messages:
            messages.extend(exemplar_messages)

        # Build user prompt text — prepend image dimensions for detection
        # to help the model ground coordinates to the original image
        prompt_text = self.prompt
        if self.task == "detect" and image_width and image_height:
            w, h = int(image_width), int(image_height)
            orientation = (
                "landscape" if w > h else "portrait" if h > w else "square"
            )
            prompt_text = (
                f"Original image: {w}x{h} pixels ({orientation}).\n"
                f"{prompt_text}"
            )

        # Final user message — the actual target image to process
        messages.append(
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "input_text", "text": prompt_text},
                ],
            }
        )
        return messages

    def get_response_model(self) -> type[BaseModel]:
        """Return the Pydantic model class for structured output.

        When *classes* are set, returns a dynamically-built model whose
        ``label`` field carries a ``Literal`` constraint that produces a
        JSON-schema ``enum``.
        """
        if self.task == "classify":
            if self.classes:
                return _constrained_classify_model(tuple(self.classes))
            return ClassifyResponse

        if self.task == "tag":
            if self.classes:
                return _constrained_tag_model(tuple(self.classes))
            return TagResponse

        if self.task == "detect":
            if self.classes:
                return _constrained_detect_model(tuple(self.classes))
            return DetectResponse

        if self.task == "vqa":
            return VQAResponse

        # caption, ocr
        return TextResponse

    def estimated_output_tokens(self) -> int:
        """Return a rough token-count estimate for cost previews."""
        return OUTPUT_TOKEN_ESTIMATES.get(self.task, 60)

    # -- Output parsing --

    def parse_response(
        self,
        parsed: BaseModel,
        image_width: float | None = None,
        image_height: float | None = None,
    ) -> fo.Classification | fo.Classifications | fo.Detections:
        """Convert a parsed Pydantic model into a FiftyOne label.

        The OpenAI SDK's ``responses.parse()`` returns
        ``output_parsed`` as an already-validated Pydantic instance.
        This method converts it to the corresponding FiftyOne label type.
        """
        if self.task in ("caption", "ocr"):
            return fo.Classification(label=parsed.text)

        if self.task == "vqa":
            return fo.Classification(label=parsed.answer)

        if self.task == "classify":
            return fo.Classification(label=parsed.label)

        if self.task == "tag":
            return fo.Classifications(
                classifications=[
                    fo.Classification(label=label) for label in parsed.labels
                ]
            )

        if self.task == "detect":
            return self._parse_detections(
                parsed, image_width=image_width, image_height=image_height
            )

        raise ValueError(f"Unknown task: {self.task}")

    def _parse_detections(
        self,
        parsed: DetectResponse,
        image_width: float | None = None,
        image_height: float | None = None,
    ) -> fo.Detections:
        """Validate detection coordinates and convert to FiftyOne format.

        The Pydantic model guarantees structure; this validates what
        schemas cannot enforce (coordinate ranges, degenerate boxes).
        """
        detections: list[fo.Detection] = []
        raw = parsed.detections

        for det in raw:
            if len(det.box) != 4:
                continue

            result = _convert_box(
                *[float(v) for v in det.box],
                coordinate_format=self.coordinate_format,
                box_format=self.box_format,
                img_w=image_width,
                img_h=image_height,
            )
            if result is None:
                continue

            detections.append(
                fo.Detection(label=det.label, bounding_box=list(result))
            )

        if len(detections) < len(raw):
            logger.warning(
                "%d/%d detections dropped (bad length or degenerate box)",
                len(raw) - len(detections),
                len(raw),
            )

        return fo.Detections(detections=detections)


_COORD_SCALE: dict[str, float] = {
    "normalized_1000": 1000.0,
    "normalized_1": 1.0,
}


def _convert_box(
    v0: float,
    v1: float,
    v2: float,
    v3: float,
    coordinate_format: str,
    box_format: str = "xyxy",
    img_w: float | None = None,
    img_h: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Convert model box output to FiftyOne ``[x, y, w, h]`` in ``[0, 1]``.

    Two-step pipeline:

    A. Convert from *box_format* to internal ``xyxy``.
    B. Normalise to ``[0, 1]`` based on *coordinate_format*.

    Returns ``None`` for degenerate boxes or missing required dimensions.
    """
    # Step A: convert to xyxy
    if box_format == "xywh":
        x1, y1, x2, y2 = v0, v1, v0 + v2, v1 + v3
    elif box_format == "cxcywh":
        x1, y1 = v0 - v2 / 2, v1 - v3 / 2
        x2, y2 = v0 + v2 / 2, v1 + v3 / 2
    else:  # xyxy
        x1, y1, x2, y2 = v0, v1, v2, v3

    # Step B: resolve per-axis scale and normalise to [0, 1] xywh
    scale = _COORD_SCALE.get(coordinate_format)
    if scale is not None:
        max_x = max_y = scale
    elif coordinate_format == "pixel":
        if img_w is None or img_h is None:
            logger.warning(
                "Pixel coordinates require image dimensions; skipping box"
            )
            return None
        max_x, max_y = float(img_w), float(img_h)
    else:
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    x1 = max(0.0, min(x1, max_x))
    y1 = max(0.0, min(y1, max_y))
    x2 = max(0.0, min(x2, max_x))
    y2 = max(0.0, min(y2, max_y))
    if x2 <= x1 or y2 <= y1:
        return None
    x = x1 / max_x
    y = y1 / max_y
    w = min((x2 - x1) / max_x, 1.0 - x)
    h = min((y2 - y1) / max_y, 1.0 - y)
    return x, y, w, h
