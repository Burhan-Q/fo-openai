"""TaskConfig: Pydantic response models, prompts, structured output schemas,
and post-generation validation for all vision inference tasks."""

import logging
from typing import Literal

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


def _constrained_classify_model(classes: list[str]):
    """Build a Pydantic model with label constrained to specific classes."""
    literal_type = Literal[tuple(classes)]
    return create_model("ClassifyConstrained", label=(literal_type, ...))


def _constrained_tag_model(classes: list[str]):
    """Build a Pydantic model with labels constrained to specific classes."""
    literal_type = Literal[tuple(classes)]
    return create_model("TagConstrained", labels=(list[literal_type], ...))


def _constrained_detect_model(classes: list[str]):
    """Build a Pydantic model with detection labels constrained."""
    literal_type = Literal[tuple(classes)]
    item = create_model("DetItem", label=(literal_type, ...), box=(list[float], ...))
    return create_model("DetectConstrained", detections=(list[item], ...))


# -- Estimated output tokens per task (for cost preview) --

_OUTPUT_TOKEN_ESTIMATES = {
    "caption": 80,
    "classify": 15,
    "tag": 40,
    "detect": 200,
    "vqa": 60,
    "ocr": 100,
}


class TaskConfig:
    """Builds prompts, Pydantic response models, and parses LLM responses."""

    TASKS = {
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
                "You are an image classifier. Respond with exactly one class label."
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

    _BOX_FORMATS = {
        "xyxy": {"labels": "[x_min, y_min, x_max, y_max]"},
        "xywh": {"labels": "[x, y, width, height]"},
        "cxcywh": {"labels": "[cx, cy, width, height]"},
    }

    _COORD_FORMATS = {
        "normalized_1000": {
            "desc": (
                "0-1000 normalized coordinates where 0 is top-left"
                " and 1000 is bottom-right"
            ),
            "item_schema": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
        },
        "normalized_1": {
            "desc": (
                "0-1 normalized coordinates where 0.0 is top-left"
                " and 1.0 is bottom-right"
            ),
            "item_schema": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "pixel": {
            "desc": "pixel coordinates",
            "item_schema": {"type": "number", "minimum": 0},
        },
    }

    def __init__(
        self,
        task,
        prompt=None,
        system_prompt=None,
        classes=None,
        coordinate_format="normalized_1",
        box_format="xyxy",
        **template_kwargs,
    ):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Must be one of {list(self.TASKS)}")

        defaults = self.TASKS[task]
        self.task = task
        self.classes = classes
        self.output_type = defaults["output_type"]
        self.default_field = defaults["default_field"]
        self.default_temperature = defaults["default_temperature"]
        self.coordinate_format = coordinate_format
        self.box_format = box_format

        if task == "detect":
            coord = self._COORD_FORMATS.get(
                coordinate_format, self._COORD_FORMATS["normalized_1"]
            )
            coord_desc = coord["desc"]
            box_fmt = self._BOX_FORMATS.get(box_format, self._BOX_FORMATS["xyxy"])
            box_labels = box_fmt["labels"]

            default_system = (
                "You are an object detector. Respond with a JSON object:"
                ' {"detections": [{"label": "...", "box": '
                + box_labels
                + "}, ...]}. Use "
                + coord_desc
                + "."
            )
            default_prompt = "Detect all objects in this image."
            default_prompt_with_classes = (
                "Detect these objects in this image: {classes}."
                " For each object, return its label and bounding box"
                " as " + box_labels + " in " + coord_desc + "."
            )
        else:
            open_ended = task in ("classify", "tag") and not classes
            default_system = defaults.get("system_open" if open_ended else "system", "")
            default_prompt = defaults.get("prompt_open" if open_ended else "prompt", "")
            default_prompt_with_classes = None

        self.system_prompt = (
            system_prompt if system_prompt is not None else default_system
        )

        if prompt is not None:
            raw_prompt = prompt
        elif task == "detect" and classes and default_prompt_with_classes:
            raw_prompt = default_prompt_with_classes
        else:
            raw_prompt = default_prompt

        fmt_kwargs = {**template_kwargs}
        if classes:
            fmt_kwargs["classes"] = ", ".join(classes)
        self.prompt = raw_prompt.format(**fmt_kwargs) if fmt_kwargs else raw_prompt

    def build_messages(self, image_content):
        """Build OpenAI-format messages for one image."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": self.prompt},
                ],
            }
        )
        return messages

    def get_response_model(self):
        """Return the Pydantic model class for structured output.

        If classes are provided, returns a dynamically constrained model
        with Literal type annotations that produce JSON schema enum
        constraints.
        """
        if self.task == "classify":
            if self.classes:
                return _constrained_classify_model(self.classes)
            return ClassifyResponse

        if self.task == "tag":
            if self.classes:
                return _constrained_tag_model(self.classes)
            return TagResponse

        if self.task == "detect":
            if self.classes:
                return _constrained_detect_model(self.classes)
            return DetectResponse

        if self.task == "vqa":
            return VQAResponse

        # caption, ocr
        return TextResponse

    def estimated_output_tokens(self):
        """Rough estimate of output tokens for cost preview."""
        return _OUTPUT_TOKEN_ESTIMATES.get(self.task, 60)

    # -- Output parsing --

    def parse_response(self, text, image_width=None, image_height=None):
        """Parse LLM response into a FiftyOne label.

        Uses Pydantic model_validate_json for validation, then converts
        to the appropriate FiftyOne label type.
        """
        model_cls = self.get_response_model()
        parsed = model_cls.model_validate_json(text)

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

    def _parse_detections(self, parsed, image_width=None, image_height=None):
        """Post-generation validation for detection output.

        The Pydantic model guarantees structure. This validates what
        schemas cannot enforce: coordinate ranges, degenerate boxes.
        """
        detections = []
        raw = parsed.detections

        for det in raw:
            box = det.box
            if len(box) != 4:
                continue

            result = _convert_box(
                *[float(v) for v in box],
                coordinate_format=self.coordinate_format,
                box_format=self.box_format,
                img_w=image_width,
                img_h=image_height,
            )
            if result is None:
                continue

            detections.append(
                fo.Detection(
                    label=det.label,
                    bounding_box=list(result),
                )
            )

        if len(detections) < len(raw):
            logger.warning(
                "%d/%d detections dropped (bad length or degenerate box)",
                len(raw) - len(detections),
                len(raw),
            )

        return fo.Detections(detections=detections)


_COORD_SCALE = {"normalized_1000": 1000.0, "normalized_1": 1.0}


def _convert_box(
    v0, v1, v2, v3, coordinate_format, box_format="xyxy", img_w=None, img_h=None
):
    """Convert model box output to FiftyOne [x, y, w, h] in [0, 1].

    Two-step pipeline:
      A. Convert from model box_format to internal xyxy.
      B. Normalize to [0, 1] based on coordinate_format.

    Returns None if degenerate or missing required dimensions.
    """
    # Step A: convert to xyxy
    if box_format == "xywh":
        x1, y1, x2, y2 = v0, v1, v0 + v2, v1 + v3
    elif box_format == "cxcywh":
        x1, y1, x2, y2 = v0 - v2 / 2, v1 - v3 / 2, v0 + v2 / 2, v1 + v3 / 2
    else:  # xyxy
        x1, y1, x2, y2 = v0, v1, v2, v3

    # Step B: resolve per-axis scale and normalize to [0, 1] xywh
    scale = _COORD_SCALE.get(coordinate_format)
    if scale is not None:
        max_x = max_y = scale
    elif coordinate_format == "pixel":
        if img_w is None or img_h is None:
            logger.warning("Pixel coordinates require image dimensions; skipping box")
            return None
        max_x, max_y = float(img_w), float(img_h)
    else:
        # Unknown format fallback — no normalization
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
