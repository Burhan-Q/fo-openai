# Detection Task

Object detection with bounding boxes. The most complex task due to coordinate system conversion.

## Coordinate Formats

Available in the Task tab's detection format selectors:

| Format | Description | Default |
|--------|-------------|---------|
| `pixel` | Integer pixel coordinates grounded to original image dimensions | **Yes (Recommended)** |
| `normalized_1000` | 0–1000 integer coordinates | No |
| `normalized_1` | 0.0–1.0 floating-point coordinates | No |

**Why pixel is the default:** Integer coordinates align well with LLM tokenizers (a float like ``0.3741`` costs multiple tokens and is rare in training data, while ``374`` is 1-2 tokens and common). The model receives the original image dimensions in the prompt ("Original image: WxH pixels (landscape/portrait/square)"), giving it concrete grounding. This approach matches industry practice (Gemini, Qwen-VL, OpenAI fine-tuning). Requires ``compute_metadata()`` (reads file headers only, fast).

**Note on accuracy:** General-purpose LLMs are not architecturally designed for precise spatial localization. Bounding box positions may be offset, especially on non-square images. For precision-critical detection, consider fine-tuned models or dedicated detectors.

## Box Formats

| Format | Layout | Conversion to xyxy |
|--------|--------|-------------------|
| `xyxy` | `[x_min, y_min, x_max, y_max]` | Identity |
| `xywh` | `[x, y, width, height]` | `x2 = x + w`, `y2 = y + h` |
| `cxcywh` | `[cx, cy, width, height]` | `x1 = cx - w/2`, etc. |

## Conversion Pipeline (`_convert_box`)

Two-step process:

1. **Box format → xyxy**: Convert from the model's box format to internal `(x1, y1, x2, y2)`
2. **Coordinate normalization → [0,1]**: Divide by scale factor, clamp to valid range, convert to FiftyOne `[x, y, w, h]` format

FiftyOne expects bounding boxes as `[x, y, w, h]` where all values are in `[0, 1]` relative to image dimensions.

**Degenerate box filtering:** Boxes where `x2 <= x1` or `y2 <= y1` after conversion are silently dropped with a warning logged.

## Detection Prompts

The system prompt is dynamically constructed based on coordinate and box format. For the default pixel + xyxy configuration:
```
"You are a precise object detector. Respond with a JSON object:
 {"detections": [{"label": "...", "box": [x_min, y_min, x_max, y_max]}, ...]}.
 Coordinate system: pixel coordinates relative to the ORIGINAL image dimensions.
 Coordinates MUST be relative to the ORIGINAL image dimensions provided with
 each image, NOT any internally resized or padded version.
 Ensure each bounding box tightly fits the detected object."
```

The user prompt includes image dimensions for detection tasks:
```
"Original image: 1920x1080 pixels (landscape).
 Detect all objects in this image."
```

When classes are provided, the user prompt includes them:
```
"Original image: 1920x1080 pixels (landscape).
 Detect these objects in this image: car, truck.
 For each object, return its label and bounding box as [x_min, y_min, x_max, y_max]
 in pixel coordinates relative to the ORIGINAL image dimensions."
```

## Pydantic Model

```python
class DetectionItem(BaseModel):
    label: str
    box: list[float]  # 4 elements

class DetectResponse(BaseModel):
    detections: list[DetectionItem]
```

With class constraints, `label` becomes `Literal["car", "truck", ...]`.
