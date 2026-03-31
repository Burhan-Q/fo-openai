# Detection Task

Object detection with bounding boxes. The most complex task due to coordinate system conversion.

## Coordinate Formats

Available in the advanced settings dropdown (detect task only):

| Format | Description | UI Exposed |
|--------|-------------|-----------|
| `normalized_1` | 0.0–1.0 relative coordinates | Yes (default) |
| `normalized_1000` | 0–1000 integer coordinates | Yes |
| `pixel` | Absolute pixel coordinates | **No** — removed from UI |

**Why pixel is excluded from UI:** OpenAI resizes images internally before processing. The model returns bounding box coordinates relative to its resized version, not the original image dimensions. There is no reliable way to reverse this scaling, so pixel coordinates would be incorrect. The `pixel` format is retained in code (for `_convert_box`) but not exposed in the operator form.

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

The system prompt is dynamically constructed based on coordinate and box format:
```
"You are an object detector. Respond with a JSON object:
 {"detections": [{"label": "...", "box": [x_min, y_min, x_max, y_max]}, ...]}.
 Use 0-1 normalized coordinates where 0.0 is top-left and 1.0 is bottom-right."
```

When classes are provided, the user prompt includes them:
```
"Detect these objects in this image: car, truck.
 For each object, return its label and bounding box as [x_min, y_min, x_max, y_max]
 in 0-1 normalized coordinates..."
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
