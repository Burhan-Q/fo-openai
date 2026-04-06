"""Image loading/encoding utilities and config persistence for OpenAI."""

from __future__ import annotations

import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_PERSIST_KEYS: list[str] = [
    "model",
    "base_url",
    "api_key",
    "task",
    "classes",
    "question",
    "prompt",
    "system_prompt",
    "prompt_override",
    "temperature",
    "max_output_tokens",
    "top_p",
    "batch_size",
    "max_concurrent",
    "max_workers",
    "timeout",
    "image_detail",
    "coordinate_format",
    "box_format",
    "log_metadata",
    "enable_logging",
    "log_level",
    "log_file",
    # Exemplar settings
    "exemplars_enabled",
    "exemplar_source",
    "exemplar_view_name",
    "exemplar_sample_ids",
    "exemplar_tag",
    "exemplar_field_name",
    "exemplar_field_value",
    "exemplar_label_field",
]


def normalize_classes(raw: str | list[Any] | None) -> list[str] | None:
    """Convert *raw* (comma-separated string, list, or ``None``) to a
    deduplicated list of stripped strings, or ``None`` if empty."""
    if not raw:
        return None
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    return [c.strip() for c in raw.split(",") if c.strip()] or None


def pick_params(
    params: dict[str, Any], exclude: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Filter *params* to persistable keys, dropping ``None`` values.

    Classes are normalised to a list for consistent storage.
    """
    out: dict[str, Any] = {}
    for k in _PERSIST_KEYS:
        if k in params and k not in exclude and params[k] is not None:
            v = normalize_classes(params[k]) if k == "classes" else params[k]
            if v is not None:
                out[k] = v
    return out


def _global_store() -> Any:
    """Return the cross-dataset ``ExecutionStore`` for plugin config."""
    from fiftyone.operators.store import ExecutionStore

    return ExecutionStore.create("openai_config", dataset_id=None)


def get_global_config() -> dict[str, Any]:
    """Read global config from the cross-dataset ``ExecutionStore``."""
    try:
        cfg = _global_store().get("config")
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def save_global_config(params: dict[str, Any]) -> None:
    """Persist *params* to the cross-dataset ``ExecutionStore``."""
    try:
        _global_store().set("config", pick_params(params))
    except Exception:
        pass


def clear_global_config() -> None:
    """Delete the cross-dataset config key."""
    try:
        _global_store().delete("config")
    except Exception:
        pass


def parse_config_json(
    json_str: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON string and filter to known config keys.

    Returns ``(config_dict, None)`` on success or ``(None, error_msg)``
    on failure.
    """
    try:
        raw = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(raw, dict):
        return None, "JSON must be an object"
    return {k: raw[k] for k in _PERSIST_KEYS if k in raw}, None


def build_image_contents(
    filepaths: list[str],
    max_workers: int = 4,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Build image-content dicts for the OpenAI Responses API.

    HTTP(S) URLs are passed through directly.  Local files are
    base64-encoded in parallel.  The *image_detail* level (``"low"``,
    ``"high"``, or ``"auto"``) is set on every resulting
    ``input_image`` dict.
    """
    results: list[dict[str, Any]] = [{}] * len(filepaths)
    to_encode: list[int] = []

    for i, fp in enumerate(filepaths):
        if fp.startswith(("http://", "https://")):
            results[i] = _url_content(fp)
        else:
            to_encode.append(i)

    if to_encode:
        paths = [filepaths[i] for i in to_encode]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            encoded = list(pool.map(_encode_base64, paths))
        for idx, enc in zip(to_encode, encoded):
            results[idx] = enc

    # Apply detail level uniformly
    for item in results:
        item["detail"] = image_detail

    return results


def _url_content(url: str) -> dict[str, Any]:
    """Wrap an HTTP(S) URL as an ``input_image`` content dict."""
    return {"type": "input_image", "image_url": url}


def _encode_base64(filepath: str) -> dict[str, Any]:
    """Read and base64-encode a single image file."""
    mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{b64}",
    }
