"""Image loading/encoding utilities and config persistence for OpenAI."""

import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor

_PERSIST_KEYS = [
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
    "max_tokens",
    "top_p",
    "seed",
    "batch_size",
    "max_concurrent",
    "max_workers",
    "image_detail",
    "coordinate_format",
    "box_format",
]


def normalize_classes(raw):
    """Convert classes from comma-separated string or list to a clean list.

    Returns a list of strings, or None if empty.
    """
    if not raw:
        return None
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    return [c.strip() for c in raw.split(",") if c.strip()] or None


def pick_params(params, exclude=()):
    """Filter params to persistable keys, dropping Nones.

    Normalizes classes to a list for consistent storage.
    """
    out = {}
    for k in _PERSIST_KEYS:
        if k in params and k not in exclude and params[k] is not None:
            v = normalize_classes(params[k]) if k == "classes" else params[k]
            if v is not None:
                out[k] = v
    return out


def _global_store():
    from fiftyone.operators.store import ExecutionStore

    return ExecutionStore.create("openai_config", dataset_id=None)


def get_global_config():
    """Read global config from the cross-dataset ExecutionStore."""
    try:
        cfg = _global_store().get("config")
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def save_global_config(params):
    """Persist params to the cross-dataset ExecutionStore."""
    try:
        _global_store().set("config", pick_params(params))
    except Exception:
        pass


def clear_global_config():
    """Delete the cross-dataset config key."""
    try:
        _global_store().delete("config")
    except Exception:
        pass


def parse_config_json(json_str):
    """Parse JSON string, filter to known keys.

    Returns (dict, None) or (None, err).
    """
    try:
        raw = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(raw, dict):
        return None, "JSON must be an object"
    return {k: raw[k] for k in _PERSIST_KEYS if k in raw}, None


def build_image_contents(filepaths, max_workers=4, image_detail="auto"):
    """Build image content dicts for OpenAI chat messages.

    URLs (http/https) are passed through directly. Local files are
    base64-encoded. The ``image_detail`` parameter is set on every
    image_url dict so the API respects the user's detail preference.

    Args:
        filepaths: paths or URLs to image files.
        max_workers: ThreadPoolExecutor size for parallel base64 encoding.
        image_detail: OpenAI image detail level ("low", "high", or "auto").
    """
    results = [None] * len(filepaths)
    to_encode = []

    for i, fp in enumerate(filepaths):
        if fp.startswith(("http://", "https://")):
            results[i] = _url_content(fp, detail=image_detail)
        else:
            to_encode.append(i)

    if to_encode:
        paths = [filepaths[i] for i in to_encode]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            encoded = list(pool.map(_encode_base64, paths))
        for idx, enc in zip(to_encode, encoded):
            enc["image_url"]["detail"] = image_detail
            results[idx] = enc

    return results


def _url_content(url, detail="auto"):
    """Wrap an HTTP(S) URL as an image_url content dict."""
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def _encode_base64(filepath):
    """Read and base64-encode a single image file."""
    mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }
