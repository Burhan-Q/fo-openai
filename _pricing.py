"""Model pricing data fetched from the LiteLLM community-maintained list.

The JSON is fetched from GitHub and cached locally.  A fresh copy is
downloaded at most once per day.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from ._log import get_logger

logger = get_logger(__name__)

_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm"
    "/main/model_prices_and_context_window.json"
)

_CACHE_DIR = Path("~/.fiftyone/plugins/cache/fo-openai").expanduser()
_CACHE_FILE = _CACHE_DIR / "model_prices.json"
_CACHE_MAX_AGE = 86400  # 24 hours in seconds

# In-memory cache so we only read from disk once per process
_model_cost: dict[str, Any] | None = None


def _is_cache_fresh() -> bool:
    """Return ``True`` if the on-disk cache exists and is less than a day old."""
    if not _CACHE_FILE.exists():
        return False
    age = time.time() - _CACHE_FILE.stat().st_mtime
    return age < _CACHE_MAX_AGE


def _fetch_remote() -> dict[str, Any]:
    """Download the pricing JSON from GitHub."""
    logger.debug("Fetching pricing data from %s", _PRICING_URL)
    with urlopen(_PRICING_URL, timeout=15) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _load_pricing() -> dict[str, Any]:
    """Load pricing data, using cached copy when available."""
    global _model_cost  # noqa: PLW0603

    if _model_cost is not None:
        return _model_cost

    if _is_cache_fresh():
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _model_cost = data
            logger.debug("Loaded pricing from cache (%s)", _CACHE_FILE)
            return _model_cost
        except Exception:
            pass  # Fall through to remote fetch

    try:
        data = _fetch_remote()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps(data), encoding="utf-8"
        )
        _model_cost = data
        logger.debug("Cached pricing data to %s", _CACHE_FILE)
        return _model_cost
    except Exception as e:
        logger.warning("Failed to fetch pricing data: %s", e)
        # Last resort: use stale cache if it exists
        if _CACHE_FILE.exists():
            try:
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                _model_cost = data
                logger.debug("Using stale pricing cache")
                return _model_cost
            except Exception:
                pass
        _model_cost = {}
        return _model_cost


def get_model_info(model: str) -> dict[str, Any] | None:
    """Look up *model* in the pricing data.

    Returns the pricing / capability dict, or ``None`` if unknown.
    """
    return _load_pricing().get(model)


def estimate_cost(
    model: str,
    num_samples: int,
    est_input_tokens: int,
    est_output_tokens: int,
) -> dict[str, float] | None:
    """Estimate total cost before execution.

    Returns a dict with ``per_image_cost``, ``total_cost``, and
    per-token rates, or ``None`` when the model is absent from
    the pricing data.
    """
    info = get_model_info(model)
    if info is None:
        return None

    input_cpt: float = info.get("input_cost_per_token", 0)
    output_cpt: float = info.get("output_cost_per_token", 0)
    per_image = (est_input_tokens * input_cpt) + (
        est_output_tokens * output_cpt
    )

    return {
        "per_image_cost": per_image,
        "total_cost": per_image * num_samples,
        "input_cost_per_token": input_cpt,
        "output_cost_per_token": output_cpt,
    }
