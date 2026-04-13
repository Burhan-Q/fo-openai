"""Plugin-scoped logging for fo-openai.

All loggers live under the ``fo_openai`` namespace so they are fully
isolated from FiftyOne's own logging and can be configured independently.

Usage inside the plugin::

    from ._log import get_logger
    logger = get_logger(__name__)
    logger.info("processing %d samples", n)

The logger hierarchy is configured once per operator run via
:func:`configure`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT_NAME = "fo_openai"
_CONFIGURED = False
_LOG_MAX_INCREMENT = 100

# Reusable format for both stream and file handlers
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``fo_openai`` namespace.

    *name* is typically ``__name__`` of the calling module.  The returned
    logger inherits the level and handlers set on the root ``fo_openai``
    logger by :func:`configure`.
    """
    short = name.rsplit(".", 1)[-1] if "." in name else name
    return logging.getLogger(f"{_ROOT_NAME}.{short}")


def _resolve_log_path(raw: str) -> Path:
    """Turn a user-supplied log path into a concrete file path.

    Rules:

    * If *raw* points to an existing directory (or ends with ``/``),
      generate a timestamped filename inside it:
      ``<dir>/2026-03-31T14-59-11.log``.
    * If *raw* points to an existing file, auto-increment a numeric
      suffix to avoid overwriting:
      ``run.log`` → ``run2.log`` → ``run3.log`` → ...
    * Otherwise, use the path as-is (creating parent dirs as needed).
    """
    path = Path(raw).expanduser()

    # Directory → generate ISO-timestamped filename
    if path.is_dir() or raw.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        return path / f"{ts}.log"

    # Existing file → auto-increment suffix (capped, then overwrite newest)
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for n in range(2, _LOG_MAX_INCREMENT + 1):
            candidate = parent / f"{stem}{n}{suffix}"
            if not candidate.exists():
                return candidate
        return parent / f"{stem}{_LOG_MAX_INCREMENT}{suffix}"

    # New file → use as-is
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure(
    *,
    enabled: bool = False,
    level: str = "INFO",
    log_file: str = "",
) -> logging.Logger:
    """Set up the plugin root logger for the current run.

    Args:
        enabled: When ``False`` the root logger is set to ``WARNING``
            so only genuine warnings/errors propagate.  When ``True``
            the logger is set to *level*.
        level: Python log level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``).
        log_file: If non-empty, a ``FileHandler`` writing to this path
            is attached.  See :func:`_resolve_log_path` for path
            resolution rules (directory, existing file, new file).

    Returns:
        The root ``fo_openai`` logger.
    """
    global _CONFIGURED  # noqa: PLW0603
    root = logging.getLogger(_ROOT_NAME)

    # Remove any handlers from a prior run so we don't duplicate
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    if not enabled:
        root.setLevel(logging.WARNING)
        _CONFIGURED = True
        return root

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # Always add a stderr handler so `fiftyone app debug` shows logs
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Optionally add a file handler
    if log_file:
        path = _resolve_log_path(log_file)
        fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
        root.info("Logging to file: %s", path)

    # Don't propagate to the root Python logger (avoids double output
    # and keeps FiftyOne's own logging unaffected)
    root.propagate = False
    _CONFIGURED = True
    return root


def is_configured() -> bool:
    """Return whether :func:`configure` has been called."""
    return _CONFIGURED


def truncate(text: str, max_len: int = 200) -> str:
    """Shorten *text* for log messages, appending ``...`` if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def summarise_errors(
    api_errors: int,
    parse_errors: int,
    total: int,
    error_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a run-summary dict for ``dataset.info``.

    Args:
        api_errors: Count of samples that failed at the API call stage.
        parse_errors: Count of samples that failed at response parsing.
        total: Total samples attempted.
        error_samples: Up to N dicts with ``{"id", "stage", "error"}``
            for the first few failures.

    Returns:
        A JSON-serialisable summary dict.
    """
    return {
        "total": total,
        "succeeded": total - api_errors - parse_errors,
        "api_errors": api_errors,
        "parse_errors": parse_errors,
        "first_errors": error_samples,
    }
