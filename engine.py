"""OpenAIEngine: async inference via the OpenAI Python SDK with structured
output parsing."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from ._log import get_logger

logger = get_logger(__name__)


class OpenAIEngine:
    """OpenAI SDK-backed engine for vision inference with Pydantic structured
    output via ``client.beta.chat.completions.parse()``."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int = 16,
        timeout: float | None = None,
        **completion_kwargs: Any,
    ) -> None:
        """Initialise the engine.

        Args:
            model: OpenAI model identifier (e.g. ``"gpt-4o"``).
            api_key: API key for authentication.
            base_url: Optional custom API base for Azure / proxy setups.
            max_concurrent: Semaphore limit for parallel requests.
            timeout: Per-request timeout in seconds.  ``None`` uses the
                SDK default.
            **completion_kwargs: Additional keyword arguments forwarded
                directly to ``client.beta.chat.completions.parse()``.
                Only user-specified values should be passed (e.g.
                ``temperature``, ``max_completion_tokens``, ``top_p``,
                ``seed``).
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.completion_kwargs = completion_kwargs

    def infer_batch(
        self,
        messages: list[list[dict[str, Any]]],
        response_model: type[BaseModel],
    ) -> list[BaseModel | Exception]:
        """Run batch inference with structured output via the OpenAI SDK.

        Uses ``client.beta.chat.completions.parse()`` which returns
        ``message.parsed`` as a validated Pydantic model instance.

        Args:
            messages: One OpenAI-format message list per sample.
            response_model: Pydantic ``BaseModel`` class used as
                ``response_format``.

        Returns:
            A list aligned with *messages* — each element is either a
            parsed Pydantic model instance or the ``Exception`` that
            occurred.
        """
        return _run_async(self._async_infer_batch(messages, response_model))

    async def _async_infer_batch(
        self,
        messages: list[list[dict[str, Any]]],
        response_model: type[BaseModel],
    ) -> list[BaseModel | Exception]:
        """Execute all completion requests concurrently."""
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        sem = asyncio.Semaphore(self.max_concurrent)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "response_format": response_model,
            **self.completion_kwargs,
        }

        async def _call(msgs: list[dict[str, Any]]) -> BaseModel:
            """Send a single parse request under the semaphore."""
            async with sem:
                resp = await client.beta.chat.completions.parse(
                    messages=msgs, **kwargs
                )
                msg = resp.choices[0].message
                if msg.refusal:
                    raise ValueError(f"Model refused: {msg.refusal}")
                if msg.parsed is None:
                    raise ValueError(
                        f"Empty parsed response (content={msg.content!r})"
                    )
                return msg.parsed

        results = await asyncio.gather(
            *[_call(m) for m in messages], return_exceptions=True
        )
        return list(results)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine safely, handling existing event loops.

    FiftyOne's App runs a Uvicorn server with its own event loop.
    Calling ``asyncio.run()`` from within that context raises
    ``RuntimeError``.  This helper detects that case and executes the
    coroutine in a dedicated thread instead.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
