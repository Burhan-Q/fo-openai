"""OpenAIEngine: async inference via the OpenAI Responses API with
Pydantic structured output parsing."""

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
    output via ``client.responses.parse()``."""

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
            model: OpenAI model identifier (e.g. ``"gpt-5.4-nano"``).
            api_key: API key for authentication.
            base_url: Optional custom API base for Azure / proxy setups.
            max_concurrent: Semaphore limit for parallel requests.
            timeout: Per-request timeout in seconds.  ``None`` uses the
                SDK default.
            **completion_kwargs: Additional keyword arguments forwarded
                directly to ``client.responses.parse()``.
                Only user-specified values should be passed (e.g.
                ``temperature``, ``max_output_tokens``, ``top_p``).
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.completion_kwargs = completion_kwargs

    def infer_batch(
        self,
        instructions: str,
        inputs: list[list[dict[str, Any]]],
        response_model: type[BaseModel],
    ) -> list[BaseModel | Exception]:
        """Run batch inference with structured output via the Responses API.

        Uses ``client.responses.parse()`` which returns
        ``output_parsed`` as a validated Pydantic model instance.

        Args:
            instructions: System instructions shared across all samples.
            inputs: One Responses API ``input`` list per sample.
            response_model: Pydantic ``BaseModel`` class used as
                ``text_format``.

        Returns:
            A list aligned with *inputs* — each element is either a
            parsed Pydantic model instance or the ``Exception`` that
            occurred.
        """
        return _run_async(
            self._async_infer_batch(instructions, inputs, response_model)
        )

    async def _async_infer_batch(
        self,
        instructions: str,
        inputs: list[list[dict[str, Any]]],
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
            "text_format": response_model,
            "store": False,
            **self.completion_kwargs,
        }

        async def _call(input_msgs: list[dict[str, Any]]) -> BaseModel:
            """Send a single parse request under the semaphore."""
            async with sem:
                resp = await client.responses.parse(
                    input=input_msgs,
                    instructions=instructions,
                    **kwargs,
                )
                # Check for refusal in output content
                for item in resp.output:
                    if hasattr(item, "content"):
                        for c in item.content:
                            if c.type == "refusal":
                                raise ValueError(
                                    f"Model refused: {c.refusal}"
                                )
                if resp.output_parsed is None:
                    raise ValueError("Empty parsed response")
                return resp.output_parsed

        results = await asyncio.gather(
            *[_call(m) for m in inputs], return_exceptions=True
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
