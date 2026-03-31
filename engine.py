"""OpenAIEngine: async inference via LiteLLM with structured output and cost
estimation."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

import litellm
from pydantic import BaseModel


class OpenAIEngine:
    """LiteLLM-backed engine for OpenAI vision inference with Pydantic
    structured output."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int = 16,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            model: LiteLLM model identifier (e.g. ``"gpt-4o"``).
            api_key: API key.  Passed as ``api_key`` kwarg to
                ``litellm.acompletion``.
            base_url: Optional custom API base for Azure / proxy setups.
            max_concurrent: Semaphore limit for parallel requests.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens per request.
            top_p: Nucleus-sampling probability mass.
            seed: Optional random seed for reproducibility.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed

    def infer_batch(
        self,
        messages: list[list[dict[str, Any]]],
        response_model: type[BaseModel],
    ) -> list[str | Exception]:
        """Run batch inference with structured output via LiteLLM.

        Args:
            messages: One OpenAI-format message list per sample.
            response_model: Pydantic ``BaseModel`` class used as
                ``response_format``.

        Returns:
            A list aligned with *messages* — each element is either the
            response text (JSON) or the ``Exception`` that occurred.
        """
        return _run_async(self._async_infer_batch(messages, response_model))

    async def _async_infer_batch(
        self,
        messages: list[list[dict[str, Any]]],
        response_model: type[BaseModel],
    ) -> list[str | Exception]:
        """Execute all completion requests concurrently."""
        sem = asyncio.Semaphore(self.max_concurrent)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "response_format": response_model,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.seed is not None:
            kwargs["seed"] = self.seed

        async def _call(msgs: list[dict[str, Any]]) -> str:
            """Send a single completion request under the semaphore."""
            async with sem:
                resp = await litellm.acompletion(messages=msgs, **kwargs)
                return resp.choices[0].message.content

        results = await asyncio.gather(
            *[_call(m) for m in messages], return_exceptions=True
        )
        return list(results)

    @staticmethod
    def estimate_cost(
        model: str,
        num_samples: int,
        est_input_tokens: int,
        est_output_tokens: int,
    ) -> dict[str, float] | None:
        """Estimate total cost before execution.

        Returns a dict with ``per_image_cost``, ``total_cost``, and
        per-token rates, or ``None`` when the model is absent from
        LiteLLM's pricing data.
        """
        if model not in litellm.model_cost:
            return None

        info = litellm.model_cost[model]
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

    @staticmethod
    def get_model_info(model: str) -> dict[str, Any] | None:
        """Look up *model* in ``litellm.model_cost``.

        Returns the pricing / capability dict, or ``None`` if unknown.
        """
        return litellm.model_cost.get(model)


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
