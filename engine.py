"""OpenAIEngine: async inference via LiteLLM with structured output and cost
estimation."""

import asyncio
import concurrent.futures

import litellm


class OpenAIEngine:
    """LiteLLM-backed engine for OpenAI vision inference with Pydantic
    structured output."""

    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        max_concurrent=16,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        seed=None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed

    def infer_batch(self, messages, response_model):
        """Run batch inference with structured output via LiteLLM.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            response_model: Pydantic BaseModel class for structured output.

        Returns:
            list of response strings (JSON conforming to the Pydantic model)
            or Exception instances for failed requests.
        """
        return _run_async(self._async_infer_batch(messages, response_model))

    async def _async_infer_batch(self, messages, response_model):

        sem = asyncio.Semaphore(self.max_concurrent)

        kwargs = {
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

        async def _call(msgs):
            async with sem:
                resp = await litellm.acompletion(messages=msgs, **kwargs)
                return resp.choices[0].message.content

        results = await asyncio.gather(
            *[_call(m) for m in messages], return_exceptions=True
        )
        return list(results)

    @staticmethod
    def estimate_cost(model, num_samples, est_input_tokens, est_output_tokens):
        """Estimate total cost before execution.

        Returns a dict with per_image_cost, total_cost, and per-token
        rates, or None if the model is not in LiteLLM's pricing data.
        """

        if model not in litellm.model_cost:
            return None

        info = litellm.model_cost[model]
        input_cpt = info.get("input_cost_per_token", 0)
        output_cpt = info.get("output_cost_per_token", 0)
        per_image = (est_input_tokens * input_cpt) + (est_output_tokens * output_cpt)

        return {
            "per_image_cost": per_image,
            "total_cost": per_image * num_samples,
            "input_cost_per_token": input_cpt,
            "output_cost_per_token": output_cpt,
        }

    @staticmethod
    def get_model_info(model):
        """Look up model in litellm.model_cost.

        Returns the model info dict or None.
        """
        import litellm

        return litellm.model_cost.get(model)


def _run_async(coro):
    """Run an async coroutine safely, handling existing event loops.

    FiftyOne's App runs a Uvicorn server with its own event loop.
    Calling asyncio.run() from within that context raises
    RuntimeError: 'This event loop is already running'.
    This helper detects that case and runs in a dedicated thread.
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
