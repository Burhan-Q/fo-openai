# Project Overview

**Plugin name:** `@Burhan-Q/fo-openai`
**Version:** 0.1.0
**Python:** >=3.11
**FiftyOne:** >=1.13.5

## Purpose

Send images from a FiftyOne dataset to OpenAI vision models for labeling (classification, tagging, detection, captioning, VQA, OCR) directly from the FiftyOne App. Results are written back as FiftyOne label fields.

## Dependencies

| Package | Min Version | Role |
|---------|-------------|------|
| `fiftyone` | 1.13.5 | Dataset framework, operator system, App UI |
| `openai` | 2.0.0 | API client — `AsyncOpenAI.responses.parse()` (Responses API) |
| `pydantic` | 2.12.5 | Structured output response models |

## Files

```
fo-openai/
├── fiftyone.yml        # Plugin manifest — operators, secrets
├── __init__.py         # Entry point — register(plugin)
├── operators.py        # OpenAIInference + CheckOpenAIStatus operators, all UI helpers
├── engine.py           # OpenAIEngine — async batch inference with structured output
├── tasks.py            # TaskConfig — prompts, Pydantic models, response parsing
├── exemplars.py        # Few-shot exemplar resolution, serialization, message building
├── utils.py            # Image encoding, config persistence (ExecutionStore)
├── _log.py             # Plugin-scoped logging (fo_openai namespace)
├── _pricing.py         # Model pricing — fetched from GitHub, cached locally
└── pyproject.toml      # Project metadata, dependencies
```

## Registered Operators

| Operator | URI | Purpose |
|----------|-----|---------|
| `OpenAIInference` | `@Burhan-Q/fo-openai/run_openai_inference` | Main inference operator |
| `CheckOpenAIStatus` | `@Burhan-Q/fo-openai/check_openai_status` | Delegated-job completion listener |

## Secrets (fiftyone.yml)

- `FIFTYONE_OPENAI_API_KEY` — primary API key
- `OPENAI_API_KEY` — fallback (common env var name)

Both are resolved via `ctx.secrets` which reads from environment variables. See [Configuration](../operations/configuration.md).
