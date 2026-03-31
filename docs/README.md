# fo-openai Documentation

Internal reference for the `@Burhan-Q/fo-openai` FiftyOne plugin.

## Architecture

- [Project Overview](architecture/overview.md) — files, dependencies, plugin manifest
- [Module Map](architecture/modules.md) — what each module does, dependency graph
- [Inference Pipeline](architecture/pipeline.md) — step-by-step execution flow from form to results
- [Structured Output](architecture/structured-output.md) — Pydantic models, OpenAI SDK parse(), constrained models

## Operations

- [Tasks](operations/tasks.md) — the 6 supported tasks, prompts, output types
- [Configuration](operations/configuration.md) — 3-mode config, persistence, secrets
- [Cost Estimation](operations/cost-estimation.md) — pricing data source, token estimates, display
- [Logging](operations/logging.md) — opt-in logging, file paths, error classification, run summaries
- [Detection](operations/detection.md) — bounding box formats, coordinate systems, conversion pipeline

## Development

- [Design Decisions](development/decisions.md) — key choices made during development and why
- [Known Issues](development/known-issues.md) — current limitations, FiftyOne UI constraints
- [Testing](development/testing.md) — how to test the plugin locally
