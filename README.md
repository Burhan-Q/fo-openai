# fo-openai

A [FiftyOne](https://github.com/voxel51/fiftyone) plugin for labeling images with [OpenAI](https://developers.openai.com/api/docs) vision models. Send images from your dataset to models like `gpt-5.4-nano`, `gpt-5.4-mini`, or `gpt-5.4` and get structured labels back — classifications, tags, detections, captions, VQA answers, or OCR text — directly in the [FiftyOne App](https://docs.voxel51.com/user_guide/app.html).

Uses the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/structured-outputs) with [Pydantic](https://docs.pydantic.dev/) structured output for reliable, schema-validated responses.

## Installation

```bash
# Install the plugin
fiftyone plugins download https://github.com/Burhan-Q/fo-openai

# Or clone and install locally
git clone https://github.com/Burhan-Q/fo-openai.git
fiftyone plugins create @Burhan-Q/fo-openai --from-dir ./fo-openai
```

### Requirements

- Python >= 3.11
- [FiftyOne](https://docs.voxel51.com) >= 1.13.5
- [OpenAI Python SDK](https://github.com/openai/openai-python) >= 2.0.0
- An [OpenAI API key](https://platform.openai.com/api-keys)

Install Python dependencies:

```bash
cd "$(fiftyone config plugins_dir)/@Burhan-Q/fo-openai"
pip install openai pydantic
```

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or use the FiftyOne-specific name:

```bash
export FIFTYONE_OPENAI_API_KEY="sk-..."
```

Both are declared in `fiftyone.yml` and resolved automatically via FiftyOne's [plugin secrets](https://docs.voxel51.com/plugins/developing_plugins.html#plugin-secrets) system.

## Usage

### From the FiftyOne App

1. Open a dataset in the [FiftyOne App](https://docs.voxel51.com/user_guide/app.html)
2. Press `` ` `` to open the [operator browser](https://docs.voxel51.com/plugins/using_plugins.html#using-operators)
3. Search for **"Run OpenAI Inference"**
4. Select a model and task, then execute

### Programmatic

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart", max_samples=10)

foo.execute_operator(
    "@Burhan-Q/fo-openai/run_openai_inference",
    params={
        "model": "gpt-5.4-nano",
        "task": "classify",
        "classes": ["dog", "cat", "person", "vehicle", "other"],
    },
    dataset_name=dataset.name,
)
```

> **Note:** `classes` accepts either a list of strings or a comma-separated string (e.g. `"dog, cat, person"`).

## Supported Tasks

| Task | Description | Output Type |
|------|-------------|-------------|
| **Caption** | Generate a text description of the image | [`fo.Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) |
| **Classify** | Assign a single class label | [`fo.Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) |
| **Tag** | Assign multiple labels | [`fo.Classifications`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classifications) |
| **Detect** | Locate objects with bounding boxes | [`fo.Detections`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Detections) |
| **VQA** | Answer a question about the image | [`fo.Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) |
| **OCR** | Extract visible text | [`fo.Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) |

## Features

- **Structured output** — responses parsed via [`client.responses.parse()`](https://developers.openai.com/api/docs/guides/structured-outputs) with [Pydantic](https://docs.pydantic.dev/) models; no regex or string matching
- **Cost preview** — estimated per-sample and total cost displayed before running, with token breakdown table
- **Few-shot exemplars** — optionally provide labeled samples as reference examples to improve output quality
- **Class labels from your dataset** — pick an existing label field to reuse its classes, or type new classes with autocomplete
- **Detection coordinate formats** — pixel (recommended), 0-1000 integers, or 0-1 floats
- **Batch processing** — configurable batch size and concurrency
- **[Delegated execution](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations)** — run large jobs in the background
- **Persistent settings** — all configuration persists across app restarts via [ExecutionStore](https://docs.voxel51.com/plugins/developing_plugins.html#execution-store)
- **Opt-in logging** — log to stderr and/or file, with auto-incrementing filenames
- **JSON config export/import** — save and reuse run configurations

## Configuration

The operator form has **model and base URL always visible** at the top, with 4 tabs below:

| Section | Settings |
|---------|----------|
| **Model** (always visible) | Model ID (e.g. `gpt-5.4-nano`, `gpt-5.4-mini`), custom base URL |
| **Task** tab | Task type, class labels (autocomplete), custom prompts, detection coordinate/box format, output field |
| **Exemplars** tab | Enable few-shot examples, select source (saved view, sample IDs, tag, or field), label field |
| **Logging** tab | Enable logging, log level, log file path |
| **Advanced** tab | Temperature, max output tokens, top P, request timeout, batch size, concurrency, image workers, image detail, system prompt override |

A cost summary table is rendered below the tabs (always visible regardless of active tab).

Settings persist between runs. Use the **Reset to defaults** config mode to clear them.

## Few-Shot Exemplars

Provide labeled samples as reference examples sent alongside every inference call. The model sees your exemplar images with their correct labels before processing each target image.

Exemplar sources:
- **Saved view** — a named [FiftyOne saved view](https://docs.voxel51.com/user_guide/using_views.html#saving-views)
- **Sample IDs** — specific sample IDs
- **Tag** — all samples matching a [tag](https://docs.voxel51.com/user_guide/using_datasets.html#tags)
- **Field** — samples where a field equals a value

See [docs/operations/exemplars.md](docs/operations/exemplars.md) for details.

## License

Apache 2.0
