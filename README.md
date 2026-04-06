# fo-openai

A [FiftyOne](https://docs.voxel51.com) plugin for labeling images with OpenAI vision models. Send images from your dataset to models like `gpt-4o`, `gpt-4.1-mini`, or `gpt-5.4-nano` and get structured labels back — classifications, tags, detections, captions, VQA answers, or OCR text — directly in the FiftyOne App.

Uses the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/structured-outputs) with Pydantic structured output for reliable, schema-validated responses.

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
- FiftyOne >= 1.13.5
- An OpenAI API key

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

## Usage

### From the FiftyOne App

1. Open a dataset in the FiftyOne App
2. Press `` ` `` to open the operator browser
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
        "model": "gpt-4o-mini",
        "task": "classify",
        "classes": "dog, cat, person, vehicle, other",
    },
    dataset_name=dataset.name,
)
```

## Supported Tasks

| Task | Description | Output |
|------|-------------|--------|
| **Caption** | Describe the image | `fo.Classification` |
| **Classify** | Assign a single label | `fo.Classification` |
| **Tag** | Assign multiple labels | `fo.Classifications` |
| **Detect** | Locate objects with bounding boxes | `fo.Detections` |
| **VQA** | Answer a question about the image | `fo.Classification` |
| **OCR** | Extract visible text | `fo.Classification` |

## Features

- **Cost preview** — estimated per-sample and total cost displayed before running, with token breakdown
- **Few-shot exemplars** — optionally provide labeled samples as reference examples to improve output quality
- **Class labels from your dataset** — pick an existing label field to reuse its classes
- **Structured output** — responses are parsed via the OpenAI Responses API with Pydantic models, not regex or string matching
- **Detection coordinate formats** — pixel (recommended), 0-1000 integers, or 0-1 floats
- **Batch processing** — configurable batch size and concurrency
- **Delegated execution** — run large jobs in the background
- **Persistent settings** — all configuration (including exemplar settings) persists across app restarts
- **Opt-in logging** — log to file for debugging, with auto-incrementing filenames
- **JSON config export/import** — save and reuse run configurations

## Configuration

The operator form is organized into 5 tabs:

| Tab | Settings |
|-----|----------|
| **Model** | Model ID (e.g. `gpt-4o`, `gpt-4.1-mini`), custom base URL |
| **Task** | Task type, class labels, custom prompts, detection coordinate/box format |
| **Exemplars** | Enable few-shot examples, select source (saved view, sample IDs, tag, or field) |
| **Logging** | Enable logging, log level, log file path |
| **Advanced** | Temperature, max output tokens, top P, batch size, concurrency, image detail |

Settings persist between runs. Use the **Reset to defaults** mode to clear them.

## Few-Shot Exemplars

Provide labeled samples as reference examples sent alongside every inference call. The model sees your exemplar images with their correct labels before processing each target image.

Exemplar sources:
- **Saved view** — a named FiftyOne saved view
- **Sample IDs** — specific sample IDs
- **Tag** — all samples matching a tag
- **Field** — samples where a field equals a value

See [docs/operations/exemplars.md](docs/operations/exemplars.md) for details.

## License

Apache 2.0
