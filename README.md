# fo-openai

A [FiftyOne](https://docs.voxel51.com) plugin for labeling images with OpenAI vision models. Send images from your dataset to models like `gpt-4o` or `gpt-5.4-mini` and get structured labels back — classifications, tags, detections, captions, VQA answers, or OCR text — directly in the FiftyOne App.

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
pip install -r requirements.txt  # or: pip install openai pydantic
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

- **Cost preview** — see estimated per-image and total cost before running
- **Class labels from your dataset** — pick an existing label field to reuse its classes
- **Structured output** — responses are parsed via OpenAI's structured output API, not regex or string matching
- **Batch processing** — configurable batch size and concurrency
- **Delegated execution** — run large jobs in the background
- **Persistent settings** — configuration survives app restarts
- **Opt-in logging** — log to file for debugging, with auto-incrementing filenames
- **JSON config export/import** — save and reuse run configurations

## Configuration

All settings are available in the operator form. Key options:

| Setting | Location | Description |
|---------|----------|-------------|
| Model | Main form | Any OpenAI model ID (e.g. `gpt-4o`, `gpt-5.4-mini`) |
| Task | Main form | One of the 6 supported tasks |
| Classes | Task settings | From dataset field, custom list, or open-ended |
| Custom prompt | Task settings | Override the default prompt |
| Enable logging | Output settings | Log progress and errors to stderr / file |
| Image detail | Advanced | `auto`, `low` (cheaper), or `high` (more detail) |
| Timeout | Advanced | Per-request timeout in seconds |

Settings persist between runs. Use the **Reset to defaults** mode to clear them.

## License

Apache 2.0
