# Testing

## Local Development Setup

```bash
# Get plugins directory
PLUGINS_DIR=$(python -c "import fiftyone as fo; print(fo.config.plugins_dir)")

# Symlink the plugin (or develop directly in plugins dir)
ln -s /path/to/fo-openai "$PLUGINS_DIR/@Burhan-Q/fo-openai"
```

## Verify Plugin Detection

```python
import fiftyone.operators as foo

# Should list @Burhan-Q/fo-openai/run_openai_inference
foo.list_operators()

# Check operator schema
foo.get_operator_schema("@Burhan-Q/fo-openai/run_openai_inference")
```

## Run with Debug Logging

```bash
fiftyone app debug <dataset-name>
```

In the operator form: enable logging, set level to DEBUG, provide a log file path.

## Test Inference

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart", max_samples=5)
fo.launch_app(dataset)

# Or programmatically:
foo.execute_operator(
    "@Burhan-Q/fo-openai/run_openai_inference",
    params={
        "model": "gpt-5.4-nano",
        "task": "classify",
        "classes": "dog, cat, bird",
    },
    dataset_name=dataset.name,
)
```

## Check Run Summary After Execution

```python
# Run summary is always written, even without log_metadata
dataset.info.get("openai_runs", {})
```

## Check Error Fields

```python
# Per-sample errors
dataset.distinct("openai_infer_classify_error")

# View only samples with errors
error_view = dataset.match(F("openai_infer_classify_error") != None)
```

## Linting

```bash
uvx ruff check .
```

## Compile Check

```bash
python -c "
import py_compile
for f in ['__init__.py', 'engine.py', 'tasks.py', 'utils.py', 'operators.py', '_log.py', '_pricing.py']:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

## Key Things to Verify After Changes

1. **Pydantic models produce valid JSON schemas** — `Model.model_json_schema()` for all 6 models + constrained variants
2. **Pricing data loads** — `from _pricing import get_model_info; get_model_info("gpt-5.4-nano")`
3. **No hardcoded SDK defaults sent** — check `_create_engine` only includes user-set values in `completion_kwargs`
4. **Secrets resolution** — test with `OPENAI_API_KEY` env var set, with `FIFTYONE_OPENAI_API_KEY` set, and with neither
5. **Log file path resolution** — directory input, existing file input, new file input
