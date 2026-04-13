# Cost Estimation

Inline cost preview shown in the operator form before execution.

## Data Source

Pricing data is fetched from the LiteLLM community-maintained JSON:
```
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
```

~2,600 models, ~1MB. Cached to `~/.fiftyone/plugins/cache/fo-openai/model_prices.json`.

**Cache strategy:**
- In-memory dict per process (fastest)
- On-disk JSON with 24-hour TTL
- Remote fetch if cache is stale or missing
- Falls back to stale cache if remote fetch fails
- Falls back to empty dict if nothing available

## Token Estimation

### Input tokens per sample
- **Prompt text:** dynamically estimated via `_estimate_prompt_tokens()` using a ~4 chars/token heuristic on the system prompt, user prompt, classes list, and question text. Minimum floor of 50 tokens (`PROMPT_TEXT_TOKENS`).
- **Image tokens:** depends on `image_detail` setting:
  - `low`: 85 tokens
  - `high`: 765 tokens
  - `auto`: 765 tokens (conservative)
- **Total input:** `prompt_tokens + image_tokens`

### Output tokens per task
Heuristic estimates in `OUTPUT_TOKEN_ESTIMATES`:

| Task | Estimated Tokens |
|------|-----------------|
| caption | 80 |
| classify | 15 |
| tag | 40 |
| detect | 200 |
| vqa | 60 |
| ocr | 100 |

## Cost Calculation

```
per_image_cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
total_cost = per_image_cost * num_samples
```

## Display

Rendered as a persistent markdown table below the tabs (always visible regardless of active tab):

```
| | Tokens/Call | Total Tokens | Cost/Sample | Total Cost |
|---|--:|--:|--:|--:|
| **Prompt** | 50 | 10,000 | | |
| **Image** | 765 | 153,000 | | |
| **Output** | 80 | 16,000 | | |
| **Inference** | 895 | 179,000 | $0.0015 | $0.30 |
| **Exemplars** (3) | 2,385 | 477,000 | $0.0040 | $0.80 |
| **Total** | 3,280 | **656,000** | $0.0055 | **$1.10** |

*200 samples*
```

The exemplar row only appears when exemplars are enabled.

A `Warning` banner appears when the total cost exceeds the threshold (default `$5.00`, configurable via `FIFTYONE_OPENAI_COST_WARN` environment variable).

Dollar amounts use dynamic precision via `_fmt_usd()`:
- `$0.00` for zero
- `$0.00097` for very small values (enough decimals for 2 significant figures)
- `$0.1943` for sub-dollar
- `$12.35` for dollar+ amounts

If the model is not in the pricing data, a notice reads "Cost preview unavailable".
