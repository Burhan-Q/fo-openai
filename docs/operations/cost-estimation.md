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
- **Prompt text:** 50 tokens (constant estimate for system + user prompt text)
- **Image tokens:** depends on `image_detail` setting:
  - `low`: 85 tokens
  - `high`: 765 tokens
  - `auto`: 765 tokens (conservative)
- **Total input:** `50 + image_tokens`

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

Shown as a `Notice` (< $1) or `Warning` (>= $1) in the form:
```
Estimated cost: $0.00097/image, $0.1943 total for 200 samples
```

Dollar amounts use dynamic precision via `_fmt_usd()`:
- `$0.00` for zero
- `$0.00097` for very small values (enough decimals for 2 significant figures)
- `$0.1943` for sub-dollar
- `$12.35` for dollar+ amounts

If the model is not in the pricing data, a notice reads "Cost preview unavailable".
