# Generation-time Watermarking

Generate watermarked text directly from prompts.
The watermark is embedded during generation by biasing the sampling process with a secret key.

## Quick Start

### Environment Setup

See [../README.md](../README.md) for environment setup.

- Activate the pre-configured env: `conda activate text_seal`
- Ensure access to a CUDA GPU for fast generation
- Log in to Hugging Face if pulling gated models/tokenizers

### Basic Usage

```bash
# Generate watermarked answers from prompts
python -m textseal.watermarking.main \
  --input_path prompts.jsonl --text_key question \
  --processing.generation_mode true \
  --processing.max_gen_len 512 \
  --model.model_name Qwen/Qwen3.5-2B \
  --watermark.watermark_type textseal \
  --processing.temperature 1.0 --processing.top_p 0.95
```

The input JSONL should contain one prompt per line:
```json
{"question": "What are the main causes of climate change?"}
{"question": "Explain how transformers work in machine learning."}
```

## Reasoning Mode

For models that support thinking (e.g., Qwen3.5), enable reasoning with `processing.reasoning.enabled`. Both the reasoning trace and the answer are watermarked, but evaluated separately.

```bash
# Generate with reasoning (model produces <think>...</think> then answers)
python -m textseal.watermarking.main \
  --input_path prompts.jsonl --text_key question \
  --processing.generation_mode true \
  --processing.reasoning.enabled true \
  --processing.reasoning.max_tokens 200 \
  --processing.max_gen_len 1024 \
  --model.model_name Qwen/Qwen3.5-2B \
  --watermark.watermark_type textseal \
  --processing.temperature 1.0 --processing.top_p 0.95
```

| Flag | Description |
|------|-------------|
| `--processing.reasoning.enabled true` | Enable thinking mode (uses `enable_thinking` in the chat template) |
| `--processing.reasoning.max_tokens N` | Force reasoning to stop after N tokens (injects `</think>` token). Set 0 for unlimited. |

When reasoning is enabled:
- The output contains `<think>...</think>` followed by the answer.
- Watermark detection is evaluated separately on the reasoning trace and the answer.
- The output JSON includes `reasoning_eval` with `think_pvalue`, `reasoning_tokens`, and `answer_tokens`.


## Configuration

### ProcessingConfig (generation-specific)

- `generation_mode` (bool): `true` to enable generation mode. Default: `false` (rephrase mode).
- `max_gen_len` (int): Maximum tokens to generate. Used as-is in generation mode (no length capping).
- `temperature` (float): Sampling temperature. Higher = more diverse.
- `top_p` (float): Nucleus sampling threshold.
- `reasoning.enabled` (bool): Enable thinking mode.
- `reasoning.max_tokens` (int): Max reasoning tokens before forcing end-of-thought. 0 = unlimited.
- `reasoning.start_token` (str): Token marking start of reasoning trace (default: `<think>`).
- `reasoning.end_token` (str): Token marking end of reasoning trace (default: `</think>`).

### Recommended Watermark: TextSeal

The `textseal` watermark type uses dual-key Gumbel-max sampling. It empirically provides the strongest detection with minimal quality impact.

```bash
--watermark.watermark_type textseal \
--watermark.mixing_alpha 0.5 \
--watermark.ngram 2 \
--watermark.scoring_method v1
```

Key parameters:
- `mixing_alpha` (float): Probability of using Key A per token (default: 0.5 = equal mixing).
- `ngram` (int): N-gram context window for watermark decisions.
- `scoring_method`: Use `v1` (deduplicate by window) for cleaner p-values.
- `secret_key` (int): Primary secret key for attribution. Default: 42.

Other watermark types (`gumbelmax`, `greenlist`, `synthid`, etc.) also work in generation mode.

### Prompt Configuration

In generation mode, the input text is used directly as the user message with no system prompt by default:

```bash
# Custom system message
--prompt.system_message "You are a helpful assistant."

# Prefill the answer (useful for structured output)
--prompt.prefill_answer "Here is the answer:\n"
```

## Output Format

```json
{
  "wm_text": "The generated watermarked answer...",
  "orig_text": "The original prompt...",
  "wm_eval": {
    "score": 2.13,
    "p_value": 3.2e-08,
    "det": true,
    "toks_gen": 156,
    "toks_scored": 154
  },
  "reasoning_eval": {
    "think_pvalue": 1.4e-06,
    "reasoning_tokens": 202,
    "answer_tokens": 156
  }
}
```

The `reasoning_eval` field is only present when reasoning is enabled.

## Full Example

```bash
python -m textseal.watermarking.main \
  --input_path prompts.jsonl --text_key question \
  --dump_dir output/ \
  --processing.generation_mode true \
  --processing.reasoning.enabled true \
  --processing.reasoning.max_tokens 200 \
  --processing.max_gen_len 1024 \
  --model.model_name Qwen/Qwen3.5-2B \
  --model.use_flash_attention true \
  --watermark.watermark_type textseal \
  --watermark.mixing_alpha 0.5 \
  --watermark.ngram 2 \
  --watermark.scoring_method v1 \
  --processing.temperature 1.0 \
  --processing.top_p 0.95 \
  --num_lines 100 \
  --verbose 1
```
