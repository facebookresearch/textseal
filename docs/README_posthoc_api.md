# API Usage Guide

This guide shows how to use the TextSeal Python API for different use cases after installing via pip.

## Installation

```bash
pip install textseal
```

## Use Cases

### 1. Watermarking + Detection (Default)

Add a watermark to text and verify it:

```python
from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig, ProcessingConfig

# Create watermarker
watermarker = PostHocWatermarker(
    watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
    model_config=ModelConfig(model_name="meta-llama/Llama-3.2-3B-Instruct"),
    processing_config=ProcessingConfig(temperature=0.8),
)

# Watermark text
result = watermarker.process_text("Your text here")
print(result["wm_text"])           # Watermarked text
print(result["wm_eval"]["p_value"]) # Detection p-value
print(result["wm_eval"]["det"])     # True if detected
```

**Returns dict with:**
- `wm_text` (str): Watermarked text
- `orig_text` (str): Original text
- `wm_eval` (dict): Detection metrics (p_value, det, score, etc.)
- `quality` (dict): Quality metrics (BLEU, ROUGE, semantic_similarity, etc.)
- `times` (dict): Timing info (t_rephrase, tps, t_wm_eval, t_quality, t_total)
- `stats` (dict): Token/length statistics (orig_len, wm_len, tok_ratio, etc.)

> **Note:**  
> Detection uses a default false positive rate (FPR) of 0.1%. You can adjust this threshold in `EvaluationConfig` by setting the `detection_threshold` parameter if you need a different trade-off between TPR and FPR.


### 2. Watermarking Only (Just the Text)

Get just the watermarked text without evaluation metrics:

```python
from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig

watermarker = PostHocWatermarker(
    watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
    model_config=ModelConfig(model_name="meta-llama/Llama-3.2-3B-Instruct"),
)

# Returns just the watermarked text string
watermarked_text = watermarker.rephrase_with_watermark("Your text here")
print(watermarked_text)
```

**Returns:** String - just the watermarked text (no metadata).

### 3. Detection Only (No Watermarking)

Detect watermarks in existing text without rephrasing:

```python
from textseal import PostHocWatermarker, WatermarkConfig, EvaluationConfig

# Create detector-only instance
detector = PostHocWatermarker(
    watermark_config=WatermarkConfig(
        watermark_type="gumbelmax",
        secret_key=42,  # Must match watermark secret key
    ),
    evaluation_config=EvaluationConfig(
        enable_detection_only=True,
    )
)

# Detect watermark
text_to_check = "Text that may be watermarked"
wm_eval = detector.evaluate_watermark(text_to_check)

print(f"P-value: {wm_eval['p_value']}")
print(f"Detected: {wm_eval['det']}")
print(f"Score: {wm_eval['score']}")
```

**Important:**
- Watermark detection does not need the LLM (unless entropy thresholds used). Set `enable_detection_only=True` to skip model loading when creating the post-hoc watermarker. If entropy thresholds are used, the model is still needed to compute token probabilities, so ensure the model config is set accordingly.
- Must specify correct `watermark_type` and `secret_key` matching watermarking settings
- For CLI batch detection: `python -m textseal.posthoc.main --evaluation.enable_detection_only true`

**Returns dict with:**
- `p_value` (float): Statistical p-value for detection
- `det` (bool): True if watermark detected
- `score` (float): Mean watermark score per token
- `toks_gen`, `toks_scored` (int): Token counts

## Command Line Interface

After installing textseal, you get the `textseal-watermark` CLI command:

```bash
# Get help
textseal-watermark --help

# Watermark a file
textseal-watermark --input_path document.txt --dump_dir output/

# Custom configuration
textseal-watermark \
  --input_path document.txt \
  --dump_dir output/ \
  --watermark.watermark_type gumbelmax \
  --model.model_name meta-llama/Llama-3.2-3B-Instruct \
  --processing.temperature 0.8

# Detection-only mode (no rephrasing)
textseal-watermark \
  --input_path text_to_check.txt \
  --evaluation.enable_detection_only true
```

You can also run it as a Python module:
```bash
python -m textseal.posthoc.main --help
```
