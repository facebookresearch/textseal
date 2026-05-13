# Attack Simulation for Posthoc Watermarking

This guide explains how to use the attack simulation features to test watermark robustness.

## Overview

The attack simulation feature allows you to:
1. **Integrated Mode**: Watermark text AND perform attack simulation in a single pipeline
2. **Standalone Mode**: Attack pre-watermarked texts from previous experiments (skip the watermarking phase)

Both modes:
- Rephrase the watermarked text using a specified attack model
- Compute p-values on both the original watermarked text and the attacked (rephrased) version
- Evaluate quality metrics (semantic similarity, ROUGE, BLEU) between watermarked and attacked text

## Mode 1: Integrated Attack Simulation

Add attack evaluation to your existing watermarking pipeline:

### Configuration

Add the `attack` section to your config file or use CLI arguments:

```yaml
# Enable attack evaluation
evaluation:
  evaluate_attack: true  # Evaluate watermark after attack

# Attack configuration
attack:
  enable_attack: true
  attack_model_name: "meta-llama/Llama-3.2-3B-Instruct"  # Model for attacks
  attack_strengths: "all"  # Run all strengths: mild, moderate, aggressive, extreme
  # Or specify specific strengths: "mild,aggressive"
  # Or use single strength: attack_strength: "moderate"
```

### Attack Strength Presets

The attack system includes 4 predefined strength levels with optimized prompts and temperatures:

| Strength | Temperature | Description |
|----------|-------------|-------------|
| **mild** | 0.5 | Minimal edits - fixes grammar/punctuation only |
| **moderate** | 0.8 | Standard rephrasing - different words, same meaning |
| **aggressive** | 1.0 | Heavy rewriting - completely different structure |
| **extreme** | 1.2 | Maximum transformation - only core meaning preserved |

**Usage:**
```bash
# Run all 4 attack strengths (default)
--attack.attack_strengths "all"

# Run specific strengths
--attack.attack_strengths "mild,aggressive"

# Run single strength (legacy)
--attack.attack_strength "moderate"
```

### Example Usage

```bash
# With config file
python -m textseal.watermarking.main \
    --config configs/my_watermark_config.yaml \
    --evaluation.evaluate_attack true \
    --attack.enable_attack true \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct

# With direct arguments
python -m textseal.watermarking.main \
    --input_path assets/sample_document.txt \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --watermark.watermark_type greenlist \
    --watermark.ngram 2 \
    --watermark.delta 4.0 \
    --watermark.gamma 0.5 \
    --evaluation.evaluate_attack true \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
    --attack.attack_temperature 0.8
```

### Output Format

When running multiple attack strengths, results are stored in an `attacks` dictionary keyed by strength:

```json
{
  "line": 0,
  "orig_text": "...",
  "wm_text": "...",
  "wm_eval": {"score": 0.82, "p_value": 1.2e-5, "det": true},
  "attacks": {
    "mild": {
      "attacked_text": "...",
      "attack_stats": {"orig_wm_tokens": 256, "attacked_tokens": 251},
      "attack_wm_eval": {"score": 0.78, "p_value": 5.5e-6, "det": true},
      "attack_quality": {"semantic_similarity": 0.92},
      "temperature": 0.5
    },
    "moderate": {
      "attacked_text": "...",
      "attack_wm_eval": {"score": 0.45, "p_value": 0.05, "det": false},
      "attack_quality": {"semantic_similarity": 0.87},
      "temperature": 0.8
    },
    "aggressive": {
      "attacked_text": "...",
      "attack_wm_eval": {"score": 0.32, "p_value": 0.24, "det": false},
      "attack_quality": {"semantic_similarity": 0.85},
      "temperature": 1.0
    },
    "extreme": {
      "attacked_text": "...",
      "attack_wm_eval": {"score": 0.28, "p_value": 0.03, "det": false},
      "attack_quality": {"semantic_similarity": 0.63},
      "temperature": 1.2
    }
  }
}
```

This allows comparing watermark robustness across different attack intensities in a single run.

## Mode 2: Standalone Attack (Pre-watermarked Texts)

If you already have watermarked texts from previous experiments, use the standalone attack script.

**Important**: The script **automatically loads watermark configuration** from the input JSONL file (if it was created with the updated main.py). You don't need to manually specify watermark parameters!

### Usage

```bash
# Minimal usage - watermark config auto-loaded from input file
python -m textseal.watermarking.attack_only \
    --input_path output/previous_results.jsonl \
    --wm_text_key wm_text \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_path output/attack_results.jsonl

# Manual override (if watermark_config not in file or you want different params)
python -m textseal.watermarking.attack_only \
    --input_path output/previous_results.jsonl \
    --wm_text_key wm_text \
    --watermark.watermark_type greenlist \
    --watermark.ngram 2 \
    --watermark.delta 4.0 \
    --watermark.gamma 0.5 \
    --watermark.scoring_method v1 \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_path output/attack_results.jsonl
```

### Arguments

- `--input_path`: Path to JSONL file with pre-watermarked texts
- `--wm_text_key`: Key in JSONL that contains the watermarked text (default: "wm_text")
- `--output_path`: Where to save attack results
- `--model.model_name`: Model used for watermark detection (should match original)
- `--attack.*`: Attack configuration
- `--watermark.*`: **Optional** - only needed if watermark_config not in input file or to override
- `--entropy_threshold`: Optional entropy threshold for detection
- `--num_lines`: Number of lines to process (-1 for all)

**Note**: If your input JSONL contains a `watermark_config` field (automatically added by main.py), the watermark parameters will be auto-loaded. You only need to specify `--watermark.*` args if:
1. The input file doesn't have `watermark_config` (old format)
2. You want to override the stored configuration

### Output Format

The output JSONL preserves all original fields and adds:
- `attacked_text`: The attacked version
- `wm_eval_before_attack`: Detection on original watermarked text
- `wm_eval_after_attack`: Detection on attacked text
- `attack_stats`: Token statistics
- `attack_quality`: Quality metrics

```json
{
  "line": 0,
  "wm_text": "...",
  "attacked_text": "...",
  "wm_eval_before_attack": {"score": 0.82, "p_value": 1.2e-5, "det": true},
  "wm_eval_after_attack": {"score": 0.45, "p_value": 0.023, "det": false},
  "attack_stats": {
    "orig_wm_tokens": 256,
    "attacked_tokens": 249
  },
  "attack_quality": {
    "semantic_similarity": 0.89,
    "rouge_scores": {...}
  }
}
```

## Advanced Features

### Multi-test Support with Attacks

You can combine attack evaluation with multi-test mode:

```bash
python -m textseal.watermarking.main \
    --input_path assets/sample_document.txt \
    --watermark.watermark_type synthid \
    --evaluation.test_entropy_thresholds "none,1.5,2.0,2.5" \
    --evaluation.evaluate_attack true \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct
```

This will run multiple detection tests on both the watermarked text and the attacked text.

### Using the Same Attack Model for All Experiments

The key advantage is that you can use the **same attack model** regardless of which model was used for watermarking:

```bash
# Watermark with Model A, attack with Model C
python -m textseal.watermarking.main \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct

# Watermark with Model B, attack with Model C (same attack model)
python -m textseal.watermarking.main \
    --model.model_name HuggingFaceTB/SmolLM2-360M-Instruct \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct
```

This allows fair comparison of watermark robustness across different watermarking models.

### Custom Attack Prompts

Override the preset prompts with custom attack behavior:

```yaml
attack:
  attack_system_message: "You are an aggressive paraphrasing tool. Heavily rewrite the text."
  attack_user_message_template: "Rewrite this text completely:\n\n{text}"
  attack_temperature: 1.0  # Higher temperature = more aggressive
```

**Note:** Custom prompts override all strength presets. If you want per-strength control, use the presets instead.

### Choosing the Right Attack Strength

- **mild**: Best for testing if minor edits break the watermark. Preserves most original wording.
- **moderate**: Balanced trade-off between attack strength and quality. Good default.
- **aggressive**: Strong attack that significantly rewrites text. Tests robustness thoroughly.
- **extreme**: Maximum attack power. May degrade text quality. Use for stress testing.

## Example Workflow

### Step 1: Initial Watermarking (without attack)

```bash
python -m textseal.watermarking.main \
    --input_path data/my_documents.jsonl \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --watermark.watermark_type greenlist \
    --dump_dir output/exp1/
```

### Step 2: Later, Attack Pre-watermarked Texts

```bash
python -m textseal.watermarking.attack_only \
    --input_path output/exp1/results.jsonl \
    --wm_text_key wm_text \
    --watermark.watermark_type greenlist \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_path output/exp1/attack_results.jsonl
```

### Step 3: Analyze Results

```python
import json

with open("output/exp1/attack_results.jsonl") as f:
    for line in f:
        data = json.loads(line)
        before_pval = data["wm_eval_before_attack"]["p_value"]
        after_pval = data["wm_eval_after_attack"]["p_value"]
        similarity = data["attack_quality"]["semantic_similarity"]
        
        print(f"Before: p={before_pval:.2e}, After: p={after_pval:.2e}, Sim={similarity:.3f}")
```

## Tips

1. **Use a consistent attack model** across all experiments for fair comparison
2. **Start with moderate attack settings** (temperature=0.8) before trying aggressive attacks
3. **Check semantic similarity** to ensure attacks preserve meaning
4. **Use entropy thresholds** for more robust detection
5. **Process large documents in chunks** - the attack system handles this automatically

## Troubleshooting

**Q: Attack makes watermark undetectable?**
- This is expected behavior - you're testing robustness!
- Try entropy filtering: `--entropy_threshold 2.0`
- Consider stronger watermark parameters

**Q: Attack changes meaning too much?**
- Lower temperature: `--attack.attack_temperature 0.6`
- Adjust system message to emphasize meaning preservation

**Q: Too slow?**
- Use a smaller attack model
- Process fewer lines: `--num_lines 100`
