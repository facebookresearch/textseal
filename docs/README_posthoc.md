# Post-hoc Watermarking for Large Language Models

A modular system that rephrases existing text with a paraphrasing LLM while embedding a detectable statistical watermark. 
This enables authenticity checks on rephrased documents, and content attribution.

How it works (2 main components):
- Paraphrasing LLM: rewrites input text with controlled sampling (temperature/top-p) and optional adaptive chunking for long documents.
- Watermarking algorithm: influences generation based on a secret key; a detector then tests for watermark presence (p-value).

## Quick Start

### Environment Setup

See [../README.md](../README.md) for environment setup.

- Activate the pre-configured env: conda activate text_seal
- Ensure access to a CUDA GPU for fast generation (e.g. with Flash Attention installed)
- Log in to Hugging Face if pulling gated models/tokenizers


### Basic Usage

Examples:
```bash
# Simple watermarking (.txt)
python -m apps.posthoc.main \
  --input_path assets/sample_document.txt \
  --dump_dir output/

# Advanced config (.txt with adaptive processing)
python -m apps.posthoc.main \
  --input_path assets/longer_sample_document.txt \
  --dump_dir output/ \
  --watermark.watermark_type greenlist \
  --watermark.delta 2.5 \
  --processing.temperature 0.9 \
  --processing.top_p 0.9 \
  --model.model_name meta-llama/Llama-3.2-1B-Instruct

# JSONL input (set the text key if not 'text')
python -m apps.posthoc.main \
  --input_path path/to/input.jsonl \
  --dump_dir output/ \
  --text_key content
```

## More details

### Python API

```python
from apps.posthoc.watermarker import PostHocWatermarker
from apps.common.watermark.core import WatermarkConfig
from apps.posthoc.config import PromptConfig, ModelConfig, ProcessingConfig

# Custom configuration
wm = PostHocWatermarker(
  watermark_config=WatermarkConfig(
    watermark_type="gumbelmax",
    ngram=2,
    secret_key=42,
  ),
  model_config=ModelConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    use_flash_attention=False,
  ),
  processing_config=ProcessingConfig(
    temperature=0.8,
    top_p=0.95,
  ),
  prompt_config=PromptConfig(
    system_message="You are a rephrasing assistant.",
    prefill_answer="Here is the rephrased text:\n",
  ),
)

text = open("assets/sample_document.txt").read()
results = wm.process_text(text)
print(results) # contains: orig_text, wm_text, wm_eval, etc.
```

### Running the main.py

#### Commands

Example command:

**On CPU**
```bash
python -m apps.posthoc.main --input_path "your/texts.jsonl" --text_key "text" --model.model_name meta-llama/Llama-3.2-1B-Instruct  --watermark.watermark_type "greenlist" --watermark.delta 1 --processing.temperature 0.8 --model.use_flash_attention false
```

**On GPU**, first allocate a GPU node, e.g.:
```bash
srun --partition=mypartition --nodes=1 --gpus-per-node=8 --cpus-per-gpu=24 --mem-per-cpu=8G --exclusive --qos=myqos --time=10:00:00 --job-name=dev_llmwm --pty bash 
conda activate text_seal
```
Then run, e.g.:
```bash
python -m apps.posthoc.main --input_path /path/to/HumanEval_processed.jsonl --model.model_name meta-llama/Llama-3.1-8B-Instruct  --watermark.watermark_type "gumbelmax" --processing.temperature 0.9 --model.use_flash_att
ention true --prompt.prefill_answer "Here is the rephrased code:\n" --prompt.preserve_style false --prompt.preserve_format false  --evaluation.enable_code_evaluation true
```


#### Processing Methods

The system selects a processing strategy based on input format and document characteristics:

**Text files (.txt)**
- Automatically chooses between full-document or adaptive chunking based on length
- Short documents (< 500 chars): processed as a single unit for optimal quality
- Long documents: split adaptively at natural boundaries with overlapping context

**JSONL files (.jsonl)**
- Processes each record independently as a separate document
- Each entry is watermarked individually with its own evaluation metrics
- Use `--text_key` to specify which field contains the text (defaults to 'text')


#### Config Keys (nested flags or YAML) - see below for details
- Model: `--model.model_name`, `--model.use_flash_attention`, `--model.compile_model`
- Watermark: `--watermark.watermark_type`, `--watermark.delta`, `--watermark.gamma`, `--watermark.ngram`, `--watermark.secret_key`, `--watermark.method`, `--watermark.alpha`, `--watermark.depth`
- Processing: `--processing_method` (`full|adaptive|auto`), `--processing.temperature`, `--processing.top_p`, `--processing.max_gen_len`, `--processing.target_chunk_size`, `--processing.max_chunk_size`, `--processing.overlap_ratio`
- Prompt: `--prompt.system_message`, `--prompt.user_message_template`, `--prompt.custom_instruction`, `--prompt.preserve_length`, `--prompt.preserve_style`, `--prompt.preserve_format`
- Evaluation: `--evaluation.enable_semantic_similarity`, `--evaluation.enable_rouge`, `--evaluation.enable_bleu`

You can also pass a YAML config with `--config path/to/config.yaml` and override any key via CLI.

#### Output

The system returns a dict with watermark and quality metrics. Example:

```json
{
  "wm_text": "The rephrased watermarked text...",
  "orig_text": "The original input text...",
  "wm_eval": {
    "score": 1.45,
    "p_value": 7.64e-05,
    "det": true,
    "toks_gen": 93,
    "toks_scored": 91,
    "entropy_filtered": false,
    "entropy_threshold": null
  },
  "quality": {
    "bleu_score": 0.086,
    "rouge_scores": {
      "rouge1": {"precision": 0.478, "recall": 0.582, "fmeasure": 0.525},
      "rouge2": {"precision": 0.167, "recall": 0.204, "fmeasure": 0.183},
      "rougeL": {"precision": 0.463, "recall": 0.564, "fmeasure": 0.508}
    },
    "semantic_similarity": 0.759,
    "length_stats": {
      "original_length": 350,
      "rephrased_length": 460,
      "length_ratio": 1.314
    }
  },
  "stats": {
    "orig_len": 350,
    "wm_len": 460,
    "orig_toks": 71,
    "wm_toks": 95,
    "tok_ratio": 1.338
  },
  "times": {
    "t_rephrase": 258.45,
    "tps": 0.368,
    "t_wm_eval": 0.005,
    "t_quality": 0.050,
    "t_total": 258.51
  },
  "input_path": "assets/sample_document.txt"
}
```



### Configs

You can pass a YAML file via `--config` and/or override any key with nested CLI flags (e.g., `--model.model_name ...`). Below are the main configuration groups and their important fields.

#### ModelConfig
- `model_name` (str): HF model id, e.g. `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.
- `use_flash_attention` (bool): Enable Flash Attention for faster/cheaper GPU inference.
- `compile_model` (bool): Torch compile for speed; can increase warmup time.


#### WatermarkConfig

- `watermark_type` (str): `greenlist`, `gumbelmax`, `dipmark`, `synthid`, `morphmark`, `watermax`, or `none` (vanilla).
- `delta` (float): Strength of bias toward the greenlist (higher = stronger watermark; too high can hurt quality).
- `gamma` (float in [0,1]): Fraction of vocab in the greenlist (typical 0.25–0.5).
- `ngram` (int): N-gram window for greenlist decisions (1–3 common).
- `secret_key` (int): Secret seed to generate watermark partitions (controls uniqueness/attribution).
- `method` (str): hashing/PRF family used by the algorithm, typically `binary` or `uniform`.
- `alpha` (float): DipMark parameter controlling the cumulative-mass split for reweighting.
- `depth` (int): SynthID tournament depth controlling how many iterative reweighting layers are applied.
- `k_morphmark` (float): MorphMark adaptive strength factor (default 1.30); controls how strongly watermark strength adapts to green token probability.
- `p_0` (float): MorphMark threshold parameter (default 0.15); minimum green probability to apply adaptive watermarking.
- `chunk_size` (int): WaterMax parameter L - number of tokens per draft sequence (default 4).
- `num_drafts` (int): WaterMax parameter m - number of candidate draft sequences to generate (default 4).
- `base_watermark` (str): WaterMax underlying watermark type for scoring (e.g., "greenlist", "gumbelmax", "synthid").
- `scoring_method` (str): Deduplication strategy for watermark detection (default: "v2")
  - `"v1"`: Deduplicate by n-gram window only (same context = count once)
  - `"v2"`: Deduplicate by window + target token (same context + token = count once)
  - `"v0"` or `"none"`: No deduplication (count all tokens)
  - Use v1/v2 for cleaner p-values by avoiding repeated patterns; use v0 for raw token counts.

**Available algorithms**:
- `greenlist` — Greenlist-based watermarking from Kirchenbauer et al. (2023); tunable via `delta` (strength) and `gamma` (greenlist size). Requires `method: binary`.
- `gumbelmax` — Uses a different sampling; control the strength with `temperature`. Requires `method: uniform`.
- `dipmark` — Distribution-preserving watermark (DiP); reweights the top-p mass using an `alpha` threshold after sorting by probability. Uses `method: uniform` during generation, and same detection as Green-list/Red-list.
- `morphmark` — Flexible adaptive watermarking with dynamic strength adjustment based on green token probability. Controlled by `k_morphmark` (adaptive strength factor) and `p_0` (threshold).
- `synthid` — Tournament-based watermark; computes per-token g-values across a given `depth` and iteratively updates probabilities. Requires `method: binary`.
- `watermax` — Multi-draft selection method that generates multiple candidate sequences and selects the one with highest watermark score. Controlled by `chunk_size` (L tokens per draft), `num_drafts` (m candidates), and `base_watermark` (underlying watermark type).
- `none`/`vanilla` — No watermarking; use as a baseline paraphrasing mode.


#### ProcessingConfig

- `temperature` (float): Sampling temperature. Start 0.7–1.0; lower = more conservative.
- `top_p` (float): Nucleus sampling. Typical 0.9–0.95.
- `max_gen_len` (int): Safety cap for generated tokens per chunk.
- `target_chunk_size` (int): Desired chunk size for adaptive processing.
- `max_chunk_size` (int): Hard cap; anything above will be chunked.
- `overlap_ratio` (float): Context overlap between adjacent chunks to preserve coherence.
- `beam_width` (Optional[int]): Enable beam search with specified number of beams. None or 1 = standard sampling.
- `candidates_per_beam` (Optional[int]): Number of candidates to maintain per beam (defaults to beam_width).
- `stochastic_beam` (bool): Use stochastic beam search (True) vs deterministic (False). Stochastic adds sampling noise.
- `use_biased_for_scoring` (bool): Score candidates with watermarked probabilities (True) vs original model probabilities (False).

**Note:** Beam search with multi-GPU is not currently tested. For best results, use single-GPU inference when enabling beam search.

Runtime method selection:
- `processing_method` (CLI-only high-level switch): `full`, `adaptive`, or `auto`.
  - full: process entire text at once (best quality within context limits).
  - adaptive: splits long text using natural boundaries + overlaps.
  - auto: chooses full vs adaptive based on token length.

#### PromptConfig
- `system_message` (str): High-level role/instructions (e.g., "Professional rephrasing assistant").
- `user_message_template` (str): Template with `{text}` placeholder for the document.
- `prefill_answer` (str): Pre-filled text that will be appended after the sytem and user prompts.
- `custom_instruction` (str): Extra constraints (e.g., "Maintain technical accuracy").
- `preserve_length` (bool): Keep output roughly same length.
- `preserve_style` (bool): Retain tone/voice.
- `preserve_format` (bool): Keep headings/lists/code blocks formatting.

#### EvaluationConfig
- `enable_semantic_similarity` (bool): Compute embedding-based similarity.
- `enable_rouge` (bool), `enable_bleu` (bool): Compute ROUGE/BLEU.- `enable_perplexity` (bool): Compute perplexity of rephrased text using a separate quality model (default: True). Set to False to skip perplexity computation if resource-constrained.
- `quality_model_name` (str): Base model for perplexity evaluation (default: "mistralai/Mistral-7B-v0.3"). Should be a base model (not instruct-tuned) and different from the watermarking model. **Note:** Loading this model requires ~14GB GPU memory. Disable perplexity evaluation if GPU memory is limited.- `entropy_threshold` (Optional[float] or str): Apply entropy-based filtering during watermark detection. Only score tokens with entropy above this threshold. Can also provide a comma-separated string like `"none,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0"` to compute p-values across multiple thresholds simultaneously (useful for finding optimal threshold). **Note:** Entropy is computed using the same model used for watermark generation.
- Detection metrics are always computed for watermarked outputs: p-value, green ratio, counts.


### Code evaluation

You can enable code evaluation by setting `--evaluation.enable_code_evaluation true`, with a code dataset. 
Please see [textseal/setup/process_humaneval.py](https://github.com/facebookresearch/textseal/blob/main/setup/process_humaneval.py) and [textseal/setup/process_mbpp.py](https://github.com/facebookresearch/textseal/blob/main/setup/process_mbpp.py) for how to create the corresponding code datasets.

