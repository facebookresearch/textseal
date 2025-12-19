# Benchmark Contamination Detection

> **Note:** This contamination system is built on top of [Meta Lingua](https://github.com/facebookresearch/lingua)'s training infrastructure. The contamination injection and watermark-based detection are the novel contributions of this project.

This feature allows you to artificially contaminate training with another source at specific steps during LLM pre-training. 
If the contaminated batches are watermarked, the codebase can also be used to detect traces of these watermarks in the trained model.
In the case of watermarked benchmarks as studied in the paper "[Detecting Benchmark Contamination Through Watermarking](https://arxiv.org/abs/2502.17259)", you can prove that the model was exposed to specific benchmarks during training ("radioactivity").

**Use cases:**
- Detect if evaluation benchmarks were in training data
- Study how contamination affects model performance
- Test contamination detection methods

## Environment Setup

> **Note:** This section is **only required for contamination detection experiments** (training with contamination injection and radioactivity detection). For post-hoc watermarking only, the basic `requirements.txt` installation is sufficient.

**For contamination detection experiments (training and evaluation), you need additional dependencies beyond the basic requirements.**

### Installation Order

1. **First, set up the Lingua environment** following the [Meta Lingua installation instructions](https://github.com/facebookresearch/lingua#installation). This includes:
   - `xformers` for efficient attention
   - Training-specific packages (FSDP, distributed training support)
   - Tokenizer setup

2. **Then, install additional packages from this repository:**
   ```bash
   conda activate your_lingua_env
   pip install -r requirements.txt
   ```
   This adds watermarking-specific dependencies like `sentence-transformers`, `rouge-score`, etc.

## Quick Start

The complete contamination detection workflow consists of three steps:

### Step 1: Watermark Benchmarks
```bash
python -m apps.posthoc.main --config configs/watermark_benchmarks_mmlu.yaml
```
See [configs/watermark_benchmarks_mmlu.yaml](../configs/watermark_benchmarks_mmlu.yaml) for configuration details.

**Note:** You'll need to run this command multiple times with different `input_path`, `secret_key`, and output locations to watermark all three benchmarks (arc_easy, arc_challenge, mmlu) with different keys.

### Step 2: Train with Contamination
```bash
python -m apps.common.stool script=apps.wmtraining.train \
  config=configs/train_with_contamination.yaml \
  nodes=4 ngpu=8 partition=learn qos=high time=4320
```
See [configs/train_with_contamination.yaml](../configs/train_with_contamination.yaml) for configuration details.

The `contamination_data.root_dir` should point to the directory containing subdirectories for each benchmark (`arc_easy/`, `arc_challenge/`, `mmlu/`). The benchmark names in `contamination_data.sources` must match the subdirectory names in the `root_dir`.


### Step 3: Detect Contamination
```bash
python -m apps.wmtraining.eval_wm --config configs/eval_contamination_mmlu.yaml
```
See [configs/eval_contamination_mmlu.yaml](../configs/eval_contamination_mmlu.yaml) for configuration details.

**Note:** To evaluate multiple benchmarks, you'll need to run this command multiple times with different `prompts_file` and `watermark.secret_key` values for each benchmark (arc_easy, arc_challenge, mmlu), or create separate config files for each.



## Overview

The contamination feature enables you to:
- Watermark datasets 
- Inject contaminated batches at specific training steps
- Control the contamination window (start and end steps)
- Specify the number of contamination batches
- Use a separate data source for contamination


## Configuration Parameters

Add the following parameters to your training configuration YAML file:

### Required Parameters (all must be set together)

- **`contamination_start_step`** (int): The training step at which to start contamination
- **`contamination_end_step`** (int): The training step at which to end contamination
- **`contamination_num_batches`** (int): Total number of contamination batches to inject
- **`contamination_data`** (DataArgs): Data configuration for the contamination source

The contamination batches are evenly distributed between `contamination_start_step` and `contamination_end_step`.

## Data Preparation

### Downloading Benchmarks

Before running contamination experiments, download the evaluation benchmarks from HuggingFace using the provided script:

```bash
python apps/analysis/download_benchmarks.py --output_dir assets/benchmarks
```

**Script:** [apps/analysis/download_benchmarks.py](../apps/analysis/download_benchmarks.py)

**Options:**
- `--output_dir`: Output directory for benchmark files (default: `assets/benchmarks`)
- `--mmlu_samples`: Number of MMLU test samples to use (default: `5000`)
- `--seed`: Random seed for sampling (default: `42`)

**Result:** Creates benchmark files in `assets/benchmarks/` with proper JSONL format:
```
assets/benchmarks/
├── arc_easy/
│   └── arc_easy.chunk.0.jsonl
├── arc_challenge/
│   └── arc_challenge.chunk.0.jsonl
└── mmlu/
    └── mmlu.chunk.0.jsonl
```

**Token counts (approximate):**
- ARC-Easy: ~112,000 tokens
- ARC-Challenge: ~64,000 tokens  
- MMLU: ~325,000 tokens (5000 samples)
- **Total: ~501,000 tokens**

These numbers are used for proportional sampling during contamination training.

### Downloading Training Data (DCLM)

For experiments that reproduce paper results, you need the DCLM dataset for main pretraining.

```bash
# Download via HuggingFace CLI
huggingface-cli download mlfoundations/dclm-baseline-1.0 \
  --repo-type dataset \
  --local-dir /path/to/datasets/dclm
```

**Or with Python:**
```python
from datasets import load_dataset

dclm = load_dataset(
    "mlfoundations/dclm-baseline-1.0",
    cache_dir="/path/to/datasets/dclm"
)
```

**Training config expects:**
```yaml
data:
  root_dir: "/path/to/training/data"  # Parent directory containing DCLM
  sources: {"dclm": 1.0}  # Subdirectory name
```

The data loader will look for DCLM files in `/path/to/training/data/dclm/`.

### Downloading Tokenizer

The training uses a TikToken tokenizer. To download the tokenizer model, follow the instructions in the [Meta Lingua README](https://github.com/facebookresearch/lingua#tokenizer) for obtaining tokenizer files.

**Training config expects:**
```yaml
data:
  tokenizer:
    name: tiktoken
    path: /path/to/tokenizer.model  # Path to downloaded tokenizer
```



#### Organizing Watermarked Output

After watermarking, organize the outputs into a training-ready structure with the naming pattern:
```
/path/to/watermarked/benchmarks/
├── arc_easy/
│   └── arc_easy.secret_key=0.chunk.0.jsonl
├── arc_challenge/
│   └── arc_challenge.secret_key=42.chunk.0.jsonl
└── mmlu/
    └── mmlu.secret_key=1234.chunk.0.jsonl
```


Each watermarked JSONL file contains:
- `wm_text`: The watermarked question 
- `wm_text_all`: quesiton + answer with template as used during contamination
- `text`: Original unwatermarked text
- `wm_eval`: Watermark evaluation statistics (green proportion, token counts)
- Other metadata from the original benchmark




#### Understanding Evaluation Results

The evaluation outputs a JSONL file (one dictionary per line) with results for each sample. The output file is saved to `{dump_dir}/results.jsonl` as specified in your config.

**Output Format:** Each line is a JSON dictionary representing one evaluation sample.

**Example output line:**
```json
{"prompt": "What species of creature does the peregrine belong to?", "input_tokens": [128000, 3923, 9606, 315, 17661, 1587, 279, 281, 486, 911, 483, 9352, 311, 30], "predicted_tokens": [791, 374, 315, 24032, 374, 433, 3492, 466, 911, 483, 617, 311, 1980, 3639], "multi_source_scores": {"source_0": 0.46153849363327026}, "multi_source_totals": {"source_0": 13}, "average_entropy": 4.125}
```

**Fields:**
- **prompt**: The input text from the benchmark
- **input_tokens**: Tokenized input sequence
- **predicted_tokens**: Model's predicted next tokens
- **multi_source_scores**: Proportion of predicted green tokens (for the green-red watermark)
- **multi_source_totals**: Number of tokens scored for each source

**Note on scoring:** By default, watermark detection performs deduplication between lines.


#### Using Different Tokenizers

The contamination system supports using different tokenizers for:
1. **Main training data**: Specified in `data.tokenizer.*`
2. **Contamination data**: Specified in `contamination_data.tokenizer.*`
3. **Watermark detection**: Specified in `wm_tokenizer_path` for evaluation

This allows experiments where:
- Benchmarks are watermarked with one tokenizer (e.g., Llama 3.1)
- Training uses a different tokenizer
- Evaluation uses cross-tokenizer detection to find watermarks

See [configs/eval_contamination_cross_tokenizer.yaml](../configs/eval_contamination_cross_tokenizer.yaml) for an example of cross-tokenizer evaluation configuration.


