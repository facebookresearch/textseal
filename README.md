# Text Seal

Meta Text Seal is a comprehensive toolkit for LLM generation-time watermarking, post-hoc text watermarking through LLM rephrasing, and contamination detection through watermark radioactivity.
It is part of the [Meta Seal](https://facebookresearch.github.io/meta-seal) family of watermarking technologies.

[[`post-hoc paper`](https://arxiv.org/abs/2512.16904)] 
[[`contamination paper`](https://ai.meta.com/research/publications/detecting-benchmark-detection-through-watermarking/)] 
[[`meta seal`](https://facebookresearch.github.io/meta-seal)]
[[`colab`](https://colab.research.google.com/github/facebookresearch/textseal/blob/main/notebooks/textseal_colab.ipynb)]

## Features

- 🔏 **Post-hoc Watermarking**: Rephrase text with an LLM while inserting a watermark using generation-time scheme (Green-list/Red-list, Gumbel-max, DipMark, SynthID, MorphMark, WaterMax, etc.).
- 🧪 **Contamination Detection**: Detect watermarked dataset membership inference through radioactivity.
- 🚀 **Training Infrastructure**: Distributed pretraining and SFT with contamination injection support for research purposes.

## Papers

This codebase implements methods from:
- **[How Good is Post-Hoc Watermarking With Language Model Rephrasing?](https://arxiv.org/abs/2512.16904)**: 
Post-hoc watermarking through rephrasing with a watermarked LLM.
- **[Detecting Benchmark Contamination Through Watermarking](https://ai.meta.com/research/publications/detecting-benchmark-detection-through-watermarking/)**:
Detecting training data contamination with watermarked benchmarks.


## Quick Start

### Installation

**Option 1: pip install (fastest)**
```bash
pip install textseal
```

**Option 2: Install from source**
```bash
git clone https://github.com/facebookresearch/textseal.git
cd textseal
pip install -e .
```

### Python API

Watermark text using the Python API:

```python
from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig, ProcessingConfig

# Basic usage with defaults
watermarker = PostHocWatermarker()
result = watermarker.process_text("Your text here")
print(result["wm_text"])  # Watermarked text
print(result["wm_eval"]["p_value"])  # Detection p-value

# Custom configuration
watermarker = PostHocWatermarker(
    watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
    model_config=ModelConfig(model_name="meta-llama/Llama-3.2-3B-Instruct"),
    processing_config=ProcessingConfig(temperature=0.8, top_p=0.95),
)
result = watermarker.process_text("Text to watermark")
```

> 💡 **Tip: Increasing watermark strength**.
> For Gumbel-max watermarking, increase `temperature` for stronger watermarks (e.g., `temperature=1.2` in ProcessingConfig).
> For Greenlist watermarking, increase `delta` (e.g., `delta=3.0` in WatermarkConfig).
> See [Watermark Configuration Guide](docs/README_posthoc.md#watermark-strength) for details.

See [docs/README_posthoc.md](docs/README_posthoc.md) for detailed documentation on the configurations and usage.

**Common Use Cases:**

- **Watermarking + Detection**: Use `process_text()` to watermark text and get detection metrics (p-value, score) in one call.
- **Watermarking Only**: Use `rephrase_with_watermark()` to get just the watermarked text without evaluation.
- **Detection Only**: Set `enable_detection_only=True` and use `evaluate_watermark()` to check existing text for watermarks without loading the LLM.

See [docs/README_posthoc_api.md](docs/README_posthoc_api.md) for complete API usage examples

### Command Line Interface

After installing textseal, you get the `textseal-watermark` CLI command:

```bash
# Get help
textseal-watermark --help

# Watermark a file
textseal-watermark --input_path document.txt --dump_dir output/

# Detection-only mode
textseal-watermark --input_path text_to_check.txt --evaluation.enable_detection_only true
```



## Using the Repository

### Installation

**Option 3: Development setup**
```bash
# Clone the repository
git clone https://github.com/facebookresearch/textseal.git
cd textseal

# Create environment and install dependencies
conda create -n text_seal python=3.11.13
conda activate text_seal
pip install -r requirements.txt
```

> 💡 For contamination detection experiments (training with contamination injection), you need additional setup. First follow the [Meta Lingua installation instructions](https://github.com/facebookresearch/lingua#installation), then install the requirements above. See [Environment Setup](docs/README_contamination.md#environment-setup) for details.

### Post-hoc Watermarking

For batch processing or command-line workflows, use the CLI:

```bash
python -m textseal.posthoc.main \
  --input_path assets/sample_document.txt \
  --dump_dir output/ \
  --watermark.watermark_type gumbelmax \
  --model.model_name meta-llama/Llama-3.2-3B-Instruct \
  --processing.temperature 1.0 \
  --processing.top_p 0.95
```
Results are saved in `output/` directory as a JSONL file containing original, watermarked text and statistics.

### Contamination Detection

Inject watermarked benchmarks during training and detect memorization through watermark radioactivity.

Download DCLM training data and benchmark datasets (ARC-Easy, ARC-Challenge, MMLU). See [Data Preparation](docs/README_contamination.md#data-preparation) in the contamination docs.

The contamination detection workflow consists of three steps, each with its own experiment configuration file:

```bash
# Step 1: Watermark benchmarks with different secret keys
python -m textseal.posthoc.main --config configs/watermark_benchmarks.yaml

# Step 2: Train model with contaminated watermarked data
python -m textseal.common.stool script=textseal.wmtraining.train \
  config=configs/train_with_contamination.yaml \
  nodes=4 ngpu=8 partition=learn qos=high time=4320

# Step 3: Detect contamination via watermark evaluation
python -m textseal.wmtraining.eval_wm --config configs/eval_contamination.yaml
```

**Configuration files:**
- [configs/watermark_benchmarks.yaml](configs/watermark_benchmarks.yaml) - Watermark benchmark datasets
- [configs/train_with_contamination.yaml](configs/train_with_contamination.yaml) - Train with contamination injection
- [configs/eval_contamination.yaml](configs/eval_contamination.yaml) - Evaluate contamination detection

See [docs/README_contamination.md](docs/README_contamination.md) for detailed documentation.


## Documentation

- **[API Usage Guide](docs/README_posthoc_api.md)** - Common use cases (detection-only, watermarking-only, etc.)
- **[Post-hoc Watermarking](docs/README_posthoc.md)** - Rephrase text while adding a watermark
- **[Attack Simulation](docs/README_attack.md)** - Test watermark robustness against rephrasing attacks
- **[Contamination Detection](docs/README_contamination.md)** - Detect benchmark memorization via watermarks


## Repository Structure

```
textseal/
├── textseal/
│   ├── posthoc/          # Post-hoc watermarking
│   ├── wmtraining/       # Training and evaluation
│   ├── analysis/         # Analysis tools
│   └── common/           # Shared utilities (LLM, watermark, config)
├── docs/                 # Detailed documentation
├── configs/              # Example configurations for watermarking and training
├── assets/               # Sample texts
├── setup/                # Setup scripts and data processing
```

## Use Cases

### 1. Content Authentication
Watermark text to enable verification and provenance tracking.

### 2. Dataset Contamination Detection
Detect if evaluation benchmarks were included in training data by injecting watermarked versions and checking for "radioactivity."

### 3. Research on Watermarking
Experiment with different watermarking algorithms and detection methods on your own models and datasets.


## License

Meta Text Seal is released under the [MIT License](LICENSE).

It relies on code and models from other repositories. 
The contamination detection app builds on [Meta Lingua](https://github.com/facebookresearch/lingua) for training, which has a BSD 3-Clause License.
The models used for post-hoc watermarking are loaded from [Hugging Face](https://huggingface.co/) and are subject to their respective licenses.


## Citation

If you use Text Seal in your research, please cite:

```bibtex
@article{sander2025detecting,
  title={Detecting benchmark contamination through watermarking},
  author={Sander, Tom and Fernandez, Pierre and Mahloujifar, Saeed and Durmus, Alain and Guo, Chuan},
  journal={arXiv preprint arXiv:2502.17259},
  year={2025}
}

@article{fernandez2025how,
  title={How Good is Post-Hoc Watermarking With Language Model Rephrasing?},
  author = {Fernandez, Pierre and Sander, Tom and Elsahar, Hady and Chang, Hongyan and Sou\v{c}ek, Tom\'{a}\v{s} and Lacatusu, Valeriu and Tran, Tuan and Rebuffi, Sylvestre-Alvise and Mourachko, Alexandre},
  year={2025}
}
```
