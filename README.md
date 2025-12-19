# Text Seal

Meta Text Seal is a comprehensive toolkit for LLM generation-time watermarking, post-hoc text watermarking through LLM rephrasing, and contamination detection through watermark radioactivity.
It is part of the [Meta Seal](https://facebookresearch.github.io/meta-seal) family of watermarking technologies.

## Features

- 🔏 **Post-hoc Watermarking**: Rephrase text with an LLM while inserting a watermark using generation-time scheme (Green-list/Red-list, Gumbel-max, DipMark, SynthID, MorphMark, WaterMax).
- 🧪 **Contamination Detection**: Detect watermarked dataset membership inference through radioactivity.
- 🚀 **Training Infrastructure**: Distributed pretraining and SFT with contamination injection support for research purposes.


## Papers

This codebase implements methods from:

- **[How Good is Post-Hoc Watermarking With Language Model Rephrasing?](https://ai.meta.com/research/publications/how-good-is-post-hoc-watermarking-with-language-model-rephrasing/)**: 
Post-hoc watermarking through rephrasing with a watermarked LLM.

- **[Detecting Benchmark Contamination Through Watermarking](https://ai.meta.com/research/publications/detecting-benchmark-detection-through-watermarking/)**:
Detecting training data contamination with watermarked benchmarks.

## Quick Start

### Installation

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

Watermark existing text by rephrasing with an LLM (here using Gumbel-max watermarking and Llama-3.2-3B-Instruct):
```bash
python -m apps.posthoc.main \
  --input_path assets/sample_document.txt \
  --dump_dir output/ \
  --watermark.watermark_type gumbelmax \
  --model.model_name meta-llama/Llama-3.2-3B-Instruct \
  --processing.temperature 1.0 \
  --processing.top_p 0.95
```
Results are saved in `output/` directory as a JSONL file containing original, watermarked text and statistics.

See [docs/README_posthoc.md](docs/README_posthoc.md) for detailed documentation.

### Contamination Detection

Inject watermarked benchmarks during training and detect memorization through watermark radioactivity.

Download DCLM training data and benchmark datasets (ARC-Easy, ARC-Challenge, MMLU). See [Data Preparation](docs/README_contamination.md#data-preparation) in the contamination docs.

The contamination detection workflow consists of three steps, each with its own experiment configuration file:

```bash
# Step 1: Watermark benchmarks with different secret keys
python -m apps.posthoc.main --config configs/watermark_benchmarks.yaml

# Step 2: Train model with contaminated watermarked data
python -m apps.common.stool script=apps.wmtraining.train \
  config=configs/train_with_contamination.yaml \
  nodes=4 ngpu=8 partition=learn qos=high time=4320

# Step 3: Detect contamination via watermark evaluation
python -m apps.wmtraining.eval_wm --config configs/eval_contamination.yaml
```

**Configuration files:**
- [configs/watermark_benchmarks.yaml](configs/watermark_benchmarks.yaml) - Watermark benchmark datasets
- [configs/train_with_contamination.yaml](configs/train_with_contamination.yaml) - Train with contamination injection
- [configs/eval_contamination.yaml](configs/eval_contamination.yaml) - Evaluate contamination detection

See [docs/README_contamination.md](docs/README_contamination.md) for detailed documentation.

## Documentation

- **[Post-hoc Watermarking](docs/README_posthoc.md)** - Rephrase text while adding a watermark
- **[Contamination Detection](docs/README_contamination.md)** - Detect benchmark memorization via watermarks

## Repository Structure

```
textseal/
├── apps/
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


## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/facebookresearch/textseal/issues)
- 💬 [Discussions](https://github.com/facebookresearch/textseal/discussions)


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