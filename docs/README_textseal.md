# TextSeal — Dual-Key Generation-Time Watermarking

[[Paper](https://arxiv.org/abs/2605.12456)]

This directory contains the **TextSeal method**: dual-key Gumbel-max watermarking for LLM text generation.

Licensed under [Apache 2.0](../../LICENSE). See [NOTICE](../../NOTICE) for attributions.

## Install

```bash
conda create -n textseal python=3.11 && conda activate textseal
pip install -e .
```

If your models are already cached elsewhere, you can redirect Hugging Face caching:

```bash
export HF_HOME=/path/to/cache
# or pass --cache_dir /path/to/cache to demo.py
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import TextSealDetector
from textseal.watermarking.generator import TextSealGenerator

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

# Generate
wm_config = WatermarkConfig(secret_key=42)
generator = TextSealGenerator(model, tokenizer, wm_config)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain why the sky is blue."}],
    tokenize=False, add_generation_prompt=True,
)
texts = generator.generate([prompt], max_gen_len=200, temperature=0.8, top_p=0.95)

# Detect
detector = TextSealDetector(tokenizer, wm_config, model=model)
result = detector.detect(texts[0])
print(f"p-value: {result['p_value']:.2e}, detected: {result['detected']}")
```

## How It Works

**Generation** — At each token position, randomly pick Key A (prob `alpha`) or Key B.
Hash the ngram context + chosen key → PRF value `r ∈ [0,1)` per vocab entry.
Select `argmax(log(r) / p)` where `p` is the model probability (Gumbel-max trick).

**Detection** — For each token, compute scores under both keys:
`fused = alpha * score_A + (1-alpha) * score_B`.
Under H0: known mean and variance → Gamma moment-matching test.
Entropy weighting gives more weight to high-entropy tokens.

**Deduplication** — `v2` (default): only score each unique (context window, token) pair once.

**Localized detection** — O(N log N) geometric cover search with Bonferroni correction:
`p_final = min(p_global, p_single_loc, p_multi_loc) * 3`.

**Speculative decoding** — Draft K tokens with Key A, target verifies.
Rejected tokens resampled from `(P_target - P_draft)+` with Key B.
100% of tokens carry watermark signal.

## Files

| File | Description |
|------|-------------|
| `config.py` | WatermarkConfig, dual-key constants |
| `core.py` | PRF helpers, dual-key kernels |
| `generator.py` | TextSealGenerator and SpeculativeGenerator |
| `detector.py` | TextSealDetector and localized detection |
