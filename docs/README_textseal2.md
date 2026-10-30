# TextSeal2

Paper: [arXiv:2610.40613](https://arxiv.org/abs/2610.40613).

TextSeal2 is a distortion-free, two-key generation-time watermark that keeps working under speculative decoding. It lives in `textseal.watermarking` next to the other methods (`watermark_type="textseal2"`), and `textseal/analysis/textseal2/` holds the scripts that reproduce the paper's experiments.

## Method

**Sampling.** For a window `w` of the last 3 tokens and every candidate token `v`, each key gives a uniform `u_k(w, v)` and a Gaussian `G_k = Φ⁻¹(u_k)`. The two keys are merged into a virtual key `G* = (G_A + G_B) / √2`, blended with fresh noise, `z = ρ G* + √(1 − ρ²) η`, and turned into an exponential clock `E_v = −log Φ(z)`. The next token is the winner of the race `argmin_v E_v / p_v`. The clocks are i.i.d. Exp(1) over keys and noise, so every token follows the model exactly (distortion-free); `ρ` trades diversity for detection.

**Detection.** Each (window, token) pair is scored once with `s = −log Φ(−G*)`, which is Exp(1) without watermark; the sum over `n` pairs is tested against Gamma(n, 1). Key A alone gives a public test.

**Speculative decoding.** Drafts are races on the drafter's law. With block verification (Sun et al., 2025), the kept length and the last token are drawn jointly by a race on the leftovers of the draft races (the *residual-J race*), so the output law is exactly the target's and the acceptance is exactly block verification's, while the emitted tokens stay keyed. A window's keyed values are used at most once per answer.

## Usage

```python
from dataclasses import replace

from transformers import AutoModelForCausalLM, AutoTokenizer
from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import build_detector
from textseal.watermarking.generator import build_generator
from textseal.watermarking.speculative import SpeculativeGenerator

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device_map="cuda")
cfg = WatermarkConfig(watermark_type="textseal2", secret_key=42, ngram=3, rho=0.9, prf="hires")

# Without speculative decoding (batched).
ids = build_generator(target, tok, cfg).generate(prompts, max_gen_len=200, temperature=1.0, top_p=0.95, return_ids=True)

# With speculative decoding: Qwen3.5-0.8B drafts 8 tokens per call, block verification, residual-J race.
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype="bfloat16", device_map="cuda")
spec = SpeculativeGenerator(draft, target, tok, replace(cfg, num_speculative_tokens=8))
out = spec.generate_one(prompt_ids, max_new=200)  # ids plus per-token / per-call logs (sources, laws, timings)

# Detection on token ids (or on texts).
det = build_detector(tok, cfg)
scores = det.get_scores_by_t([out["ids"]], scoring_method="v2", seen_windows=set())[0]
p_value = det.get_pvalue(sum(scores), len(scores))
```

`WatermarkConfig` fields used by TextSeal2 and the baselines of the paper:

| Field | Meaning |
|---|---|
| `watermark_type` | `textseal2`, or the baselines `textseal`, `automark` (AutoMark's one-key copula), `textgrain`, `synthid` |
| `rho` | Copulas: weight of the keyed Gaussian vs fresh noise |
| `prf` | `hires` (53-bit splitmix64, used for the copulas) or `textseal` (TextSeal's 13-bit hash) |
| `repeated_windows` | `reuse`, `skip` (sample without watermark on a repeated window) or `per_key` (TextSeal: other key, then no watermark) |
| `num_speculative_tokens`, `verification`, `spec_decision` | Speculative decoding: drafts per call, `block` / `token`, TextSeal2 decision rule (`residual_j` by default) |
| `synthid_hash` | `official`: Google DeepMind's g-values (`pip install synthid-text`) |
| `textgrain_beta`, `textgrain_blocks`, `textgrain_columns`, `textgrain_solver`, `textgrain_verify_watermarked` | textGrain, implemented from its report (`textgrain.py`; `chunked32` uses a Triton kernel) |

## Code

| File | Content |
|---|---|
| `watermarking/core.py` | 53-bit PRF (`prf_hires`, `keyed_uniform`), copula clocks and scores, races, official SynthID g-values |
| `watermarking/generator.py` | `CopulaGenerator` (TextSeal2, AutoMark), `TextGrainGenerator`, TextSeal's `per_key` and SynthID's official options |
| `watermarking/detector.py` | `TextSeal2Detector`, `TextGrainDetector`, token-id inputs for every detector |
| `watermarking/speculative.py` | `SpeculativeGenerator` (KV-cached, rollback for hybrid models such as Qwen3.5) and the baselines' block verification |
| `watermarking/verification.py` | Block verification law, the residual-J race and its ablations, the token-level race coupling (with exactness self-tests) |
| `watermarking/textgrain.py`, `sinkhorn_triton.py` | textGrain (Algorithms 1 and A.1 of its report) |
| `analysis/textseal2/` | Paper scripts: `generate.py`, `detect.py`, `analyze.py`, `figures.py`, `common.py` (run names → configs) |

## Reproducing the paper

### Data

```bash
pip install -e ".[textseal2]"
python -m textseal.analysis.textseal2.prepare_data --out_dir data
```

This writes the prompt files that the scripts read by default (or set `TEXTSEAL2_ELI5` / `TEXTSEAL2_AIME`):

| File | Source | Content |
|---|---|---|
| `data/eli5.jsonl` | [sentence-transformers/eli5](https://huggingface.co/datasets/sentence-transformers/eli5), train split | 267,901 questions with human answers, in the original order, without the math-heavy rows listed in `textseal/analysis/textseal2/eli5_dropped_rows.txt.gz`. Detection uses the first 3000 questions, diversity the first 100 (16 or 32 answers each), the unwatermarked calibration set rows 100,000 to 109,999. |
| `data/aime_all.jsonl` | [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) | 933 AIME problems (id, question, answer, year); the paper samples 100 with seed 0 and asks for a `\boxed{}` answer. |

For code, MBPP prompts come from `setup/process_mbpp.py` (pass `--code_file` to `generate`).

### Runs

Paper settings: Qwen3.5-9B (target) and Qwen3.5-0.8B (drafter), temperature 1, top-p 0.95, window 3, 200 new tokens on ELI5, thresholds at a false-positive rate of 1e-3 calibrated on 10 keys × 10,000 unwatermarked answers. Every script shards with `--shard k --n_shards K` and resumes.

| Paper item | Commands |
|---|---|
| Null set for thresholds | `generate --configs ns_none --kinds det --n_det 10000 --eli5_offset 100000` |
| Table 1, Fig. 1 (speculative panel), number of drafts | `generate --configs residj:{0.7,0.8,0.85,0.9,0.925,0.95,0.99} tsb:{0.5,0.7,0.8,0.9,0.95} tgb:{0.25,0.5,0.75} tgpb:{0.25,0.5,0.75} sidb:{2,3,5,10,15,20} amb:{0.7,…,0.99} noneb --ndrafts 4 8 16` |
| Fig. 1 (without speculative decoding), three models | `generate --configs ns_ts2:… ns_ts:… ns_tg:… ns_sid:… ns_am:… ns_none --n_div_answers 32 [--target …]` |
| Distortion tests | `generate --kinds dist --dist_configs residj:0.9 residjraw:0.9 tsb:0.5 … noneb none` |
| AIME | `generate --datasets aime --n_aime_answers 16 --max_new_aime 8192 --configs …` then `detect --dataset aime --lengths 256 512 1024 2048` |
| Second drafter–target pair | `generate --draft meta-llama/Llama-3.2-1B-Instruct --target meta-llama/Llama-3.1-8B-Instruct --prf textseal --configs …` |
| Code (MBPP) | `generate --datasets code --code_file mbpp_prompts.jsonl --configs ns_…` (pass@1 with `textseal.watermarking.evaluation`) |
| Detection, entropy weighting | `detect entropy …` then `detect pv --entropy --gen_dirs … --null_dir …/eli5/det/ns_none --out …` |
| Tables, figures | `analyze all --gen_dirs … --pv_dir … --out …` then `figures --analysis … --detect … --ndraft 8` |

Commands are `python -m textseal.analysis.textseal2.<script>`. Under speculative decoding the records also store, per token, the source (draft / replaced / bonus), the target and drafter probabilities and entropies, and per call the kept drafts and timings, so acceptance, tokens per call, latency and distortion come from the same runs.
