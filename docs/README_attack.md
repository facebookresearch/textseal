# Attacks

## Rephrasing Robustness

Rephrase each watermarked text with another model and detect again, to test how well the watermark survives rephrasing.

### Integrated Mode

Watermark and attack in one run: set `--attack_model_name`, and each text is rephrased once per temperature in `--attack_temperatures`.

```bash
python -m textseal.watermarking.main \
    --input_path assets/sample_document.txt \
    --model.model_name meta-llama/Llama-3.2-1B-Instruct \
    --watermark.watermark_type textseal --watermark.ngram 3 \
    --attack_model_name meta-llama/Llama-3.2-3B-Instruct \
    --attack_temperatures "0.5,0.8,1.0,1.2"
```

Each output row gets an `attacks` dictionary keyed by temperature:

```json
"attacks": {
  "0.8": {
    "attacked_text": "...",
    "attack_wm_eval": {"score": 0.45, "p_value": 0.05, "det": false},
    "attack_quality": {"semantic_similarity": 0.87}
  }
}
```

### Standalone Mode

Attack texts watermarked in an earlier run (`results.jsonl` with a `wm_text` field):

```bash
python -m textseal.attacks.run_attack --attack rephrasing --mode scrub --access nobox \
    --input output/results.jsonl --temperature 0.8 \
    --model_id meta-llama/Llama-3.2-3B-Instruct \
    --tokenizer_id meta-llama/Llama-3.2-1B-Instruct \
    --wm_config '{"watermark_type": "textseal", "ngram": 3}' --output_dir output/rephrased
```

## Informed Attacks: Public Detector Access

Removal (`--mode scrub`) and forgery (`--mode spoof`) attacks on dual-key watermarks: `textseal`, or `greenlist` / `synthid` with key routing (`"key_routing": true` in `--wm_config`). Key A is released as a public detector, key B stays private. The attacker reads the public detector at one of three access levels:

| Access | The attacker reads |
|--------|--------------------|
| `nobox` | nothing (no public detector) |
| `blackbox` | the public decision at `--threshold`, one bit per text |
| `whitebox` | the public detector's token-level scores |

Two attacks:

- **Rephrasing** (`--attack rephrasing`): a surrogate LLM (`--model_id`) rephrases the text. With white-box access its sampling is biased by each candidate token's score, either additively (`--sampling additive --strength 1.0`) or by Gumbel-max selection (`--sampling gumbel`, TextSeal only). With black-box access it rephrases at temperatures 0.2, 0.4, ..., `--temperature` and keeps the first paraphrase that crosses the public decision.
- **Word edits** (`--attack word_edits`): a masked LM replaces `--edit_fraction` of the words. With white-box access the words carrying the most (scrub) or least (spoof) signal go first; with black-box access the attack stops on the first word that crosses the public decision (`--overshoot m` keeps editing to `m` times the edits that took).

`--adaptive --overshoot m` (white-box) aims at `m` times past `--threshold`: rephrasing scales its bias by the distance to that aim, word edits stop once the public p-value reaches it.

```bash
# Remove the watermark by white-box rephrasing
python -m textseal.attacks.run_attack --attack rephrasing --mode scrub --access whitebox \
    --input output/results.jsonl --model_id meta-llama/Llama-3.2-3B-Instruct \
    --tokenizer_id meta-llama/Llama-3.2-1B-Instruct \
    --wm_config '{"watermark_type": "textseal", "ngram": 3}' --output_dir output/scrub

# Forge the watermark with black-box word edits
python -m textseal.attacks.run_attack --attack word_edits --mode spoof --access blackbox \
    --input data/clean.jsonl --edit_fraction 0.1 \
    --tokenizer_id meta-llama/Llama-3.2-1B-Instruct \
    --wm_config '{"watermark_type": "textseal", "ngram": 3}' --output_dir output/spoof
```

`--input` rows carry `wm_text` (scrub) or `clean_text` (spoof), and `--wm_config` is the config the text was watermarked with. Each row of `<output_dir>/results.jsonl` holds the source and attacked texts, the public / private / fused p-values of both, and the attacked text's quality against the source: BERTScore F1, MiniLM similarity, GPT-2 perplexity ratio and word overlap.

### Tampering Test

An attacker steering on the public detector moves key A's scores and leaves key B's alone. The test contrasts the two, `T = Σ_t (s_A,t - s_B,t)`, against a null that swaps the key labels per token (exact at `mixing_alpha = 0.5`):

```bash
python -m textseal.watermarking.tamper --input output/scrub/results.jsonl --text_key attacked_text \
    --tokenizer_id meta-llama/Llama-3.2-1B-Instruct \
    --wm_config '{"watermark_type": "textseal", "ngram": 3}'
```

Each row gets `p_scrub` (public channel deflated: removal) and `p_spoof` (public channel inflated: forgery).

### Python API

Remove the watermark from `wm_text` with white-box word edits, then detect:

```python
import random
from transformers import AutoTokenizer
from textseal.attacks import oracle
from textseal.attacks.word_edits import MLMProposer, WordEditAttack
from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import build_detector, text_channels
from textseal.watermarking.tamper import TamperTest

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")  # the watermarked model's
wm_config = WatermarkConfig(watermark_type="textseal", ngram=3)  # as at generation

view = oracle.build("whitebox", tokenizer, wm_config)  # or "nobox", or "blackbox" with threshold=1e-3
mlm = MLMProposer.load("roberta-large", "cuda", min_prob=1e-4)
attack = WordEditAttack(view, mlm, tokenizer, random.Random(0), goal="scrub", edit_fraction=0.2)
attacked = attack.run(wm_text)["text"]

channels = text_channels(build_detector(tokenizer, wm_config), attacked)
print({name: ch["p_value"] for name, ch in channels.items()})  # public / private / fused
print(TamperTest(tokenizer, wm_config)(attacked))             # T, p_scrub, p_spoof
```

`RephrasingAttack(model, tokenizer, view, goal=..., wm_tokenizer=tokenizer)` from `textseal.attacks.rephrasing` takes the same oracle, with `run_batch(texts)` in place of `run(text)`.
