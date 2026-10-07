"""Attack driver: rephrasing or word edits, removal (scrub) or forgery (spoof).

    --mode scrub   attack each `wm_text` of --input to remove the watermark
    --mode spoof   attack each `clean_text` of --input to forge it

--access sets what the attacker reads from the public detector (oracle.py). Writes one
row per text to <output_dir>/results.jsonl: the source and attacked texts, the public /
private / fused p-values of both under the deployed detector, and the attacked text's
quality against its source.

    python -m textseal.attacks.run_attack --attack word_edits --mode scrub --access whitebox \\
        --input results.jsonl --wm_config '{"watermark_type": "textseal", "ngram": 3}' \\
        --output_dir out
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textseal.attacks import oracle
from textseal.attacks.word_edits import MLMProposer, WordEditAttack
from textseal.attacks.rephrasing import RephrasingAttack
from textseal.attacks.utils import QualityScorer, load_jsonl, shard
from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import build_detector, text_channels

SOURCE_KEY = {"scrub": "wm_text", "spoof": "clean_text"}


def parse():
    ap = argparse.ArgumentParser(description="Watermark attack (rephrasing | word edits)")
    ap.add_argument("--attack", choices=["rephrasing", "word_edits"], required=True)
    ap.add_argument("--mode", choices=["scrub", "spoof"], required=True)
    ap.add_argument("--access", choices=["nobox", "blackbox", "whitebox"], default="whitebox",
                    help="What the attacker reads from the public detector. Black-box rephrasing "
                         "is the no-box rephrase with the detector as a stopping rule.")
    ap.add_argument("--input", required=True,
                    help="JSONL with `wm_text` (scrub) or `clean_text` (spoof) per row.")
    ap.add_argument("--wm_config", required=True,
                    help="The victim's watermark config as a JSON object.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tokenizer_id", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Tokenizer of the watermarked model (the detector's).")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="Published p-value threshold of the public detector: the black-box "
                         "decision, and the target the --adaptive attack aims past.")
    ap.add_argument("--adaptive", action="store_true",
                    help="(whitebox) rephrasing: scale the bias by the running public p-value; "
                         "edits: stop once the public p-value is on target.")
    ap.add_argument("--overshoot", type=float, default=1.0,
                    help="The overshoot factor m. White-box --adaptive aims at threshold*m (scrub) or "
                         "threshold/m (spoof); black-box word edits keep editing to m times the edits "
                         "the first crossing cost. 1 stops on the threshold.")
    ap.add_argument("--start_index", type=int, default=0,
                    help="First row of this shard; row `idx` is start_index + i.")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    reph = ap.add_argument_group("rephrasing")
    reph.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct",
                      help="Surrogate rephrasing model.")
    reph.add_argument("--sampling", choices=["additive", "gumbel"], default="additive",
                      help="additive: soft logit bias by +/-strength*log(r); "
                           "gumbel: Gumbel-max selection (spoof on r, scrub on 1-r).")
    reph.add_argument("--strength", type=float, default=1.0, help="Additive bias strength.")
    reph.add_argument("--temperature", type=float, default=1.0)
    reph.add_argument("--top_k", type=int, default=50)
    reph.add_argument("--max_new_tokens", type=int, default=400)
    reph.add_argument("--batch_size", type=int, default=16)
    edit = ap.add_argument_group("word edits")
    edit.add_argument("--mlm_model_id", default="roberta-large")
    edit.add_argument("--edit_fraction", type=float, default=1.0,
                      help="Edit budget as a fraction of the words.")
    edit.add_argument("--tokens_per_pass", type=int, default=5)
    edit.add_argument("--top_k_mlm", type=int, default=50)
    edit.add_argument("--min_mlm_prob", type=float, default=1e-4,
                      help="Drop masked-LM candidates below this probability.")
    edit.add_argument("--r_threshold", type=float, default=0.3,
                      help="(scrub) accept a candidate whose worst r is below this.")
    edit.add_argument("--fluency_margin", type=float, default=1.0,
                      help="Among candidates this close to the best objective, take the "
                           "masked LM's top rank.")
    return ap.parse_args()


def build_attack(args, wm_tokenizer, view, device):
    if args.attack == "rephrasing":
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device).eval()
        return RephrasingAttack(
            model, tokenizer, view, goal=args.mode, wm_tokenizer=wm_tokenizer,
            strength=args.strength, top_k=args.top_k, temperature=args.temperature,
            sampling=args.sampling, adaptive=args.adaptive, target_pvalue=args.threshold,
            overshoot=args.overshoot)
    mlm = MLMProposer.load(args.mlm_model_id, device, top_k=args.top_k_mlm, min_prob=args.min_mlm_prob)
    return WordEditAttack(
        view, mlm, wm_tokenizer, random.Random(args.seed), goal=args.mode,
        edit_fraction=args.edit_fraction, tokens_per_pass=args.tokens_per_pass,
        adaptive=args.adaptive, target_pvalue=args.threshold,
        overshoot=args.overshoot, r_threshold=args.r_threshold, fluency_margin=args.fluency_margin)


def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm_config = WatermarkConfig(**json.loads(args.wm_config))
    if args.access != "nobox" and not wm_config.dual_key:
        raise SystemExit("informed attacks need a dual-key scheme (textseal, or key_routing)")
    wm_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    view = oracle.build(args.access, wm_tokenizer, wm_config, args.threshold)
    attack = build_attack(args, wm_tokenizer, view, device)
    detector = build_detector(wm_tokenizer, wm_config)
    quality = QualityScorer(device=device)

    def channels(text):
        """Public / private / fused p-values, or the single p-value of a one-key scheme."""
        if wm_config.dual_key:
            return {name: ch["p_value"] for name, ch in text_channels(detector, text).items()}
        sm = wm_config.scoring_method
        scores = detector.get_scores_by_t([text], scoring_method=sm,
                                          seen_windows=set() if sm in ("v1", "v2") else None)[0]
        return {"p_value": detector.get_pvalue(sum(scores), len(scores), 1e-200) if scores else 1.0}

    torch.manual_seed(args.seed)
    rows = shard(load_jsonl(args.input), args.start_index, args.max_samples)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.jsonl")
    print(f"{args.attack} / {args.mode} / {args.access}: {len(rows)} texts -> {out_path}")
    batch_size = args.batch_size if args.attack == "rephrasing" else 1
    with open(out_path, "w") as fh:
        for start in range(0, len(rows), batch_size):
            srcs = [r[SOURCE_KEY[args.mode]] for r in rows[start:start + batch_size]]
            t0 = time.time()
            results = (attack.run_batch(srcs, max_new_tokens=args.max_new_tokens)
                       if args.attack == "rephrasing" else [attack.run(srcs[0])])
            elapsed = (time.time() - t0) / len(srcs)
            for i, (src, res) in enumerate(zip(srcs, results), start=start):
                out = res.pop("text")
                row = {"idx": args.start_index + i, "attack": args.attack, "mode": args.mode,
                       "access": args.access, "source_text": src, "attacked_text": out,
                       "source": channels(src), "attacked": channels(out),
                       "quality": quality.score(src, out), "time": elapsed, **res}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                ps = "  ".join(f"{k} p {v:.2e}" for k, v in row["attacked"].items())
                print(f"  [{i + 1}/{len(rows)}] {ps}  bertscore {row['quality']['bertscore_f1']:.3f}")


if __name__ == "__main__":
    main()
