# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Channel-difference tampering test for dual-key schemes (textseal, or key_routing).

Each token is generated under exactly one of the two keys, key A (public) with
probability alpha. Score every token under both keys, s_{A,t} and s_{B,t}, on the
detector's own positions (get_scores_by_t(..., per_key=True)), and contrast them:

    T = sum_t (s_{A,t} - s_{B,t})

An attacker steering on the public detector moves key A's scores and leaves key B's
alone. p_scrub = Pr(T <= T_obs) flags a deflated public channel (removal),
p_spoof = Pr(T >= T_obs) an inflated one (forgery).

The null is the per-position key-label permutation: at alpha = 1/2 the two keys are
exchangeable at every position of an untampered text, so swapping them position-wise
gives an exact null. It is not exact at any other alpha, so other alphas raise.

    python -m textseal.watermarking.tamper --input results.jsonl --text_key wm_text \\
        --wm_config '{"watermark_type": "textseal", "ngram": 3}'
"""
import argparse
import json
import os

import numpy as np

from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import build_detector, per_key_scores


def tamper_pvalues(scores: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """T and its one-sided permutation p-values, with the standard +1 correction."""
    diff = scores[:, 0] - scores[:, 1]
    t_obs = float(diff.sum())
    swapped = rng.integers(0, 2, size=(n_perm, len(diff))) == 1
    t_null = np.where(swapped, -diff, diff).sum(axis=1)
    return {"T": t_obs, "n_tokens": len(diff),
            "p_scrub": float((1 + np.sum(t_null <= t_obs)) / (n_perm + 1)),
            "p_spoof": float((1 + np.sum(t_null >= t_obs)) / (n_perm + 1))}


class TamperTest:
    """Tampering test of the deployed dual-key detector."""

    def __init__(self, tokenizer, wm_config, n_perm=50000, seed=0):
        if not wm_config.dual_key:
            raise ValueError("the tampering test needs a dual-key scheme (textseal, or key_routing)")
        if wm_config.mixing_alpha != 0.5:
            raise ValueError("the permutation null is exact only at mixing_alpha = 1/2, "
                             f"got {wm_config.mixing_alpha}")
        self.detector = build_detector(tokenizer, wm_config)
        self.n_perm = n_perm
        self.rng = np.random.default_rng(seed)

    def __call__(self, text) -> dict:
        return tamper_pvalues(per_key_scores(self.detector, text), self.n_perm, self.rng)


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description="Channel-difference tampering test")
    ap.add_argument("--input", required=True, help="JSONL with one text per row.")
    ap.add_argument("--text_key", default="wm_text")
    ap.add_argument("--wm_config", required=True,
                    help="The watermark config as a JSON object.")
    ap.add_argument("--tokenizer_id", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Tokenizer of the watermarked model.")
    ap.add_argument("--output", default=None, help="Default: <input without extension>.tamper.jsonl")
    ap.add_argument("--n_perm", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    test = TamperTest(AutoTokenizer.from_pretrained(args.tokenizer_id),
                      WatermarkConfig(**json.loads(args.wm_config)), args.n_perm, args.seed)
    output = args.output or os.path.splitext(args.input)[0] + ".tamper.jsonl"
    with open(args.input) as fin, open(output, "w") as fout:
        for i, line in enumerate(fin):
            row = json.loads(line)
            fout.write(json.dumps({"idx": row.get("idx", row.get("line", i)), **test(row[args.text_key])}) + "\n")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
