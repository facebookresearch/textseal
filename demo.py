#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
TextSeal Minimal Demo — Generate dual-key watermarked text and detect it.

Usage:
    python demo.py [--model MODEL] [--prompt "your prompt"]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.detector import TextSealDetector, localized_detect
from textseal.watermarking.generator import TextSealGenerator


def main():
    parser = argparse.ArgumentParser(description="TextSeal minimal demo")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model name")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory, e.g. /path/to/cache",
    )
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.", help="Prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max generation tokens")
    parser.add_argument("--secret_key", type=int, default=42, help="Watermark secret key")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    # ── 1. Load model & tokenizer ─────────────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    # ── 2. Configure watermark ────────────────────────────────────────────
    wm_config = WatermarkConfig(secret_key=args.secret_key)
    print(f"Keys: A={wm_config.key_a}, B={wm_config.key_b}, alpha={wm_config.mixing_alpha}")

    # ── 3. Generate watermarked text (dual-key) ──────────────────────────
    generator = TextSealGenerator(model, tokenizer, wm_config)

    messages = [{"role": "user", "content": args.prompt}]
    # Enable thinking for reasoning models (Qwen3.5, etc.)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        # Older models/tokenizers don't support enable_thinking
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"\nGenerating (max {args.max_tokens} tokens)...")
    texts = generator.generate(
        [prompt_text], max_gen_len=args.max_tokens, temperature=args.temperature, top_p=args.top_p
    )
    watermarked_text = texts[0]

    print(f"\n{'=' * 60}")
    print("WATERMARKED TEXT:")
    print(f"{'=' * 60}")
    print(watermarked_text)

    # ── 4. Detect watermark (dual-key + entropy-aware) ────────────────────
    detector = TextSealDetector(tokenizer, wm_config, model=model)
    result = detector.detect(watermarked_text)

    print(f"\n{'=' * 60}")
    print("DETECTION RESULT:")
    print(f"{'=' * 60}")
    print(f"  p-value:   {result['p_value']:.2e}")
    print(f"  n_tokens:  {result['n_tokens']}")
    print(f"  detected:  {result['detected']}")
    print(f"  entropy:   {result['entropy_weighted']}")
    if result.get("p_value_weighted") is not None:
        print(f"  p_weighted:   {result['p_value_weighted']:.2e}")
        print(f"  p_unweighted: {result['p_value_unweighted']:.2e}")

    # ── 5. Localized detection ──────────────────────────────────────────
    loc = localized_detect(watermarked_text, tokenizer, wm_config, model=model)

    print(f"\n{'=' * 60}")
    print("LOCALIZED DETECTION:")
    print(f"{'=' * 60}")
    print(f"  global p:    {loc.global_pvalue:.2e}")
    print(f"  localized p: {loc.localized_pvalue:.2e}")
    print(f"  final p:     {loc.final_pvalue:.2e}")
    print(f"  region:      [{loc.region_start}, {loc.region_end})")
    print(f"  wm tokens:   {sum(loc.token_labels)} / {loc.n_tokens}")

    # ── 6. Control: detect on unwatermarked text ──────────────────────────
    print(f"\n{'=' * 60}")
    print("CONTROL — detecting on unwatermarked text:")
    print(f"{'=' * 60}")
    ctrl = detector.detect("The sky appears blue because of Rayleigh scattering of sunlight in the atmosphere.")
    print(f"  p-value:   {ctrl['p_value']:.2e}")
    print(f"  detected:  {ctrl['detected']}")


if __name__ == "__main__":
    main()
