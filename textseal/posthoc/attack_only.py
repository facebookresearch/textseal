"""
Standalone attack script for pre-watermarked text.

This script performs rephrasing attacks on pre-watermarked text (from previous experiments)
and evaluates watermark detection on both the original watermarked text and the attacked version.

It allows you to skip the initial watermarking phase when you already have watermarked texts.

IMPORTANT: The script automatically loads watermark configuration from the input JSONL file
if it contains a 'watermark_config' field (automatically added by main.py). You only need to
manually specify watermark parameters if the field is missing or you want to override them.

Usage:
    # Minimal usage - watermark config auto-loaded from input file
    python -m textseal.posthoc.attack_only \
        --input_path output/results.jsonl \
        --wm_text_key wm_text \
        --model.model_name meta-llama/Llama-3.2-1B-Instruct \
        --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
        --output_path output/attack_results.jsonl

    # Manual watermark config (if not in file or to override)
    python -m textseal.posthoc.attack_only \
        --input_path output/results.jsonl \
        --wm_text_key wm_text \
        --watermark.watermark_type greenlist \
        --watermark.ngram 2 \
        --watermark.delta 4.0 \
        --watermark.gamma 0.5 \
        --watermark.scoring_method v1 \
        --model.model_name meta-llama/Llama-3.2-1B-Instruct \
        --attack.attack_model_name meta-llama/Llama-3.2-3B-Instruct \
        --output_path output/attack_results.jsonl

    # With custom attack parameters
    python -m textseal.posthoc.attack_only \
        --input_path output/results.jsonl \
        --wm_text_key wm_text \
        --model.model_name meta-llama/Llama-3.2-1B-Instruct \
        --attack.attack_temperature 0.7 \
        --attack.attack_top_p 0.9 \
        --entropy_threshold 2.0
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
import torch
import numpy as np
import random

from textseal.common.utils.config import cfg_from_cli
from textseal.common.watermark.core import WatermarkConfig
from textseal.posthoc.config import ModelConfig, AttackConfig, EvaluationConfig
from textseal.posthoc.attack import AttackSimulator
from textseal.posthoc.detector import build_detector
from textseal.posthoc.evaluation import WatermarkEvaluator


@dataclass
class AttackOnlyArgs:
    """Configuration for attack-only mode."""
    # Input/output
    input_path: str = ""  # Path to JSONL file with watermarked text
    wm_text_key: str = "wm_text"  # Key containing the watermarked text
    output_path: str = "output/attack_results.jsonl"  # Where to save results
    
    # Entropy filtering for detection
    entropy_threshold: Optional[float] = None  # If set, only use tokens with entropy < threshold
    
    # Number of lines to process
    num_lines: int = -1  # -1 for all lines
    
    # Seed for reproducibility
    seed: int = 0
    
    # Sub-configurations
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def main():
    """Main function for attack-only mode."""
    try:
        # Old way of parsing CLI args with OmegaConf
        cli_args = OmegaConf.from_cli()
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
        default_cfg = OmegaConf.structured(AttackOnlyArgs())
        cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
        cfg = OmegaConf.to_object(cfg)
    except Exception as e:
        # Parsing as arguments with argparse
        cli_args_dict = cfg_from_cli()
        default_cfg = OmegaConf.structured(AttackOnlyArgs())
        if "config" in cli_args_dict:
            file_cfg = OmegaConf.load(cli_args_dict["config"])
            del cli_args_dict["config"]
            cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args_dict)
        else:
            cfg = OmegaConf.merge(default_cfg, cli_args_dict)
        cfg: AttackOnlyArgs = OmegaConf.to_object(cfg)
    
    # Validate input path exists
    input_path = Path(cfg.input_path)
    if not input_path.exists():
        print(f"Error: Input file '{cfg.input_path}' does not exist.")
        return 1
    
    # Seed for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create output directory
    output_dir = Path(cfg.output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output file: {cfg.output_path}")
    
    # Initialize attack simulator
    print("Initializing attack simulator...")
    attack_simulator = AttackSimulator(
        attack_config=cfg.attack,
        cache_dir=cfg.model.cache_dir
    )
    
    # Load model and tokenizer for watermark detection
    print(f"Loading model for watermark detection: {cfg.model.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=True,
        cache_dir=cfg.model.cache_dir
    )
    
    # Load model if entropy threshold is specified (needed for entropy computation)
    model = None
    if cfg.entropy_threshold is not None or \
       (cfg.evaluation.test_entropy_thresholds is not None and 
        len(cfg.evaluation.test_entropy_thresholds) > 0):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cfg.model.cache_dir
        }
        if cfg.model.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name,
            **model_kwargs
        )
        print("✓ Model loaded for entropy computation")
    
    # Build detector
    detector = build_detector(tokenizer, cfg.watermark, model)
    
    # Initialize evaluator
    evaluator = WatermarkEvaluator(cfg.evaluation)
    
    # Process each line in the input file
    print(f"Processing input file: {cfg.input_path}")
    line_num = -1
    num_successful = 0
    num_failed = 0
    
    with open(cfg.input_path, 'r', encoding='utf-8') as in_f, \
         open(cfg.output_path, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            line_num += 1
            if cfg.num_lines > 0 and line_num >= cfg.num_lines:
                break
            line = line.strip()
            
            try:
                data = json.loads(line)
                wm_text = data.get(cfg.wm_text_key)
                
                if not wm_text:
                    print(f"Warning: Line {line_num} missing '{cfg.wm_text_key}' key, skipping")
                    num_failed += 1
                    continue
                
                # Auto-load watermark config from the input file if available
                # This allows us to use the exact same watermark parameters that were used originally
                if "watermark_config" in data and line_num == 0:
                    print("\n" + "="*60)
                    print("Found watermark_config in input file!")
                    print("Using stored watermark parameters from original watermarking.")
                    stored_config = data["watermark_config"]
                    print(f"  watermark_type: {stored_config.get('watermark_type')}")
                    print(f"  delta: {stored_config.get('delta')}")
                    print(f"  gamma: {stored_config.get('gamma')}")
                    print(f"  ngram: {stored_config.get('ngram')}")
                    print(f"  scoring_method: {stored_config.get('scoring_method')}")
                    print(f"  secret_key: {stored_config.get('secret_key')}")
                    print("="*60 + "\n")
                    
                    # Update cfg.watermark with stored config (CLI args take precedence)
                    for key, value in stored_config.items():
                        if hasattr(cfg.watermark, key):
                            # Only override if not explicitly set via CLI
                            setattr(cfg.watermark, key, value)
                    
                    # Rebuild detector with correct config
                    detector = build_detector(tokenizer, cfg.watermark, model)
                
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                num_failed += 1
                continue
            
            try:
                print(f"\nLine {line_num}: Processing watermarked text...")
                
                # Evaluate watermark on original watermarked text
                print(f"  Evaluating watermark on original watermarked text...")
                wm_eval_before = evaluator.evaluate_watermark(
                    wm_text,
                    detector,
                    cfg.watermark.watermark_type,
                    cfg.watermark.scoring_method,
                    entropy_threshold=cfg.entropy_threshold,
                    tokenizer=tokenizer,
                    wm_config=cfg.watermark,
                    model=model
                )
                
                # Perform attack
                print(f"  Performing attack...")
                attack_result = attack_simulator.attack(
                    wm_text,
                    max_gen_len=cfg.attack.attack_max_gen_len,
                    temperature=cfg.attack.attack_temperature,
                    top_p=cfg.attack.attack_top_p
                )
                
                attacked_text = attack_result.get("attacked_text", "")
                attack_stats = attack_result.get("attack_stats", {})
                
                # Evaluate watermark on attacked text
                print(f"  Evaluating watermark on attacked text...")
                wm_eval_after = evaluator.evaluate_watermark(
                    attacked_text,
                    detector,
                    cfg.watermark.watermark_type,
                    cfg.watermark.scoring_method,
                    entropy_threshold=cfg.entropy_threshold,
                    tokenizer=tokenizer,
                    wm_config=cfg.watermark,
                    model=model
                )
                
                # Evaluate quality (similarity between watermarked and attacked)
                attack_quality = evaluator.evaluate_quality(wm_text, attacked_text)
                
                # Prepare output
                output_data = {
                    **data,  # Preserve original fields
                    "line": line_num,
                    "wm_text": wm_text,
                    "attacked_text": attacked_text,
                    "attack_stats": attack_stats,
                    "wm_eval_before_attack": wm_eval_before,
                    "wm_eval_after_attack": wm_eval_after,
                    "attack_quality": attack_quality,
                }
                
                # Print summary
                if isinstance(wm_eval_before, dict) and "tests" in wm_eval_before:
                    primary_before = wm_eval_before.get("primary", {})
                    primary_after = wm_eval_after.get("primary", {})
                    print(json.dumps({
                        "line": line_num,
                        "before_pvalue": primary_before.get("p_value"),
                        "after_pvalue": primary_after.get("p_value"),
                        "before_detected": primary_before.get("det"),
                        "after_detected": primary_after.get("det"),
                        "attack_similarity": attack_quality.get("semantic_similarity"),
                    }, ensure_ascii=False))
                else:
                    print(json.dumps({
                        "line": line_num,
                        "before_pvalue": wm_eval_before.get("p_value"),
                        "after_pvalue": wm_eval_after.get("p_value"),
                        "before_detected": wm_eval_before.get("det"),
                        "after_detected": wm_eval_after.get("det"),
                        "attack_similarity": attack_quality.get("semantic_similarity"),
                    }, ensure_ascii=False))
                
                # Write to output
                out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                out_f.flush()
                num_successful += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                import traceback
                traceback.print_exc()
                num_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {num_successful}")
    print(f"  Failed: {num_failed}")
    print(f"  Output: {cfg.output_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
