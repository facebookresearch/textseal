# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Command-line interface for the post-hoc watermarking system.

This module provides a CLI for running post-hoc watermarking with configurable
parameters for model selection, watermarking settings, and processing options.

Run with direct arguments:
    python -m textseal.watermarking.main \
        --input_path assets/sample_document.txt \
        --model.model_name meta-llama/Llama-3.2-1B-Instruct \
        --model.use_flash_attention False \
        --watermark.watermark_type greenlist \
        --watermark.ngram 2 \
        --watermark.delta 4.0 \
        --watermark.gamma 0.5 \
        --watermark.scoring_method v1 \
        --processing.temperature 1.0 \
        --processing.top_p 0.95

Run with beam search:
    python -m textseal.watermarking.main \
        --input_path assets/sample_document.txt \
        --watermark.watermark_type greenlist \
        --processing.beam_width 5 \
        --processing.candidates_per_beam 5 \
        --processing.use_biased_for_scoring false

Run with code evaluation by using these arguments:
    --input_path path/to/HumanEval_processed.jsonl \
    --prompt.prefill_answer "Here is the rephrased code:\n" --prompt.preserve_style false --prompt.preserve_format false \
    --evaluation.enable_code_evaluation true

Run with config file:
    python -m textseal.watermarking.main --config path/to/config.yaml

Run with config file and overrides:
    python -m textseal.watermarking.main --config path/to/config.yaml \
        --watermark.delta 3.0 --processing.temperature 0.8
"""

import random
import os
import json
import time
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from omegaconf import OmegaConf

import torch

from textseal.utils.config import cfg_from_cli
from textseal.watermarking.watermarker import PostHocWatermarker
from textseal.watermarking.attack import AttackSimulator
from textseal.watermarking.config import (
    WatermarkConfig,
    ModelConfig,
    MODEL_NAMES,
    ProcessingConfig,
    PromptConfig,
    EvaluationConfig,
    AttackConfig,
)


@dataclass
class CLIArgs:
    """Main configuration for CLI that includes all sub-configs."""
    # Required arguments
    name: str = "posthoc"
    dump_dir: str = "output/"  # Directory to save watermarked results
    
    # Seed for reproducibility
    seed: int = 0
    
    # Processing method
    input_path: str = ""  # Path to input document (.txt or .jsonl)
    text_key: str = "text"  # Key to extract text from JSONL lines
    processing_method: str = "auto"  # Method for processing documents
    num_lines: int = -1  # Number of lines to process from input file (if jsonl)
    max_input_tokens: int = -1  # Maximum tokens to use from input text (-1 = no limit)
    
    # Verbose output settings
    verbose: int = -1  # Print detailed results every N lines (-1 to disable)
    
    # Sub-configurations
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)


def main():
    """Main function for command-line usage."""
    try:
        # Old way of parsing CLI args with OmegaConf
        cli_args = OmegaConf.from_cli()
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
        default_cfg = OmegaConf.structured(CLIArgs())
        cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
        cfg = OmegaConf.to_object(cfg)
    except Exception as e:
        # Parsing as arguments with argparse
        cli_args_dict = cfg_from_cli()
        default_cfg = OmegaConf.structured(CLIArgs())
        if "config" in cli_args_dict:
            file_cfg = OmegaConf.load(cli_args_dict["config"])
            del cli_args_dict["config"]
            cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args_dict)
        else:
            cfg = OmegaConf.merge(default_cfg, cli_args_dict)
        cfg: CLIArgs = OmegaConf.to_object(cfg)
    
    # Validate input path exists
    input_path = Path(cfg.input_path)
    if not input_path.exists():
        print(f"Error: Input file '{cfg.input_path}' does not exist.")
        return 1
    
    # Validate some argument values
    if not torch.cuda.is_available():
        assert not cfg.model.use_flash_attention, "Flash Attention requires a CUDA-capable GPU."
    if cfg.model.model_name not in MODEL_NAMES:
        print(f"Model '{cfg.model.model_name}' hasn't been tested with TextSeal. Proceed with caution.")

    # Seed for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create output directory
    os.makedirs(cfg.dump_dir, exist_ok=True)
    print(f"Output directory: {cfg.dump_dir}")
    
    def process_line(original_text, line_num, aux_data=None, print_chunks=True):
        """
        Process a single line/document with watermarking.
        
        Args:
            original_text: The text to process
            line_num: Line number for logging
            aux_data: Optional auxiliary data to pass to watermarker
            print_chunks: Whether to print per-chunk metrics (for adaptive chunking)
        
        Returns:
            Dictionary containing processing results
        """
        # Truncate input text to max_input_tokens if specified
        original_token_count = None
        if cfg.max_input_tokens > 0:
            tokens = watermarker.tokenizer.encode(original_text)
            original_token_count = len(tokens)
            if len(tokens) > cfg.max_input_tokens:
                truncated_tokens = tokens[:cfg.max_input_tokens]
                original_text = watermarker.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                print(f"Line {line_num}: Truncated from {original_token_count} to {cfg.max_input_tokens} tokens")
        
        if cfg.evaluation.enable_detection_only:
            watermark_eval = watermarker.evaluator.evaluate_watermark(
                original_text,
                watermarker.detector,
                cfg.watermark.watermark_type,
                cfg.watermark.scoring_method,
                entropy_threshold=cfg.evaluation.entropy_threshold,
                tokenizer=watermarker.tokenizer,
                wm_config=cfg.watermark,
                model=watermarker.model
            )
            print(json.dumps({
                "line": line_num,
                "wm_score": watermark_eval.get("score"),
                "pvalue": watermark_eval.get("p_value"),
            }, ensure_ascii=False))
            return {"wm_eval": watermark_eval}

        # Determine processing method based on token count
        text_tokens = len(watermarker.tokenizer.encode(original_text))
        if text_tokens > watermarker.processing_config.max_chunk_size:
            print(f"Line {line_num}: Using adaptive chunking (text has {text_tokens:,} tokens)...")
            line_results = watermarker.process_large_text(
                original_text,
                max_gen_len=cfg.processing.max_gen_len,
                temperature=cfg.processing.temperature,
                top_p=cfg.processing.top_p,
                aux_data=aux_data,
                verbose=print_chunks
            )
            # Print per-chunk metrics if requested
            if print_chunks:
                for cr in line_results.get('chunk_results', []):
                    cr_wm = cr.get('wm_eval', {}) or {}
                    cr_q = cr.get('quality', {}) or {}
                    
                    # Check if multi-test mode
                    if isinstance(cr_wm, dict) and "tests" in cr_wm:
                        primary = cr_wm.get("primary", {})
                        print(json.dumps({
                            "line": line_num,
                            "chunk_index": cr.get('idx'),
                            "wm_score": primary.get("score"),
                            "pvalue": primary.get("p_value"),
                            "orig_tokens": cr.get('orig_toks'),
                            "wm_tokens": cr.get('wm_toks'),
                            "semantic_similarity": cr_q.get('semantic_similarity'),
                            "num_tests": len(cr_wm.get("tests", [])),
                        }, ensure_ascii=False))
                    else:
                        print(json.dumps({
                            "line": line_num,
                            "chunk_index": cr.get('idx'),
                            "wm_score": cr_wm.get("score"),
                            "pvalue": cr_wm.get("p_value"),
                            "orig_tokens": cr.get('orig_toks'),
                            "wm_tokens": cr.get('wm_toks'),
                            "semantic_similarity": cr_q.get('semantic_similarity'),
                        }, ensure_ascii=False))
        else:
            print(f"Line {line_num}: Using full processing (text has {text_tokens:,} tokens)...")
            line_results = watermarker.process_text(
                original_text,
                max_gen_len=cfg.processing.max_gen_len,
                temperature=cfg.processing.temperature,
                top_p=cfg.processing.top_p,
                aux_data=aux_data
            )
        stats = line_results.get('stats', {})
        times = line_results.get('times', {})
        wm_eval = line_results.get('wm_eval', {})
        qual_eval = line_results.get('quality', {})
        
        # Check if multi-test mode
        if isinstance(wm_eval, dict) and "tests" in wm_eval:
            # Multi-test mode: print results for each test
            primary = wm_eval.get("primary", {})
            print(json.dumps({
                "line": line_num,
                "wm_score": primary.get("score"),
                "pvalue": primary.get("p_value"),
                "orig_tokens": stats.get("orig_toks"),
                "wm_tokens": stats.get("wm_toks"),
                "tps": times.get("tps"),
                "code_pass": qual_eval.get("code_eval", {}).get("wm_passed"),
                "semantic_similarity": qual_eval.get("semantic_similarity"),
                "num_tests": len(wm_eval.get("tests", [])),
            }, ensure_ascii=False))
            
            # Print individual test results
            for test in wm_eval.get("tests", []):
                print(json.dumps({
                    "line": line_num,
                    "test_name": test.get("test_name"),
                    "watermark_type": test.get("watermark_type"),
                    "entropy_threshold": test.get("entropy_threshold"),
                    "wm_score": test.get("score"),
                    "pvalue": test.get("p_value"),
                    "detected": test.get("det"),
                    "toks_scored": test.get("toks_scored"),
                }, ensure_ascii=False))
        else:
            # Single test mode (backward compatible)
            print(json.dumps({
                "line": line_num,
                "wm_score": wm_eval.get("score"),
                "pvalue": wm_eval.get("p_value"),
                "orig_tokens": stats.get("orig_toks"),
                "wm_tokens": stats.get("wm_toks"),
                "tps": times.get("tps"),
                "code_pass": qual_eval.get("code_eval", {}).get("wm_passed"),
                "semantic_similarity": qual_eval.get("semantic_similarity"),
            }, ensure_ascii=False))

        # Verbose printing
        if cfg.verbose > 0 and line_num % cfg.verbose == 0:
            print("-" * 40)
            print("Watermarked text:")
            print(line_results.get("wm_text", "N/A"))
            print("-" * 40)
            print("Original text:")
            print(line_results.get("orig_text", "N/A"))
            print("-" * 40)
            
            # Print test results
            if isinstance(wm_eval, dict) and "tests" in wm_eval:
                print("Test Results:")
                for test in wm_eval.get("tests", []):
                    print(f"  {test.get('test_name')}: p-value={test.get('p_value'):.2e}, detected={test.get('det')}")
            else:
                print(f"P-value: {wm_eval.get('p_value', 'N/A')}")
            
            print(f"Code eval: {qual_eval.get('code_eval', 'N/A')}")
            print("-" * 40)
        
        # Perform attack if enabled - run all configured attack strengths
        if cfg.evaluation.evaluate_attack and attack_simulator is not None:
            print(f"Line {line_num}: Performing attack simulations...")
            wm_text = line_results.get("wm_text", "")
            
            if wm_text:
                # Run attacks with all configured strengths
                attack_results = attack_simulator.attack_all_strengths(
                    wm_text,
                    max_gen_len=cfg.attack.attack_max_gen_len,
                    top_p=cfg.attack.attack_top_p,
                    verbose=True
                )
                
                # Evaluate watermark and quality for each attack strength
                attacks_evaluated = {}
                for strength, attack_result in attack_results.items():
                    attacked_text = attack_result.get("attacked_text", "")
                    attack_stats = attack_result.get("attack_stats", {})
                    
                    # Evaluate watermark detection on attacked text
                    attack_wm_eval = watermarker.evaluator.evaluate_watermark(
                        attacked_text,
                        watermarker.detector,
                        cfg.watermark.watermark_type,
                        cfg.watermark.scoring_method,
                        entropy_threshold=cfg.evaluation.entropy_threshold,
                        tokenizer=watermarker.tokenizer,
                        wm_config=cfg.watermark,
                        model=watermarker.model
                    )
                    
                    # Evaluate quality of attack
                    attack_quality = watermarker.evaluator.evaluate_quality(wm_text, attacked_text)
                    
                    # Store results for this strength
                    attacks_evaluated[strength] = {
                        "attacked_text": attacked_text,
                        "attack_stats": attack_stats,
                        "attack_wm_eval": attack_wm_eval,
                        "attack_quality": attack_quality,
                        "temperature": attack_result.get("temperature"),
                    }
                    
                    # Print summary for this strength
                    if isinstance(attack_wm_eval, dict) and "tests" in attack_wm_eval:
                        primary_attack = attack_wm_eval.get("primary", {})
                        p_val = primary_attack.get("p_value")
                        detected = primary_attack.get("det")
                    else:
                        p_val = attack_wm_eval.get("p_value")
                        detected = attack_wm_eval.get("det")
                    
                    print(json.dumps({
                        "line": line_num,
                        "attack_strength": strength,
                        "attack_pvalue": p_val,
                        "attack_detected": detected,
                        "attack_tokens": attack_stats.get("attacked_tokens"),
                        "attack_similarity": attack_quality.get("semantic_similarity"),
                    }, ensure_ascii=False))
                
                # Add all attack results to line_results
                line_results["attacks"] = attacks_evaluated

        return line_results
    
    # Initialize watermarker with sub-configs
    watermarker = PostHocWatermarker(
        watermark_config=cfg.watermark,
        model_config=cfg.model,
        processing_config=cfg.processing,
        evaluation_config=cfg.evaluation,
        prompt_config=cfg.prompt
    )
    
    # Initialize attack simulator if attack mode is enabled
    attack_simulator = None
    if cfg.attack.enable_attack or cfg.evaluation.evaluate_attack:
        attack_simulator = AttackSimulator(
            attack_config=cfg.attack,
            cache_dir=cfg.model.cache_dir
        )
        print(f"✓ Attack simulator initialized")

    # Detect file format
    file_extension = input_path.suffix.lower()
    
    print(f"Processing input file: {cfg.input_path}")
    if file_extension == ".jsonl":
        output_jsonl_path = os.path.join(cfg.dump_dir, f"results.jsonl")
        line_num = -1
        num_successful = 0
        num_failed = 0

        with open(cfg.input_path, 'r', encoding='utf-8') as in_f, \
             open(output_jsonl_path, 'w', encoding='utf-8') as out_f:

            for line in in_f:
                line_num += 1
                if cfg.num_lines > 0 and line_num >= cfg.num_lines:
                    break
                line = line.strip()

                try:
                    data = json.loads(line)
                    original_text = data[cfg.text_key]
                except Exception as e:
                    print(f"Error line {line_num}: {e}")
                    num_failed += 1
                    continue

                # try:
                line_results = process_line(original_text, line_num, aux_data=data, print_chunks=True)
                
                # Save to JSONL with full structure:
                # - Original input data fields preserved
                # - wm_text, orig_text: the texts
                # - wm_eval: watermark evaluation results
                #   * Single test: {score, p_value, det, ...}
                #   * Multi-test: {tests: [{test_name, watermark_type, entropy_threshold, score, p_value, det, ...}, ...], primary: {...}}
                # - quality: quality metrics
                # - stats: token counts, ratios
                # - times: timing information
                # - watermark_config: watermark parameters used (for later attack/detection)
                line_results = {
                    **data, 
                    "line": line_num, 
                    **line_results,
                    "watermark_config": asdict(cfg.watermark)
                }
                out_f.write(json.dumps(line_results, ensure_ascii=False) + '\n')
                out_f.flush()
                num_successful += 1

                # except Exception as e:
                #     print(f"Error processing line {line_num}: {e}")
                #     import traceback
                #     traceback.print_exc()
                #     num_failed += 1
        
    elif file_extension == ".txt":
        # Read the text file
        with open(cfg.input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Process the document
        results = process_line(text, line_num=1, aux_data=None, print_chunks=False)
        results["input_path"] = cfg.input_path
        results["watermark_config"] = asdict(cfg.watermark)

        # Save a single-line JSONL with stats and texts
        output_jsonl_path = os.path.join(cfg.dump_dir, f"results.jsonl")
        with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(results, ensure_ascii=False) + '\n')
    else:
        print(f"Error: Unsupported file format '{file_extension}'. Only .txt and .jsonl are supported.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
