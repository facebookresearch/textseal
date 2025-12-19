# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Watermark evaluation script for evaluating watermarked LLM checkpoints.

salloc --partition=learn --nodes=1 --gpus-per-node=4 --cpus-per-gpu=24 --mem-per-cpu=8G --qos=high --time=6:00:00 --job-name=dev_llmwm
cd /path/to/textseal
conda activate lingua

Run with:
    python -m apps.wmtraining.eval_wm --ckpt /path/to/checkpoint --mode generate --prompts "Hello world" "Test prompt"
    python -m apps.wmtraining.eval_wm --ckpt /path/to/checkpoint --mode forward --input_text "Sample input text"
    python -m apps.wmtraining.eval_wm --ckpt /path/to/checkpoint --mode generate --prompts_file prompts.txt --num_sources 5
    python -m apps.wmtraining.eval_wm --ckpt /path/to/data --mode forward --prompts_file /path/to/data --num_sources 4 --sft_jsonl True --use_filter True --filter_path /path/to/textseal --filter_keys en_dickens_charles,en_doyle_arthur_conan,en_kipling_rudyard,en_shakespeare_william --watermark.ngram 2 --watermark.gamma 0.5 --watermark.secret_key 0 --watermark.attenuation_weight 0.0
 
    srun --partition=learn --nodes=1 --gpus-per-node=1 --qos=high --time=6:00:00 --job-name=eval_wm --cpus-per-gpu=24--mem-per-cpu=8G \
    python -m apps.wmtraining.eval_wm --text_key wm_text\
        --mode forward --prompts_file /path/to/data \
        --num_sources 1 --watermark.ngram 2 --watermark.gamma 0.5 --watermark.secret_key 0 --watermark.scoring_method v1 --ckpt /path/to/data \
        --ckpt /path/to/data \
        # --mode generate --prompts_file /path/to/data \

    python -m apps.wmtraining.eval_wm --text_key wm_text\
        --mode forward --prompts_file /path/to/textseal \
        --num_sources 1 --watermark.ngram 2 --watermark.gamma 0.5 --watermark.secret_key 0 --watermark.scoring_method v1 --ckpt /path/to/data \
        --watermark.secret_key 42

    python -m apps.wmtraining.eval_wm --text_key wm_text\
        --mode forward --prompts_file /path/to/data \
        --num_sources 1 --watermark.ngram 2 --watermark.gamma 0.5 --watermark.secret_key 8192 --watermark.scoring_method v1 --ckpt /path/to/data --num_samples 100\


--mode forward --prompts_file /path/to/data



"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from apps.common.utils.config import dataclass_from_dict, cfg_from_cli
from apps.wmtraining.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.wmtraining.transformer import LMTransformerWM, LMTransformerArgs
from apps.common.watermark.core import WatermarkArgs
from apps.common.watermark.core import score_batch
from IPython.display import display, HTML  # Add for HTML display

logger = logging.getLogger(__name__)

MAX_PRINTED_CHARS = 250


@dataclass
class EvalWMArgs:
    # SFT-style input
    sft_jsonl: bool = False  # If true, treat prompts_file as SFT jsonl with question/answer fields
    """Arguments for watermark evaluation."""
    
    # Model and checkpoint
    ckpt: str = ""  # Path to consolidated checkpoint
    
    # Evaluation mode
    mode: str = "generate"  # "generate" or "forward"
    
    # Input specification
    prompts: list[str] = field(default_factory=list) # Single prompt or list of prompts for generation or input texts for forward
    prompts_file: str | None = None  # File containing prompts/input texts (one per line in JSONL format)
    text_key: str = "text"  # Key in JSONL for the text field
    max_tokens: int = 512  # Maximum number of tokens per prompt/input text (used as max context for both generation and forward mode)
    num_samples: int = 1000  # Maximum number of samples to load from file
    
    # Generation parameters (used when mode="generate")
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    
    # Watermark evaluation parameters
    watermark: WatermarkArgs = field(default_factory=WatermarkArgs)
    num_sources: int = 1  # Number of different source watermarks to evaluate

    # Cross-tokenizer detection parameters
    wm_tokenizer_path: str = ""  # Path to watermark tokenizer (T1) if different from suspect model tokenizer (T2)

    # Filtering parameters
    use_filter: bool = False  # Whether to use a filter during watermark scoring
    filter_path: str = ""  # Path to the filter file
    filter_keys: str = ""  # Comma-separated keys for the filter
    
    # Deduplication parameters
    deduplicate_per_line: bool = False  # If True, reset seen_windows for each line/sample; if False, deduplicate across all lines
    
    # Output
    dump_dir: str = "outputs"  # Directory to save results
    verbose: bool = True
    seed: int = 42  # Random seed for reproducibility


def load_prompts(args: EvalWMArgs) -> list[str] | list[tuple[str, str]]:
    """Load prompts based on args. Returns either list of strings or list of (question, answer) tuples."""
    if args.prompts_file:
        if args.sft_jsonl:
            return load_sft_pairs_from_file(args.prompts_file, args.num_samples)
        else:
            return load_prompts_from_file(args.prompts_file, args.text_key, args.num_samples)
    elif args.prompts:
        prompts = [args.prompts] if isinstance(args.prompts, str) else args.prompts
        return prompts
    else:
        # Interactive input
        return _interactive_prompt_input(args.num_samples)

def _interactive_prompt_input(max_samples: int) -> list[str]:
    """Interactively collect prompts from user."""
    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)
        if len(prompts) >= max_samples:
            break
    return prompts

def truncate_prompts(prompts: list[str], tokenizer, max_tokens: int) -> list[str]:
    """Truncate prompts to max_tokens."""
    truncated = []
    for p in prompts:
        tokens = tokenizer.encode(p, add_bos=True, add_eos=False)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated.append(tokenizer.decode(truncated_tokens))
        else:
            truncated.append(p)
    return truncated

def load_prompts_from_file(file_path: str, text_key: str = "text", num_samples: int = None) -> list[str]:
    """Load prompts/input texts from a JSONL file, one per line, up to num_samples."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                prompts.append(json.loads(line)[text_key])
                if num_samples is not None and len(prompts) >= num_samples:
                    break
    print(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts

def load_sft_pairs_from_file(file_path: str, num_samples: int = None):
    """Load (question, answer) pairs from SFT JSONL file."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                q = obj.get("question", None)
                a = obj.get("answer", None)
                if q is not None and a is not None:
                    pairs.append((q, a))
                if num_samples is not None and len(pairs) >= num_samples:
                    break
    print(f"Loaded {len(pairs)} SFT Q/A pairs from {file_path}")
    return pairs


def score_to_ansi_color(score):
    # score: 0 (red) to 1 (green)
    r = int(255 * (1 - score))
    g = int(255 * score)
    b = 0
    # 38;2;r;g;b for foreground color
    return f"\033[38;2;{r};{g};{b}m"


def visualize_tokens_with_mask_terminal(tokens, tokenizer, mask):
    token_strs = [tokenizer.decode([t]) for t in tokens]
    colored_strs = []
    for ii, token_str in enumerate(token_strs):
        color = score_to_ansi_color(mask[ii])
        # Remove newlines for terminal display, or keep as is
        token_str = token_str.replace('\n', '\\n')
        colored_strs.append(f"{color}{token_str}\033[0m")
    print(''.join(colored_strs))

def _build_t1_vocab_cache(wm_tokenizer, verbose: bool = False):
    """Build and cache T1 vocabulary mapping: string -> token_id."""
    if not hasattr(_build_t1_vocab_cache, 'cache'):
        _build_t1_vocab_cache.cache = {}
    
    tokenizer_key = id(wm_tokenizer)
    if tokenizer_key not in _build_t1_vocab_cache.cache:
        if verbose:
            print("Building T1 vocabulary mapping...")
        t1_string_to_id = {}
        
        try:
            vocab_size = getattr(wm_tokenizer, 'vocab_size', 
                               getattr(wm_tokenizer, 'n_words', 
                                     getattr(wm_tokenizer, 'n_vocab', 100000)))
            
            for token_id in range(vocab_size):
                try:
                    token_str = wm_tokenizer.decode([token_id])
                    t1_string_to_id[token_str] = token_id
                except:
                    continue
            
            _build_t1_vocab_cache.cache[tokenizer_key] = t1_string_to_id
            if verbose:
                print(f"Built vocab mapping with {len(t1_string_to_id)} tokens")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not build vocab mapping: {e}")
            _build_t1_vocab_cache.cache[tokenizer_key] = None
    
    return _build_t1_vocab_cache.cache[tokenizer_key]

def _cross_tokenizer_detection(
    score_input_ids: torch.Tensor,
    score_pred_tokens: torch.Tensor,
    tokenizer,
    wm_tokenizer,
    wm_args: WatermarkArgs,
    verbose: bool = False
):
    """
    Perform cross-tokenizer watermark detection.
    
    Returns:
        wm_windows: Watermark context windows (T1 tokenization)
        targets: Target tokens to score (T1 token IDs)
        cross_tokenizer_info: Dictionary with alignment statistics
    """
    # Get T1 vocabulary mapping
    t1_string_to_id = _build_t1_vocab_cache(wm_tokenizer, verbose)
    
    # Use tokens from forward pass (T2)
    t2_tokens = score_input_ids.cpu().tolist()
    t2_pred_tokens = score_pred_tokens.cpu().tolist()
    
    # Reconstruct text from T2 tokens
    t2_strings = [tokenizer.decode([t]) for t in t2_tokens]
    text_reconstructed = ''.join(t2_strings)
    
    # Tokenize with T1 (watermark tokenizer)
    t1_tokens = wm_tokenizer.encode(text_reconstructed, add_bos=False, add_eos=False)
    t1_strings = [wm_tokenizer.decode([t]) for t in t1_tokens]
    
    # Build cumulative prefix positions for T1
    t1_prefixes = []
    pos = 0
    for s in t1_strings:
        pos += len(s)
        t1_prefixes.append(pos)
    
    # Align T2 tokens to T1 tokens based on prefix matching
    alignment_mask = []
    t2_to_t1_map = []
    t2_pos = 0
    for t2_str in t2_strings:
        t2_pos += len(t2_str)
        # Find matching T1 index
        matched_idx = -1
        for t1_idx, t1_prefix in enumerate(t1_prefixes):
            if t1_prefix == t2_pos:
                matched_idx = t1_idx
                break
        t2_to_t1_map.append(matched_idx)
        alignment_mask.append(matched_idx >= 0)
    
    # Build T1 watermark windows
    t1_tensor = torch.tensor(t1_tokens, dtype=torch.long)
    wm_windows_t1 = torch.stack(
        [t1_tensor.roll(j, dims=0) for j in range(wm_args.ngram)],
        dim=-1,
    ).flip(-1)
    
    # Filter to aligned positions
    valid_indices = [i for i, aligned in enumerate(alignment_mask) if aligned]
    
    if len(valid_indices) == 0:
        return None, None, {
            "t1_tokens": t1_tokens,
            "t2_tokens": t2_tokens,
            "alignment_mask": alignment_mask,
            "num_aligned": 0,
            "num_scored": 0,
        }
    
    # Get T1 windows for aligned positions
    valid_t1_indices = [t2_to_t1_map[i] for i in valid_indices]
    wm_windows = wm_windows_t1[valid_t1_indices].unsqueeze(0)
    
    # Get T2 predicted tokens at aligned positions
    valid_t2_preds = [t2_pred_tokens[i] for i in valid_indices]
    
    # Check if T2 predictions exist as single tokens in T1 vocab
    final_valid_mask = []
    final_targets = []
    
    if t1_string_to_id is not None:
        # Use pre-built mapping (fast)
        for t2_pred in valid_t2_preds:
            try:
                t2_str = tokenizer.decode([t2_pred])
                if t2_str in t1_string_to_id:
                    final_valid_mask.append(True)
                    final_targets.append(t1_string_to_id[t2_str])
                else:
                    final_valid_mask.append(False)
            except:
                final_valid_mask.append(False)
    else:
        # Fallback to encoding (slower but works if vocab mapping failed)
        for t2_pred in valid_t2_preds:
            try:
                t2_str = tokenizer.decode([t2_pred])
                t1_encoded = wm_tokenizer.encode(t2_str, add_bos=False, add_eos=False)
                if len(t1_encoded) == 1:
                    final_valid_mask.append(True)
                    final_targets.append(t1_encoded[0])
                else:
                    final_valid_mask.append(False)
            except:
                final_valid_mask.append(False)
    
    if sum(final_valid_mask) == 0:
        return None, None, {
            "t1_tokens": t1_tokens,
            "t2_tokens": t2_tokens,
            "alignment_mask": alignment_mask,
            "num_aligned": sum(alignment_mask),
            "num_scored": 0,
        }
    
    # Filter to final valid positions
    final_valid_indices = [i for i, m in enumerate(final_valid_mask) if m]
    wm_windows = wm_windows[:, final_valid_indices, :]
    targets = torch.tensor(final_targets, dtype=torch.long).unsqueeze(0)
    
    cross_tokenizer_info = {
        "t1_tokens": t1_tokens,
        "t2_tokens": t2_tokens,
        "alignment_mask": alignment_mask,
        "num_aligned": sum(alignment_mask),
        "num_scored": len(final_targets),
    }
    
    return wm_windows, targets, cross_tokenizer_info

def evaluate_forward_watermarks(
    model: torch.nn.Module,
    tokenizer,
    input_data: str | tuple[str, str],
    wm_args: WatermarkArgs,
    num_sources: int,
    verbose: bool = True,
    output_file = None,
    use_filter: bool = False,
    filter_path: str = "",
    filter_keys: str = "",
    sft: bool = False,
    wm_tokenizer = None,
    max_tokens: int = 512,
    deduplicate_per_line: bool = True,
) -> dict:
    """
    Perform forward pass and evaluate watermark scores for next token predictions.
    
    Args:
        model: The language model
        tokenizer: The suspect model's tokenizer (T2)
        input_data: Input text (str) or (question, answer) tuple if sft=True
        wm_args: Watermark arguments
        num_sources: Number of different source watermarks to evaluate
        verbose: Whether to print verbose output
        output_file: File object to write results to (if any)
        use_filter: Whether to use a filter during scoring
        filter_path: Path to the filter file
        filter_keys: Keys to use from the filter
        sft: If True, input_data is a (question, answer) tuple and only answer tokens are scored
        wm_tokenizer: Watermark tokenizer (T1) if different from suspect tokenizer. If provided,
                      uses cross-tokenizer detection where T1 determines green/red split and
                      only T2 tokens with shared prefixes are scored.
        max_tokens: Maximum number of tokens to process in forward pass
        deduplicate_per_line: If True, reset seen_windows for each sample (dedup per-line only);
                              if False, maintain seen_windows across all samples (global dedup)
        
    Returns:
        Dictionary containing forward pass results and watermark scores
    """
    # Handle SFT mode (question-answer pairs)
    if sft:
        q, a = input_data
        if verbose:
            truncated_q = q[:MAX_PRINTED_CHARS].replace('\n', '\\n')
            truncated_a = a[:MAX_PRINTED_CHARS].replace('\n', '\\n')
            print(f"\nQ: {truncated_q}")
            print(f"A: {truncated_a}")
        
        # Encode question and answer
        q_ids = tokenizer.encode(q, add_bos=False, add_eos=False)
        a_ids = tokenizer.encode(a, add_bos=False, add_eos=False)
        input_ids = q_ids + a_ids
        
        # Truncate if exceeds max_tokens
        if len(input_ids) > max_tokens:
            input_ids = input_ids[:max_tokens]
            # Adjust answer start if needed
            if len(q_ids) >= max_tokens:
                # Question itself is too long, skip this sample
                if verbose:
                    print(f"Warning: Question length ({len(q_ids)}) exceeds max_tokens ({max_tokens}), truncating")
                q_ids = q_ids[:max_tokens]
                a_ids = []
            else:
                # Truncate answer
                a_ids = input_ids[len(q_ids):]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_tensor)
            pred_tokens = torch.argmax(logits, dim=-1)
        
        # Only score answer tokens
        answer_start = len(q_ids)
        score_input_ids = input_tensor[0, answer_start:]
        score_pred_tokens = pred_tokens[0, answer_start:]
        score_logits = logits[0, answer_start:, :]
    else:
        # Regular forward mode
        input_text = input_data
        if verbose:
            truncated_input = input_text[:MAX_PRINTED_CHARS].replace('\n', '\\n')
            print(f"\nInput: {truncated_input}")
        
        # Tokenize input
        input_ids = tokenizer.encode(input_text, add_bos=False, add_eos=False)
        
        # Truncate if exceeds max_tokens
        if len(input_ids) > max_tokens:
            if verbose:
                print(f"Warning: Input length ({len(input_ids)}) exceeds max_tokens ({max_tokens}), truncating")
            input_ids = input_ids[:max_tokens]
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_tensor)
            pred_tokens = torch.argmax(logits, dim=-1)
        
        # Score all tokens
        score_input_ids = input_tensor.squeeze(0)
        score_pred_tokens = pred_tokens.squeeze(0)
        score_logits = logits.squeeze(0)
    
    # Watermark scoring
    multi_source_scores = {}
    multi_source_totals = {}
    cross_tokenizer_info = {}
    
    # Initialize seen_windows tracking for v1 scoring method (per source)
    if deduplicate_per_line:
        # Reset for each sample to enable per-line deduplication only
        seen_windows_per_source = {}
    else:
        # Use global seen_windows across all samples
        if not hasattr(evaluate_forward_watermarks, '_seen_windows_per_source'):
            evaluate_forward_watermarks._seen_windows_per_source = {}
        seen_windows_per_source = evaluate_forward_watermarks._seen_windows_per_source
    
    if num_sources >= 1:
        # Cross-tokenizer detection mode
        if wm_tokenizer is not None:
            wm_windows, targets, cross_tokenizer_info = _cross_tokenizer_detection(
                score_input_ids, score_pred_tokens, tokenizer, wm_tokenizer, wm_args, verbose
            )
            
            # Handle case where no tokens can be scored
            if wm_windows is None:
                for source_id in range(num_sources):
                    multi_source_scores[f"source_{source_id}"] = 0.0
                    multi_source_totals[f"source_{source_id}"] = 0
                wm_windows = None
        else:
            # Standard single-tokenizer mode
            wm_windows = torch.stack(
                [score_input_ids.roll(j, dims=0) for j in range(wm_args.ngram)],
                dim=-1,
            ).flip(-1).unsqueeze(0)
            targets = score_pred_tokens.unsqueeze(0)
        
        # Score across sources if we have valid windows
        if wm_windows is not None:
            for source_id in range(num_sources):
                source_tensor = torch.zeros_like(targets) + source_id
                
                # Get or create seen_windows set for this source if using v1 or v2 scoring
                seen_windows = None
                if wm_args.scoring_method in ["v1", "v2"]:
                    if source_id not in seen_windows_per_source:
                        seen_windows_per_source[source_id] = set()
                    seen_windows = seen_windows_per_source[source_id]
                
                wm_mask, source_score = score_batch(
                    wm_windows, targets, wm_args, sources=source_tensor,
                    use_filter=use_filter, filter_path=filter_path, filter_keys=filter_keys,
                    seen_windows=seen_windows
                )
                multi_source_scores[f"source_{source_id}"] = source_score
                multi_source_totals[f"source_{source_id}"] = len(wm_mask)
    
    # Compute entropy
    probabilities = torch.softmax(score_logits, dim=-1)
    entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
    
    # Build result dictionary
    if sft:
        result = {
            "prompt": q,
            "answer": a,
            "input_tokens": input_ids,
            "answer_tokens": score_input_ids.cpu().tolist(),
            "predicted_tokens": score_pred_tokens.cpu().tolist(),
            "multi_source_scores": multi_source_scores,
            "multi_source_totals": multi_source_totals,
            "average_entropy": entropies.mean().item(),
        }
    else:
        result = {
            "prompt": input_data,
            "input_tokens": score_input_ids.cpu().tolist(),
            "predicted_tokens": score_pred_tokens.cpu().tolist(),
            "multi_source_scores": multi_source_scores,
            "multi_source_totals": multi_source_totals,
            "average_entropy": entropies.mean().item(),
        }
    
    # Add cross-tokenizer info if available
    if cross_tokenizer_info:
        result["cross_tokenizer_info"] = cross_tokenizer_info

    if verbose:
        print(f"Scores: {multi_source_scores}")
        print(f"Average entropy: {entropies.mean().item():.4f}")
    
    if output_file:
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    return result

def evaluate_generation_watermarks(
    generator: PackedCausalTransformerGenerator,
    prompts: list[str],
    wm_args: WatermarkArgs,
    num_sources: int,
    verbose: bool = True,
    output_file = None,
    use_filter: bool = False,
    filter_path: str = "",
    filter_keys: str = "",
    deduplicate_per_line: bool = True,
) -> list[dict]:
    """
    Generate text from prompts and evaluate watermark scores.
    
    Args:
        generator: The text generator
        prompts: List of prompts to generate from
        wm_args: Watermark arguments
        num_sources: Number of different source watermarks to evaluate
        verbose: Whether to print verbose output
        output_file: File object to write results to (if any)
        use_filter: Whether to use a filter during scoring
        filter_path: Path to the filter file
        filter_keys: Keys to use from the filter
        deduplicate_per_line: If True, reset seen_windows for each sample (dedup per-line only);
                              if False, maintain seen_windows across all samples (global dedup)

    Returns:
        List of dictionaries containing results for each prompt
    """
    results = []
    
    if verbose:
        print(f"Generating from {len(prompts)} prompts...")
    
    start_time = time.time()
    generations, loglikelihoods, greedy, aux = generator.generate(prompts, return_aux=True)
    end_time = time.time()
    
    total_tokens = sum(len(generator.tokenizer.encode(gen, False, False)) for gen in generations)
    tokens_per_second = total_tokens / (end_time - start_time)
    
    if verbose:
        print(f"Generated {total_tokens} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    
    # Initialize global seen_windows if needed
    if not deduplicate_per_line:
        if not hasattr(evaluate_generation_watermarks, '_seen_windows_per_source'):
            evaluate_generation_watermarks._seen_windows_per_source = {}
    
    for i, (prompt, generation) in enumerate(zip(prompts, generations)):
        truncated_prompt = prompt[:MAX_PRINTED_CHARS].replace('\n', '\\n')
        truncated_generation = generation[:MAX_PRINTED_CHARS].replace('\n', '\\n')
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {truncated_prompt}")
        print(f"Generated: {truncated_generation}")
        
        tokens = generator.tokenizer.encode(generation, add_bos=False, add_eos=False)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        wm_windows = torch.stack(
            [tokens_tensor.roll(j + 1, dims=-1) for j in range(wm_args.ngram)],
            dim=-1,
        ).flip(-1).unsqueeze(0)
        targets = tokens_tensor.unsqueeze(0)
        
        multi_source_scores = {}
        multi_source_masks = {}
        multi_source_totals = {}
        
        # Initialize seen_windows tracking
        if deduplicate_per_line:
            # Reset for each generation (per-sample deduplication)
            seen_windows_per_source = {}
        else:
            # Use global seen_windows across all samples
            seen_windows_per_source = evaluate_generation_watermarks._seen_windows_per_source

        for source_id in range(num_sources):
            source_tensor = torch.zeros_like(targets) + source_id
            
            # Get or create seen_windows set for this source if using v1 or v2 scoring
            seen_windows = None
            if wm_args.scoring_method in ["v1", "v2"]:
                if source_id not in seen_windows_per_source:
                    seen_windows_per_source[source_id] = set()
                seen_windows = seen_windows_per_source[source_id]
            
            source_mask, source_score = score_batch(wm_windows, targets, wm_args, sources=source_tensor, use_filter=use_filter, filter_path=filter_path, filter_keys=filter_keys, seen_windows=seen_windows)
            multi_source_scores[f"source_{source_id}"] = source_score
            multi_source_masks[f"source_{source_id}"] = source_mask
            multi_source_totals[f"source_{source_id}"] = len(source_mask)

            # Print mask summary
            mask_sum = sum(source_mask)
            mask_len = len(source_mask)
            print(f"Source {source_id} watermark score: {source_score:.4f} ({mask_sum:.1f}/{mask_len} tokens watermarked)")
            
            # Visualize tokens with mask in terminal
            if not use_filter:
                visualize_tokens_with_mask_terminal(tokens, generator.tokenizer, source_mask)

        result = {
            "prompt": prompt,
            "generation": generation,
            "tokens": tokens,
            "num_tokens": mask_len,
            "multi_source_scores": multi_source_scores,
            "multi_source_masks": multi_source_masks,
            "multi_source_totals": multi_source_totals,
        }

        if output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        results.append(result)
    
    return results

def _setup_watermark_args(saved_wm_args, config, args):
    """Setup watermark args from various sources.
    
    Priority: args.watermark (eval config) > checkpoint config.watermark > saved_wm_args
    This ensures eval config watermark parameters (matching benchmark watermarking)
    take precedence over checkpoint's saved training parameters.
    """
    # Use args.watermark directly - this comes from the eval config and should take priority
    return args.watermark

def _open_output_file(dump_dir: str):
    """Open output file if dump_dir is specified."""
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        output_file_path = os.path.join(dump_dir, "results.jsonl")
        print(f"Saving results to {output_file_path}")
        return open(output_file_path, 'w', encoding='utf-8')
    return None

def main():
    """Main evaluation function."""
    
    def dict_to_namespace(d):
        """Recursively convert dict to SimpleNamespace."""
        from types import SimpleNamespace
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    # Parse arguments - handle both stool.py (with --config) and direct CLI
    saved_wm_args = None
    try:
        # When using stool.py with --config argument
        cli_args = OmegaConf.from_cli()
        file_cfg = OmegaConf.load(cli_args.config)
        
        # Save stool.py-specific fields that aren't in EvalWMArgs dataclass
        dump_dir_from_config = file_cfg.get("dump_dir", None)
        
        # Remove stool.py-specific fields before merging with structured config
        if "name" in file_cfg:
            del file_cfg["name"]
        if "dump_dir" in file_cfg:
            del file_cfg["dump_dir"]
        
        # Remove config from cli_args since we already loaded it
        del cli_args.config
        
        # Merge without strict validation to allow extra fields like wm_tokenizer_path
        cfg = OmegaConf.merge(file_cfg, cli_args)
        
        # Apply defaults from EvalWMArgs for any missing fields
        default_cfg = OmegaConf.structured(EvalWMArgs())
        default_dict = OmegaConf.to_container(default_cfg)
        for key, value in default_dict.items():
            if key not in cfg:
                cfg[key] = value
        
        # Also apply watermark defaults for any missing watermark fields
        if 'watermark' in cfg:
            from apps.common.watermark.core import WatermarkConfig
            wm_defaults = OmegaConf.to_container(OmegaConf.structured(WatermarkConfig()))
            for key, value in wm_defaults.items():
                if key not in cfg['watermark']:
                    cfg['watermark'][key] = value
        
        # Convert to plain dict, then recursively to object with attributes
        cfg_dict = OmegaConf.to_container(cfg)
        cfg = dict_to_namespace(cfg_dict)
        
        # Restore dump_dir
        if dump_dir_from_config:
            cfg.dump_dir = dump_dir_from_config
            
    except Exception as e:
        # Parsing as arguments with argparse
        cli_args = cfg_from_cli()
        if isinstance(cli_args.get("prompts", []), str):
            cli_args["prompts"] = [cli_args["prompts"]]  # Ensure prompts is a list
        saved_wm_args = cli_args.get("watermark", None)
        
        if "config" in cli_args:
            file_cfg = OmegaConf.load(cli_args["config"])
            del cli_args["config"]
            
            # Merge without strict validation to allow extra fields
            cfg = OmegaConf.merge(file_cfg, cli_args)
            
            # Apply defaults for missing fields
            default_cfg = OmegaConf.structured(EvalWMArgs())
            default_dict = OmegaConf.to_container(default_cfg)
            for key, value in default_dict.items():
                if key not in cfg:
                    cfg[key] = value
            
            # Also apply watermark defaults for any missing watermark fields
            if 'watermark' in cfg:
                from apps.common.watermark.core import WatermarkConfig
                wm_defaults = OmegaConf.to_container(OmegaConf.structured(WatermarkConfig()))
                for key, value in wm_defaults.items():
                    if key not in cfg['watermark']:
                        cfg['watermark'][key] = value
            
            # Convert to plain dict, then recursively to object with attributes
            cfg_dict = OmegaConf.to_container(cfg)
            from types import SimpleNamespace
            cfg = dict_to_namespace(cfg_dict)
        else:
            default_cfg = OmegaConf.structured(EvalWMArgs())
            cfg = OmegaConf.merge(default_cfg, cli_args)
            cfg = OmegaConf.to_object(cfg)
    
    args = cfg

    if not args.ckpt:
        raise ValueError("Must specify checkpoint path with --ckpt")
    if args.mode not in ["generate", "forward"]:
        raise ValueError("Mode must be 'generate' or 'forward'")
    
    # Set random seed
    torch.manual_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading model from {args.ckpt}...")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)
    
    # Load watermark tokenizer if specified (for cross-tokenizer detection)
    wm_tokenizer = None
    if args.wm_tokenizer_path:
        print(f"Loading watermark tokenizer from {args.wm_tokenizer_path}...")
        _, wm_tokenizer, _ = load_consolidated_model_and_tokenizer(args.wm_tokenizer_path)
        print("Cross-tokenizer detection mode enabled")
    
    # Setup watermark args
    wm_args = _setup_watermark_args(saved_wm_args, config, args)
    print(f"Using watermark args: {wm_args}")
    print(f"Evaluating {args.num_sources} source(s)")
    
    # Open output file
    output_file = _open_output_file(args.dump_dir)

    # Load prompts/pairs
    data = load_prompts(args)
    if not data:
        raise ValueError("No prompts/data provided")
    
    if args.mode == "generate":
        if args.sft_jsonl:
            # Extract questions from (q, a) pairs
            prompts = [q for q, a in data]
        else:
            prompts = truncate_prompts(data, tokenizer, args.max_tokens)
        
        generator = PackedCausalTransformerGenerator(args.generator, model, tokenizer)
        results = evaluate_generation_watermarks(
            generator, prompts, wm_args, args.num_sources, args.verbose, output_file,
            use_filter=args.use_filter, filter_path=args.filter_path, filter_keys=args.filter_keys,
            deduplicate_per_line=args.deduplicate_per_line
        )
    elif args.mode == "forward":
        results = []
        if args.sft_jsonl:
            # data is list of (question, answer) tuples
            for qa_pair in data:
                result = evaluate_forward_watermarks(
                    model, tokenizer, qa_pair, wm_args, args.num_sources, args.verbose, output_file,
                    use_filter=args.use_filter, filter_path=args.filter_path, filter_keys=args.filter_keys,
                    sft=True, wm_tokenizer=wm_tokenizer, max_tokens=args.max_tokens,
                    deduplicate_per_line=args.deduplicate_per_line
                )
                results.append(result)
        else:
            # data is list of text strings
            input_texts = truncate_prompts(data, tokenizer, args.max_tokens)
            for input_text in input_texts:
                result = evaluate_forward_watermarks(
                    model, tokenizer, input_text, wm_args, args.num_sources, args.verbose, output_file,
                    use_filter=args.use_filter, filter_path=args.filter_path, filter_keys=args.filter_keys,
                    sft=False, wm_tokenizer=wm_tokenizer, max_tokens=args.max_tokens,
                    deduplicate_per_line=args.deduplicate_per_line
                )
                results.append(result)

    # if True:
    #     from scipy import special
    #     for source in results[0]["multi_source_scores"]:
    #         total_scored = 0
    #         total_green = 0
    #         for res in results:
    #             total_scored += res["multi_source_totals"][source]
    #             total_green += round(res["multi_source_scores"][source] * res["multi_source_totals"][source])
    #         # compute p-value
    #         p_value = special.betainc(total_green, 1 + total_scored - total_green, 0.5)
    #         print(f"Source {source}: Total scored: {total_scored}, Total green: {total_green}, proportion: {total_green / total_scored if total_scored > 0 else 0:.4f}, p-value: {p_value:.6e}")

    # Close output file if opened
    if output_file:
        output_file.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
