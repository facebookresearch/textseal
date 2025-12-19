# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
K-gram analysis for source attribution. Idea is to:
1. Extract k-grams from training sources and count their occurrences
2. Validate k-gram overlap between validation texts and training data

salloc --partition=learn --nodes=1 --gpus-per-node=4 --qos=lowest --time=6:00:00 --job-name=dev_llmwm --cpus-per-gpu=24--mem-per-cpu=8G
cd /path/to/textseal
conda activate lingua

Run with:
    python -m apps.wmtraining.kgram \
        --sources_train /path/to/data/source1.jsonl /path/to/data/source2.jsonl \
        --source_val /path/to/data/val.jsonl \
        --ngram 3 --max_tokens 512

    # Forward mode (uses model checkpoint for predictions):
    python -m apps.wmtraining.kgram --mode forward \
        --cache_dir /path/to/cache \
        --ckpt /path/to/checkpoint/consolidated \
        --sources_train /path/to/data/source1.jsonl /path/to/data/source2.jsonl \
        --source_val /path/to/data/val1.jsonl \
        --ngram 3 --max_tokens 512 \
        
    --source_val /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data \


    python -m apps.wmtraining.kgram --mode overlap \
        --cache_dir /path/to/data \
        --sources_train  /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data /path/to/data \
        --source_val /path/to/data \
        --ngram 3 --max_tokens 512 \
"""

import os
import pickle
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from apps.common.utils.config import dataclass_from_dict, cfg_from_cli
from apps.wmtraining.lingua.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class KGramArgs:
    """Arguments for k-gram analysis."""
    
    # Input files
    sources_train: list[str] = field(default_factory=list)  # list of training JSONL files
    source_val: str = ""  # Validation JSONL file
    
    # Mode
    mode: str = "overlap"  # "overlap" or "forward"
    
    # Model for forward mode
    ckpt: str = ""  # Path to consolidated checkpoint for forward mode
    
    # Tokenization
    tokenizer_name: str = "tiktoken"  # Tokenizer name
    tokenizer_path: str = "/path/to/data"  # Path to tokenizer
    text_key: str = "text"  # Key in JSONL for the text field
    max_tokens: int = 512  # Maximum number of tokens per text
    num_samples: int = -1  # Maximum number of validation samples to process (-1 means no limit)
    
    # K-gram parameters
    ngram: int = 3  # Size of k-grams to extract
    unique_only: bool = False  # Only score k-grams that don't overlap between training sources
    
    # Cache
    cache_dir: str = ""  # Directory containing cached k-gram counts (optional)
    
    # Output
    dump_dir: str = "kgram_output"  # Directory to save results
    verbose: bool = True


def load_texts_from_jsonl(file_path: str, text_key: str = "text", max_samples: int = -1) -> list[str]:
    """Load texts from a JSONL file."""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for ii, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                if text_key in data:
                    texts.append(data[text_key])
                    if max_samples != -1 and len(texts) >= max_samples:
                        break
    return texts


def extract_kgrams(tokens: list[int], k: int) -> list[tuple[int, ...]]:
    """Extract k-grams from a list of tokens."""
    if len(tokens) < k:
        return []
    return [tuple(tokens[ii:ii+k]) for ii in range(len(tokens) - k + 1)]


def tokenize_and_truncate(text: str, tokenizer, max_tokens: int) -> list[int]:
    """Tokenize text and truncate to max_tokens."""
    tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
    return tokens[:max_tokens]


def count_lines_in_file(file_path: str) -> int:
    """Count total lines in a file for progress bar."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def save_kgram_counts(kgram_counts: dict[str, Counter], dump_dir: str, ngram: int, tokenizer):
    """Save k-gram counts to files."""
    os.makedirs(dump_dir, exist_ok=True)
    
    # Save complete k-gram counts as pickle files
    for source_name, counter in kgram_counts.items():
        counts_file = os.path.join(dump_dir, f"{source_name}_kgram_counts_ngram{ngram}.pkl")
        with open(counts_file, 'wb') as f:
            pickle.dump(counter, f)
        print(f"Saved k-gram counts for {source_name} to {counts_file}")
    
    # Save training k-gram statistics
    stats_file = os.path.join(dump_dir, f"training_stats_ngram{ngram}.json")
    training_stats = {}
    for source_name, counter in kgram_counts.items():
        top_10_kgrams = []
        for kgram_tuple, count in counter.most_common(10):
            # Convert token IDs to text
            kgram_text = tokenizer.decode(list(kgram_tuple))
            top_10_kgrams.append({
                "tokens": list(kgram_tuple),
                "text": kgram_text,
                "count": count
            })
        
        training_stats[source_name] = {
            'unique_kgrams': len(counter),
            'total_kgrams': sum(counter.values()),
            'top_10_kgrams': top_10_kgrams
        }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(training_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved training statistics to {stats_file}")


def build_kgram_counts_from_sources(
    sources_train: list[str],
    tokenizer,
    ngram: int,
    text_key: str = "text",
    max_tokens: int = 512,
    verbose: bool = True
) -> dict[str, Counter]:
    """
    Build k-gram counts from training sources.
    
    Args:
        sources_train: list of paths to training JSONL files
        tokenizer: Tokenizer for text processing
        ngram: Size of k-grams
        text_key: Key for text field in JSONL
        max_tokens: Maximum tokens per text
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping source file names to k-gram counters
    """
    source_kgram_counts = {}
    
    for source_file in tqdm(sources_train, desc="Processing training sources"):
        if verbose:
            print(f"Processing training source: {source_file}")
        
        # Count lines for progress bar
        total_lines = count_lines_in_file(source_file)
        
        kgram_counter = Counter()
        processed_texts = 0
        
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing texts", disable=not verbose):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if text_key in data:
                            text = data[text_key]
                            tokens = tokenize_and_truncate(text, tokenizer, max_tokens)
                            kgrams = extract_kgrams(tokens, ngram)
                            kgram_counter.update(kgrams)
                            processed_texts += 1
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"Warning: Skipping invalid JSON line in {source_file}")
                        continue
        
        source_name = Path(source_file).stem
        source_kgram_counts[source_name] = kgram_counter
        
        if verbose:
            print(f"  Total texts: {processed_texts}")
            print(f"  Unique k-grams: {len(kgram_counter)}")
            print(f"  Total k-gram occurrences: {sum(kgram_counter.values())}")
    
    return source_kgram_counts


def get_unique_kgrams_per_source(source_kgram_counts: dict[str, Counter]) -> dict[str, set]:
    """
    Identify k-grams that are unique to each training source (don't overlap with other sources).
    Args:
        source_kgram_counts: Dictionary mapping source names to k-gram counters
    Returns:
        Dictionary mapping source names to sets of unique k-grams
    """
    unique_kgrams = {}
    source_names = list(source_kgram_counts.keys())
    
    for source_name in source_names:
        source_kgrams = set(source_kgram_counts[source_name].keys())
        
        # Find k-grams that appear in other sources
        overlapping_kgrams = set()
        for other_source in source_names:
            if other_source != source_name:
                other_kgrams = set(source_kgram_counts[other_source].keys())
                overlapping_kgrams.update(source_kgrams.intersection(other_kgrams))
        
        # Unique k-grams are those that don't overlap with any other source
        unique_kgrams[source_name] = source_kgrams - overlapping_kgrams
    
    return unique_kgrams


def validate_kgram_overlap(
    source_val: str,
    source_kgram_counts: dict[str, Counter],
    tokenizer,
    ngram: int,
    text_key: str = "text",
    max_tokens: int = 512,
    verbose: bool = True,
    num_samples: int = -1,
    unique_only: bool = False
) -> list[dict]:
    """
    Validate k-gram overlap between validation texts and training sources.
    
    Args:
        source_val: Path to validation JSONL file
        source_kgram_counts: Dictionary of training source k-gram counts
        tokenizer: Tokenizer for text processing
        ngram: Size of k-grams
        text_key: Key for text field in JSONL
        max_tokens: Maximum tokens per text
        verbose: Whether to print progress
        num_samples: Maximum number of validation samples to process
        unique_only: Only consider k-grams that don't overlap between training sources
        
    Returns:
        list of dictionaries containing overlap statistics for each validation text
    """
    if verbose:
        print(f"Processing validation source: {source_val}")
        if unique_only:
            print("  Using unique-only mode: only considering k-grams that don't overlap between training sources")
    
    # Get unique k-grams per source if unique_only is enabled
    unique_kgrams_per_source = None
    if unique_only:
        unique_kgrams_per_source = get_unique_kgrams_per_source(source_kgram_counts)
        if verbose:
            print("  Unique k-grams per source:")
            for source_name, unique_kgrams in unique_kgrams_per_source.items():
                total_kgrams = len(source_kgram_counts[source_name])
                unique_count = len(unique_kgrams)
                unique_ratio = unique_count / total_kgrams if total_kgrams > 0 else 0.0
                print(f"    {source_name}: {unique_count}/{total_kgrams} ({unique_ratio:.2%} unique)")
    
    # Count lines for progress bar
    total_lines = count_lines_in_file(source_val)
    
    results = []
    text_index = 0
    
    with open(source_val, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing validation texts"):
            if line.strip():
                try:
                    data = json.loads(line)
                    if text_key in data:
                        text = data[text_key]
                        tokens = tokenize_and_truncate(text, tokenizer, max_tokens)
                        val_kgrams = extract_kgrams(tokens, ngram)
                        val_kgram_set = set(val_kgrams)
                        
                        # Calculate overlap with each training source
                        overlaps = {}
                        for source_name, train_counter in source_kgram_counts.items():
                            if unique_only:
                                # Use only unique k-grams for this source
                                train_kgram_set = unique_kgrams_per_source[source_name]
                            else:
                                # Use all k-grams for this source
                                train_kgram_set = set(train_counter.keys())
                            
                            common_kgrams = val_kgram_set.intersection(train_kgram_set)
                            
                            # Calculate statistics
                            overlap_count = len(common_kgrams)
                            val_total = len(val_kgram_set)
                            train_total = len(train_kgram_set)
                            
                            # Calculate overlap ratios
                            val_overlap_ratio = overlap_count / val_total if val_total > 0 else 0.0
                            train_overlap_ratio = overlap_count / train_total if train_total > 0 else 0.0
                            
                            # Calculate frequency-weighted overlap (only for k-grams that exist in train_counter)
                            if unique_only:
                                # For unique mode, normalize by count of unique k-grams only
                                train_total_count = sum(train_counter[kgram] for kgram in unique_kgrams_per_source[source_name] if kgram in train_counter)
                            else:
                                train_total_count = sum(train_counter.values())
                            kgram_count = sum(train_counter[kgram] for kgram in common_kgrams if kgram in train_counter)
                            kgram_frequency = kgram_count / train_total_count if train_total_count > 0 else 0.0
                            
                            overlaps[source_name] = {
                                'common_kgrams': overlap_count,
                                'val_total_kgrams': val_total,
                                'train_total_kgrams': train_total,
                                'val_overlap_ratio': val_overlap_ratio,
                                'train_overlap_ratio': train_overlap_ratio,
                                'kgram_count': kgram_count,
                                'kgram_frequency': kgram_frequency
                            }
                        
                        result = {
                            'text_index': text_index,
                            'text': text[:200] + "..." if len(text) > 200 else text,  # Truncate for storage
                            'num_tokens': len(tokens),
                            'num_kgrams': len(val_kgrams),
                            'unique_kgrams': len(val_kgram_set),
                            'overlaps': overlaps
                        }
                        
                        results.append(result)
                        text_index += 1
                        if num_samples != -1 and text_index >= num_samples:
                            break
                        
                except json.JSONDecodeError:
                    if verbose:
                        print(f"Warning: Skipping invalid JSON line in {source_val}")
                    continue
    
    if verbose:
        print(f"  Total validation texts: {text_index}")
    
    return results


def validate_kgram_overlap_forward(
    source_val: str,
    source_kgram_counts: dict[str, Counter],
    model,
    tokenizer,
    ngram: int,
    text_key: str = "text",
    max_tokens: int = 512,
    verbose: bool = True,
    num_samples: int = -1,
    unique_only: bool = False
) -> list[dict]:
    """
    Validate k-gram overlap using forward pass predictions.
    
    Args:
        source_val: Path to validation JSONL file
        source_kgram_counts: Dictionary of training source k-gram counts
        model: Language model for forward pass
        tokenizer: Tokenizer for text processing
        ngram: Size of k-grams
        text_key: Key for text field in JSONL
        max_tokens: Maximum tokens per text
        verbose: Whether to print progress
        num_samples: Maximum number of validation samples to process
        unique_only: Only consider k-grams that don't overlap between training sources
        
    Returns:
        list of dictionaries containing overlap statistics for each validation text
    """
    if verbose:
        print(f"Processing validation source with forward pass: {source_val}")
        if unique_only:
            print("  Using unique-only mode: only considering k-grams that don't overlap between training sources")
    
    # Get unique k-grams per source if unique_only is enabled
    unique_kgrams_per_source = None
    if unique_only:
        unique_kgrams_per_source = get_unique_kgrams_per_source(source_kgram_counts)
        if verbose:
            print("  Unique k-grams per source:")
            for source_name, unique_kgrams in unique_kgrams_per_source.items():
                total_kgrams = len(source_kgram_counts[source_name])
                unique_count = len(unique_kgrams)
                unique_ratio = unique_count / total_kgrams if total_kgrams > 0 else 0.0
                print(f"    {source_name}: {unique_count}/{total_kgrams} ({unique_ratio:.2%} unique)")
    
    # Count lines for progress bar
    total_lines = count_lines_in_file(source_val)
    
    results = []
    text_index = 0
    
    with open(source_val, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing validation texts (forward)"):
            if line.strip():
                try:
                    data = json.loads(line)
                    if text_key in data:
                        text = data[text_key]
                        input_tokens = tokenize_and_truncate(text, tokenizer, max_tokens)
                        
                        # Perform forward pass
                        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).cuda()
                        
                        with torch.no_grad():
                            logits = model(input_tensor)  # 1 x seq x vocab_size
                            pred_tokens = torch.argmax(logits, dim=-1).squeeze(0)  # seq
                        
                        # Create k-grams from (k-1)-gram context + predicted token
                        val_kgrams = []
                        for ii in range(len(input_tokens) - ngram + 1):
                            # Take (k-1) tokens from input and 1 predicted token
                            context = input_tokens[ii:ii+ngram-1]
                            pred_token = pred_tokens[ii+ngram-2].item()  # Predicted token at position ii+ngram-2
                            kgram = tuple(context + [pred_token])
                            val_kgrams.append(kgram)
                        
                        val_kgram_set = set(val_kgrams)
                        
                        # Calculate overlap with each training source
                        overlaps = {}
                        for source_name, train_counter in source_kgram_counts.items():
                            if unique_only:
                                # Use only unique k-grams for this source
                                train_kgram_set = unique_kgrams_per_source[source_name]
                            else:
                                # Use all k-grams for this source
                                train_kgram_set = set(train_counter.keys())
                            
                            common_kgrams = val_kgram_set.intersection(train_kgram_set)
                            
                            # Calculate statistics
                            overlap_count = len(common_kgrams)
                            val_total = len(val_kgram_set)
                            train_total = len(train_kgram_set)
                            
                            # Calculate overlap ratios
                            val_overlap_ratio = overlap_count / val_total if val_total > 0 else 0.0
                            train_overlap_ratio = overlap_count / train_total if train_total > 0 else 0.0
                            
                            # Calculate frequency-weighted overlap (only for k-grams that exist in train_counter)
                            if unique_only:
                                # For unique mode, normalize by count of unique k-grams only
                                train_total_count = sum(train_counter[kgram] for kgram in unique_kgrams_per_source[source_name] if kgram in train_counter)
                            else:
                                train_total_count = sum(train_counter.values())
                            kgram_count = sum(train_counter[kgram] for kgram in common_kgrams if kgram in train_counter)
                            kgram_frequency = kgram_count / train_total_count if train_total_count > 0 else 0.0
                            
                            overlaps[source_name] = {
                                'common_kgrams': overlap_count,
                                'val_total_kgrams': val_total,
                                'train_total_kgrams': train_total,
                                'val_overlap_ratio': val_overlap_ratio,
                                'train_overlap_ratio': train_overlap_ratio,
                                'kgram_count': kgram_count,
                                'kgram_frequency': kgram_frequency
                            }
                        
                        result = {
                            'text_index': text_index,
                            'text': text[:200] + "..." if len(text) > 200 else text,  # Truncate for storage
                            'num_input_tokens': len(input_tokens),
                            'num_kgrams': len(val_kgrams),
                            'unique_kgrams': len(val_kgram_set),
                            'overlaps': overlaps,
                            'mode': 'forward'
                        }
                        
                        results.append(result)
                        text_index += 1
                        if num_samples != -1 and text_index >= num_samples:
                            break
                        
                except json.JSONDecodeError:
                    if verbose:
                        print(f"Warning: Skipping invalid JSON line in {source_val}")
                    continue
    
    if verbose:
        print(f"  Total validation texts: {text_index}")
    
    return results


def save_results(results: list[dict], kgram_counts: dict[str, Counter], dump_dir: str, ngram: int):
    """Save analysis results to files."""
    os.makedirs(dump_dir, exist_ok=True)
    
    # Save validation results
    results_file = os.path.join(dump_dir, f"validation_overlap_ngram{ngram}.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Saved validation results to {results_file}")


def print_summary(results: list[dict], source_names: list[str]):
    """Print summary statistics."""
    print("\n=== K-gram Overlap Summary ===")
    
    # Calculate average overlaps per source
    avg_overlaps = {}
    for source_name in source_names:
        val_ratios = [r['overlaps'][source_name]['val_overlap_ratio'] for r in results]
        weighted_ratios = [r['overlaps'][source_name]['kgram_frequency'] for r in results]
        
        avg_overlaps[source_name] = {
            'avg_val_overlap_ratio': sum(val_ratios) / len(val_ratios),
            'avg_kgram_frequency': sum(weighted_ratios) / len(weighted_ratios)
        }
   
   
    for source_name, stats in avg_overlaps.items():
        print(f"\nSource: {source_name}")
        print(f"  Average validation overlap ratio: {stats['avg_val_overlap_ratio']:.4f}")
        print(f"  Average weighted overlap ratio: {stats['avg_kgram_frequency']:.4f}")


def load_existing_kgram_counts(dump_dir: str, ngram: int, sources_train: list[str], cache_dir: str = "") -> dict[str, Counter]:
    """Load existing k-gram counts from pickle files if they exist."""
    source_kgram_counts = {}
    
    # Determine which directory to check first
    check_dirs = []
    if cache_dir:
        check_dirs.append(cache_dir)
    check_dirs.append(dump_dir)
    
    for source_file in sources_train:
        source_name = Path(source_file).stem
        loaded = False
        
        for check_dir in check_dirs:
            counts_file = os.path.join(check_dir, f"{source_name}_kgram_counts_ngram{ngram}.pkl")
            
            if os.path.exists(counts_file):
                try:
                    with open(counts_file, 'rb') as f:
                        counter = pickle.load(f)
                    source_kgram_counts[source_name] = counter
                    print(f"Loaded existing k-gram counts for {source_name} from {counts_file}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Warning: Failed to load {counts_file}: {e}")
                    continue
        
        if not loaded:
            print(f"No existing k-gram counts found for {source_name}")
            return {}  # Return empty dict to force recomputation
    
    return source_kgram_counts


def main():
    """Main k-gram analysis function."""
    # Parse arguments
    cfg = cfg_from_cli()
    sources_train = cfg.get('sources_train', None)
    if sources_train is not None:
        # Single string or list of strings.
        if isinstance(sources_train, str):
            sources_train = [sources_train]
        else:
            sources_train = list(sources_train)
    cfg['sources_train'] = sources_train

    # Merge with defaults
    default_cfg = dataclass_from_dict(KGramArgs, {})
    cfg = OmegaConf.merge(default_cfg, cfg)
    args = OmegaConf.to_object(cfg)
    
    if not args.sources_train:
        raise ValueError("Must specify training sources with --sources_train")
    if not args.source_val:
        raise ValueError("Must specify validation source with --source_val")
    if args.mode not in ["overlap", "forward"]:
        raise ValueError("Mode must be 'overlap' or 'forward'")
    if args.mode == "forward" and not args.ckpt:
        raise ValueError("Must specify checkpoint path with --ckpt for forward mode")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = build_tokenizer(args.tokenizer_name, args.tokenizer_path)
    
    # Load model if in forward mode
    model = None
    if args.mode == "forward":
        from apps.wmtraining.generate import load_consolidated_model_and_tokenizer
        print(f"Loading model from {args.ckpt}...")
        model, _, _ = load_consolidated_model_and_tokenizer(args.ckpt)
    
    # Try to load existing k-gram counts first
    print(f"\nChecking for existing {args.ngram}-gram counts...")
    if args.cache_dir:
        print(f"  Cache directory: {args.cache_dir}")
    print(f"  Output directory: {args.dump_dir}")
    
    source_kgram_counts = load_existing_kgram_counts(args.dump_dir, args.ngram, args.sources_train, args.cache_dir)
    
    if source_kgram_counts:
        print("Using existing k-gram counts.")
    else:
        # Build k-gram counts from training sources
        print(f"\nBuilding {args.ngram}-gram counts from training sources...")
        source_kgram_counts = build_kgram_counts_from_sources(
            args.sources_train,
            tokenizer,
            args.ngram,
            args.text_key,
            args.max_tokens,
            args.verbose
        )
        
        # Save k-gram counts immediately after computation
        save_kgram_counts(source_kgram_counts, args.dump_dir, args.ngram, tokenizer)
    
    # Validate k-gram overlap
    if args.mode == "overlap":
        print(f"\nValidating {args.ngram}-gram overlap with validation source...")
        results = validate_kgram_overlap(
            args.source_val,
            source_kgram_counts,
            tokenizer,
            args.ngram,
            args.text_key,
            args.max_tokens,
            args.verbose,
            args.num_samples,
            args.unique_only
        )
    elif args.mode == "forward":
        print(f"\nValidating {args.ngram}-gram overlap with forward pass predictions...")
        results = validate_kgram_overlap_forward(
            args.source_val,
            source_kgram_counts,
            model,
            tokenizer,
            args.ngram,
            args.text_key,
            args.max_tokens,
            args.verbose,
            args.num_samples,
            args.unique_only
        )
    
    # Save results
    save_results(results, source_kgram_counts, args.dump_dir, args.ngram)
    
    # Print summary
    source_names = list(source_kgram_counts.keys())
    print_summary(results, source_names)
    
    print(f"\nAnalysis complete! Results saved to {args.dump_dir}")

if __name__ == "__main__":
    main()
