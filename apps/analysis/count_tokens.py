# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Count tokens in a single file
    python -m apps.analysis.count_tokens /datasets/llm/pretraining/dclm/dclm_baseline_1.0.chunk.00.jsonl --num_processes 10
    python -m apps.analysis.count_tokens /datasets/llm/pretraining/dclm/dclm_baseline_1.0.chunk.00.jsonl --time_limit 100 

Count tokens in all JSONL files in a folder
    python -m apps.analysis.count_tokens /datasets/llm/pretraining/dclm --num_processes 10

Count tokens in all JSONL files recursively in subdirectories
    python -m apps.analysis.count_tokens /path/to/folder --recursive --num_processes 10

Save results and show token histogram
    python -m apps.analysis.count_tokens /path/to/folder --save_dir ./results --do_hist True

Use custom tokenizer
    python -m apps.analysis.count_tokens /path/to/folder --tokenizer_name sentencepiece --tokenizer_path /path/to/tokenizer.model

Approximate count with time limit (faster for large files)
    python -m apps.analysis.count_tokens /path/to/folder --time_limit 30


python -m apps.analysis.count_tokens /path/to/data/ --recursive --num_processes 10
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional
from functools import partial
import os
import argparse
import json
import numpy as np
from pathlib import Path
import time

import tqdm

from apps.wmtraining.lingua.tokenizer import build_tokenizer
from apps.wmtraining.lingua.data import TRAIN_DATA_FILE_PATTERN


def get_content_key(fpath: str) -> str:
    """Determine whether to use 'text' or 'content' key by sampling the file."""
    with open(fpath, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "text" in data:
                    return "text"
                elif "content" in data:
                    return "content"
                else:
                    raise ValueError(f"No 'text' or 'content' key found in {fpath}")
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON lines found in {fpath}")


def _count(rank: int, size: int, fpath: str, tokenizer_name: str = "tiktoken", 
           tokenizer_path: Optional[str] = "~/.cache/tokenizers/llama3.model") -> Tuple[List[int], List[int], List[int]]:
    """Count tokens in a file using distributed processing."""
    
    # Validate inputs
    assert os.path.isfile(fpath), f"File not found: {fpath}"
    assert 0 <= rank < size, f"Invalid rank/size: {rank}/{size}"
    
    content_key = get_content_key(fpath)
    
    # Build tokenizer and get vocab size
    tokenizer = build_tokenizer(name=tokenizer_name, path=tokenizer_path)
    vocab_size = tokenizer.n_words
    
    # Initialize counters
    n_chars: List[int] = []
    n_toks: List[int] = []
    tok_counts = [0] * vocab_size
    
    # Only show progress bar for rank 0 to avoid cluttered output
    show_progress = (rank == 0 and size > 1) or size == 1
    
    with open(fpath, "r") as f:
        lines = enumerate(f)
        if show_progress:
            lines = tqdm.tqdm(lines, desc=f"Process {rank+1}/{size}" if size > 1 else "Processing")
            
        for i, line in lines:
            if i % size != rank:
                continue
                
            try:
                x = json.loads(line)
                text = x[content_key]
                tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
                
                n_chars.append(len(text))
                n_toks.append(len(tokens))
                
                # Count token frequencies
                for tok in tokens:
                    if 0 <= tok < vocab_size:
                        tok_counts[tok] += 1
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid line {i} in {fpath}: {e}")
                continue

    return n_chars, n_toks, tok_counts


def approximate_count_tokens(fpath: str, time_limit: int, tokenizer_name: str = "tiktoken",
                           tokenizer_path: Optional[str] = "~/.cache/tokenizers/llama3.model") -> dict:
    """Approximate token count by sampling lines and using file size."""
    
    print(f"Approximating token count for {fpath} (time limit: {time_limit}s)")
    
    content_key = get_content_key(fpath)
    tokenizer = build_tokenizer(name=tokenizer_name, path=tokenizer_path)
    
    # Get file size in bytes
    file_size = os.path.getsize(fpath)
    print(f"File size: {file_size:,} bytes")
    
    start_time = time.time()
    sample_bytes = 0
    sample_tokens = 0
    sample_chars = 0
    sample_lines = 0
    
    with open(fpath, "r") as f:
        progress_bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc='Processing')
        
        for line_num, line in enumerate(f):
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"\nTime limit reached after {line_num + 1} lines")
                break
                
            try:
                x = json.loads(line)
                text = x[content_key]
                tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
                
                # Accumulate sample statistics
                line_bytes = len(line.encode('utf-8'))
                sample_bytes += line_bytes
                sample_tokens += len(tokens)
                sample_chars += len(text)
                sample_lines += 1
                
                # Update progress bar
                progress_bar.update(line_bytes)
                progress_bar.set_postfix({
                    'tokens': f'{sample_tokens:,}',
                    'lines': sample_lines
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid line {line_num} in {fpath}: {e}")
                continue

            print(sample_tokens, sample_bytes, sample_chars, sample_lines)
            break
        
        progress_bar.close()
    
    if sample_bytes == 0 or sample_tokens == 0:
        raise ValueError("No valid data sampled - cannot approximate")
    
    # Calculate ratios from sample
    tokens_per_byte = sample_tokens / sample_bytes
    chars_per_byte = sample_chars / sample_bytes
    tokens_per_line = sample_tokens / sample_lines
    chars_per_line = sample_chars / sample_lines
    
    # Estimate totals using file size
    estimated_total_tokens = int(file_size * tokens_per_byte)
    estimated_total_chars = int(file_size * chars_per_byte)
    estimated_total_lines = int(file_size / (sample_bytes / sample_lines))
    
    print(f"\n=== Approximation Results for {fpath} ===")
    print(f"Sample statistics:")
    print(f"  Lines sampled: {sample_lines:,}")
    print(f"  Bytes sampled: {sample_bytes:,}")
    print(f"  Tokens sampled: {sample_tokens:,}")
    print(f"  Characters sampled: {sample_chars:,}")
    print(f"Ratios from sample:")
    print(f"  Tokens per byte: {tokens_per_byte:.6f}")
    print(f"  Characters per byte: {chars_per_byte:.6f}")
    print(f"  Tokens per line: {tokens_per_line:.2f}")
    print(f"  Characters per line: {chars_per_line:.2f}")
    print(f"Estimated totals:")
    print(f"  Lines: ~{estimated_total_lines:,}")
    print(f"  Tokens: ~{estimated_total_tokens:,}")
    print(f"  Characters: ~{estimated_total_chars:,}")
    print(f"  Tokens per character ratio: {estimated_total_tokens / estimated_total_chars:.3f}")
    
    return {
        'estimated_total_tokens': estimated_total_tokens,
        'estimated_total_chars': estimated_total_chars,
        'estimated_total_lines': estimated_total_lines,
        'sample_lines': sample_lines,
        'sample_tokens': sample_tokens,
        'sample_chars': sample_chars,
        'tokens_per_byte': tokens_per_byte,
        'file_size': file_size,
        'is_approximation': True
    }


def count_tokens_in_file(fpath: str, num_processes: int = 1, save_dir: Optional[str] = None, 
                        do_hist: bool = False, tokenizer_name: str = "tiktoken",
                        tokenizer_path: Optional[str] = "~/.cache/tokenizers/llama3.model",
                        time_limit: Optional[int] = None):
    """Count tokens in a single JSONL file."""
    
    # Use approximation if time limit is provided
    if time_limit is not None:
        return approximate_count_tokens(fpath, time_limit, tokenizer_name, tokenizer_path)
    
    print(f"Processing file: {fpath}")
    
    # Get vocab size from tokenizer
    tokenizer = build_tokenizer(name=tokenizer_name, path=tokenizer_path)
    vocab_size = tokenizer.n_words
    print(f"Using tokenizer '{tokenizer_name}' with vocab size: {vocab_size}")
    
    if num_processes == 1:
        n_chars, n_toks, tok_counts = _count(
            rank=0, size=1, fpath=fpath, tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path
        )
    else:
        n_chars = []
        n_toks = []
        tok_counts = [0] * vocab_size
        
        f = partial(_count, size=num_processes, fpath=fpath, 
                   tokenizer_name=tokenizer_name, tokenizer_path=tokenizer_path)
        args = list(range(num_processes))
        
        with ProcessPoolExecutor(max_workers=min(num_processes, 40)) as executor:
            for i, (_n_chars, _n_toks, _tok_counts) in enumerate(executor.map(f, args, chunksize=1)):
                print(f"Process {i+1}/{len(args)} completed -- {len(_n_toks)} lines processed.")
                n_chars.extend(_n_chars)
                n_toks.extend(_n_toks)
                for j, c in enumerate(_tok_counts):
                    tok_counts[j] += c

    # Print statistics
    assert len(n_chars) == len(n_toks)
    total_tokens = sum(n_toks)
    total_chars = sum(n_chars)
    
    print(f"\n=== Statistics for {fpath} ===")
    print(f"Lines: {len(n_toks):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average tokens per line: {np.mean(n_toks):.2f} (±{np.std(n_toks):.2f})")
    print(f"Average characters per line: {np.mean(n_chars):.2f} (±{np.std(n_chars):.2f})")
    print(f"Tokens per character ratio: {total_tokens / total_chars:.3f}")
    
    if do_hist:
        print(f"\nToken frequency distribution:")
        non_zero_counts = [(i, count) for i, count in enumerate(tok_counts) if count > 0]
        print(f"Unique tokens used: {len(non_zero_counts)}/{vocab_size}")
        if len(non_zero_counts) <= 20:
            for token_id, count in sorted(non_zero_counts, key=lambda x: x[1], reverse=True):
                print(f"  Token {token_id}: {count}")
    
    # Save results if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename_base = Path(fpath).stem
        np.save(os.path.join(save_dir, f"{filename_base}_n_chars.npy"), np.array(n_chars))
        np.save(os.path.join(save_dir, f"{filename_base}_n_toks.npy"), np.array(n_toks))
        np.save(os.path.join(save_dir, f"{filename_base}_tok_counts.npy"), np.array(tok_counts))
        print(f"Results saved to {save_dir}")
    
    return {
        'n_chars': n_chars,
        'n_toks': n_toks,
        'tok_counts': tok_counts,
        'total_tokens': total_tokens,
        'total_chars': total_chars
    }


def count_tokens_in_folder(folder_path: str, num_processes: int = 1, save_dir: Optional[str] = None,
                          do_hist: bool = False, tokenizer_name: str = "tiktoken",
                          tokenizer_path: Optional[str] = "~/.cache/tokenizers/llama3.model",
                          file_pattern: str = TRAIN_DATA_FILE_PATTERN, time_limit: Optional[int] = None,
                          recursive: bool = False):
    """Count tokens in all JSONL files in a folder."""
    
    folder_path = Path(folder_path)
    
    if recursive:
        # Search recursively for all .jsonl files in subdirectories
        jsonl_files = list(folder_path.rglob("*.jsonl"))
        search_desc = "recursively"
    else:
        # Use the specified pattern in the current directory only
        jsonl_files = list(folder_path.glob(file_pattern))
        search_desc = f"matching pattern {file_pattern}"
    
    if not jsonl_files:
        print(f"No JSONL files found in {folder_path} {search_desc}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files in {folder_path} {search_desc}")
    
    # Get vocab size from tokenizer
    tokenizer = build_tokenizer(name=tokenizer_name, path=tokenizer_path)
    vocab_size = tokenizer.n_words
    print(f"Using tokenizer '{tokenizer_name}' with vocab size: {vocab_size}")
    
    all_results = {}
    total_lines = 0
    total_tokens = 0
    total_chars = 0
    combined_tok_counts = [0] * vocab_size
    is_approximation = False
    
    for fpath in sorted(jsonl_files):
        print(f"\n{'='*60}")
        results = count_tokens_in_file(
            str(fpath), num_processes=num_processes, save_dir=save_dir,
            do_hist=do_hist, tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path, time_limit=time_limit
        )
        
        all_results[str(fpath)] = results
        
        if results.get('is_approximation', False):
            is_approximation = True
            total_lines += results['estimated_total_lines']
            total_tokens += results['estimated_total_tokens']
            total_chars += results['estimated_total_chars']
        else:
            total_lines += len(results['n_toks'])
            total_tokens += results['total_tokens']
            total_chars += results['total_chars']
            
            for i, count in enumerate(results['tok_counts']):
                combined_tok_counts[i] += count
    
    # Print combined statistics
    print(f"\n{'='*60}")
    result_type = "APPROXIMATE" if is_approximation else "EXACT"
    print(f"=== {result_type} COMBINED STATISTICS FOR {len(jsonl_files)} FILES ===")
    print(f"Total lines: {'~' if is_approximation else ''}{total_lines:,}")
    print(f"Total tokens: {'~' if is_approximation else ''}{total_tokens:,}")
    print(f"Total characters: {'~' if is_approximation else ''}{total_chars:,}")
    print(f"Average tokens per line (overall): {total_tokens / total_lines:.2f}")
    print(f"Average characters per line (overall): {total_chars / total_lines:.2f}")
    print(f"Tokens per character ratio (overall): {total_tokens / total_chars:.3f}")
    
    if do_hist and not is_approximation:
        non_zero_counts = [(i, count) for i, count in enumerate(combined_tok_counts) if count > 0]
        print(f"\nCombined unique tokens used: {len(non_zero_counts)}/{vocab_size}")
    elif is_approximation:
        print("\nNote: Token frequency histogram not available for approximations")
    
    # Save combined results
    if save_dir is not None:
        combined_save_dir = os.path.join(save_dir, "combined")
        os.makedirs(combined_save_dir, exist_ok=True)
        np.save(os.path.join(combined_save_dir, "combined_tok_counts.npy"), np.array(combined_tok_counts))
        
        # Save summary stats
        summary = {
            'total_files': len(jsonl_files),
            'total_lines': total_lines,
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'avg_tokens_per_line': total_tokens / total_lines,
            'avg_chars_per_line': total_chars / total_lines,
            'tokens_per_char': total_tokens / total_chars
        }
        
        with open(os.path.join(combined_save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Combined results saved to {combined_save_dir}")
    
    return all_results


def count(path: str, num_processes: int = 1, save_dir: Optional[str] = None,
          do_hist: bool = False, tokenizer_name: str = "tiktoken",
          tokenizer_path: Optional[str] = "~/.cache/tokenizers/llama3.model",
          time_limit: Optional[int] = None, recursive: bool = False):
    """Count tokens in a file or folder of JSONL files."""
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        return count_tokens_in_file(
            path, num_processes=num_processes, save_dir=save_dir,
            do_hist=do_hist, tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path, time_limit=time_limit
        )
    elif path_obj.is_dir():
        return count_tokens_in_folder(
            path, num_processes=num_processes, save_dir=save_dir,
            do_hist=do_hist, tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path, time_limit=time_limit,
            recursive=recursive
        )
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")


def main():
    """Main function with argparse CLI."""
    parser = argparse.ArgumentParser(description="Count tokens in JSONL files")
    parser.add_argument("path", help="Path to file or folder containing JSONL files")
    parser.add_argument("--num_processes", type=int, default=1, 
                       help="Number of processes to use (default: 1)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save results (default: None)")
    parser.add_argument("--do_hist", action="store_true",
                       help="Show token frequency histogram")
    parser.add_argument("--tokenizer_name", type=str, default="tiktoken",
                       help="Tokenizer name (default: tiktoken)")
    parser.add_argument("--tokenizer_path", type=str, 
                       default="~/.cache/tokenizers/llama3.model",
                       help="Path to tokenizer model")
    parser.add_argument("--time_limit", type=int, default=None,
                       help="Time limit in seconds for approximate counting (default: None for exact count)")
    parser.add_argument("--recursive", action="store_true",
                       help="Search for all .jsonl files recursively in subdirectories (ignores file_pattern)")
    
    args = parser.parse_args()
    
    count(
        path=args.path,
        num_processes=args.num_processes,
        save_dir=args.save_dir,
        do_hist=args.do_hist,
        tokenizer_name=args.tokenizer_name,
        tokenizer_path=args.tokenizer_path,
        time_limit=args.time_limit,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()

# Usage examples:
# python -m apps.analysis.count_tokens /path/to/file.jsonl --num_processes 10
# python -m apps.analysis.count_tokens /path/to/folder --num_processes 10 --save_dir ./results
# python -m apps.analysis.count_tokens /path/to/folder --tokenizer_name sentencepiece --tokenizer_path /path/to/tokenizer.model
# python -m apps.analysis.count_tokens /path/to/folder --time_limit 30  # Approximate count with 30s time limit
# python -m apps.analysis.count_tokens /path/to/folder --recursive  # Search all .jsonl files recursively in subdirectories
