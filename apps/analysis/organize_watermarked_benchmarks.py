# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/usr/bin/env python3
"""
Script to organize watermarked benchmark outputs into training-ready directory structure.

This script takes the output from watermarking experiments (which are in folders like
input_path=0_watermark.secret_key=42) and organizes them into a directory structure
suitable for contamination training:
  
  output_dir/
    arc_easy/
      arc_easy.secret_key=XXX.chunk.0.jsonl
    arc_challenge/
      arc_challenge.secret_key=XXX.chunk.0.jsonl
    mmlu/
      mmlu.secret_key=XXX.chunk.0.jsonl

It also computes and displays statistics about the watermark quality (green token proportions).

Usage:
    python organize_watermarked_benchmarks.py \
        --source /path/to/watermarked/output \
        --dest /path/to/benchmarks_organized \
        --delta 4 \
        --group-name group1
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# Mapping of input_path indices to benchmark names
BENCHMARK_MAP = {
    '0': 'arc_easy',
    '1': 'arc_challenge',
    '2': 'mmlu'
}


def parse_folder_name(folder_name: str) -> Tuple[str, str, str]:
    """
    Parse folder name to extract input_path, delta, and secret_key.
    
    Example: "input_path=0_watermark.secret_key=42_watermark.delta=4"
    Returns: ('0', '4', '42')
    """
    parts = folder_name.split('_')
    
    input_path = None
    delta = None
    secret_key = None
    
    for part in parts:
        if part.startswith('input_path=') or part.startswith('path='):
            input_path = part.split('=')[1]
        elif part.startswith('watermark.delta=') or part.startswith('delta='):
            delta = part.split('=')[1]
        elif part.startswith('watermark.secret_key=') or part.startswith('key='):
            secret_key = part.split('=')[1]
    
    return input_path, delta, secret_key


def process_benchmark_file(
    result_file: str,
    benchmark_name: str,
    dest_file: str
) -> Dict[str, float]:
    """
    Process a results.jsonl file and save it to the destination with proper format.
    
    Returns statistics about the watermarked data.
    """
    green_proportions = []
    num_green_tokens = []
    num_scored_tokens = []
    num_items = 0
    
    with open(result_file, 'r') as f_in, open(dest_file, 'w') as f_out:
        for line in f_in:
            item = json.loads(line)
            
            # Add wm_text_all field based on benchmark type
            if benchmark_name in ['arc_challenge', 'arc_easy']:
                # ARC format
                answer_key = item['answerKey']
                answer_idx = item['choices']['label'].index(answer_key)
                answer_text = item['choices']['text'][answer_idx]
                item['wm_text_all'] = f"Question: {item['wm_text']}\nAnswer: {answer_text}"
            else:  # mmlu
                # MMLU format
                answer_idx = item['answer']
                answer_text = item['choices'][answer_idx]
                item['wm_text_all'] = f"Question: {item['wm_text']}\nAnswer: {answer_text}"
            
            # Extract watermark statistics if available
            if 'wm_eval' in item:
                wm_eval = item['wm_eval']
                if 'green_proportion' in wm_eval:
                    green_proportions.append(wm_eval['green_proportion'])
                if 'toks_green' in wm_eval:
                    num_green_tokens.append(wm_eval['toks_green'])
                if 'toks_scored' in wm_eval:
                    num_scored_tokens.append(wm_eval['toks_scored'])
            
            json.dump(item, f_out)
            f_out.write('\n')
            num_items += 1
    
    # Compute statistics
    stats = {
        'num_items': num_items,
        'avg_green_proportion': np.mean(green_proportions) if green_proportions else 0.0,
        'std_green_proportion': np.std(green_proportions) if green_proportions else 0.0,
        'total_green_tokens': sum(num_green_tokens) if num_green_tokens else 0,
        'total_scored_tokens': sum(num_scored_tokens) if num_scored_tokens else 0,
    }
    
    return stats


def organize_benchmarks(
    source_dir: str,
    dest_dir: str,
    verbose: bool = True
):
    """
    Organize watermarked benchmarks from source to destination directory.
    
    Args:
        source_dir: Source directory containing watermark experiment outputs
        dest_dir: Destination directory for organized benchmarks
        verbose: Whether to print progress information
    """
    # Get all subdirectories in source
    subdirs = glob.glob(os.path.join(source_dir, '*'))
    
    all_stats = {}
    
    for subdir in sorted(subdirs):
        if not os.path.isdir(subdir):
            continue
        
        folder_name = os.path.basename(subdir)
        
        # Parse folder name
        input_path, folder_delta, secret_key = parse_folder_name(folder_name)
        
        # Check if results.jsonl exists
        result_file = os.path.join(subdir, 'results.jsonl')
        if not os.path.exists(result_file):
            if verbose:
                print(f"⚠️  Skipping {folder_name}: no results.jsonl")
            continue
        
        # Get benchmark name from input_path
        if input_path not in BENCHMARK_MAP:
            if verbose:
                print(f"⚠️  Skipping {folder_name}: unknown input_path={input_path}")
            continue
        
        benchmark_name = BENCHMARK_MAP[input_path]
        
        # Create destination folder structure: dest_dir/benchmark_name/
        dest_folder = os.path.join(dest_dir, benchmark_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Destination file with secret key in filename
        dest_file = os.path.join(dest_folder, f"{benchmark_name}.secret_key={secret_key}.chunk.0.jsonl")
        
        # Process the file
        if verbose:
            print(f"\n📝 Processing: {benchmark_name} (delta={folder_delta}, key={secret_key})")
        
        stats = process_benchmark_file(result_file, benchmark_name, dest_file)
        
        # Store stats
        delta_str = folder_delta if folder_delta else "unknown"
        stats_key = f"{benchmark_name}_key={secret_key}_delta={delta_str}"
        all_stats[stats_key] = stats
        all_stats[stats_key]['benchmark'] = benchmark_name
        all_stats[stats_key]['secret_key'] = secret_key
        all_stats[stats_key]['delta'] = delta_str
        
        # Print statistics
        if verbose:
            print(f"   ✓ Saved to: {dest_file}")
            print(f"   📊 Items: {stats['num_items']}")
            print(f"   📊 Avg green proportion: {stats['avg_green_proportion']:.4f} ± {stats['std_green_proportion']:.4f}")
            if stats['total_scored_tokens'] > 0:
                print(f"   📊 Total green tokens: {stats['total_green_tokens']:,} / {stats['total_scored_tokens']:,} "
                      f"({100 * stats['total_green_tokens'] / stats['total_scored_tokens']:.2f}%)")
    
    return all_stats


def print_summary(stats: Dict[str, Dict[str, float]]):
    """Print summary statistics across all benchmarks."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Extract unique benchmarks, keys, and deltas from the stored metadata
    benchmarks = set()
    keys = set()
    deltas = set()
    
    for stat in stats.values():
        benchmarks.add(stat['benchmark'])
        keys.add(stat['secret_key'])
        deltas.add(stat['delta'])
    
    benchmarks = sorted(benchmarks)
    
    print(f"\nBenchmarks processed: {', '.join(benchmarks)}")
    print(f"Secret keys: {', '.join(sorted(keys, key=str))}")
    print(f"Delta values: {', '.join(sorted(deltas, key=str))}")
    
    # Print table of results
    print("\n" + "-"*80)
    print("Watermark Statistics by Benchmark")
    print("-"*80)
    print(f"{'Benchmark':<20} {'Secret Key':<15} {'Items':<10} {'Green Prop':<15} {'Scored Tokens':<20}")
    print("-"*80)
    
    total_items = 0
    total_green_tokens = 0
    total_scored_tokens = 0
    
    for benchmark in benchmarks:
        # Get all stats for this benchmark
        matching_stats = [(k, v) for k, v in stats.items() if v['benchmark'] == benchmark]
        
        for stat_key, stat in sorted(matching_stats, key=lambda x: str(x[1]['secret_key'])):
            secret_key = stat['secret_key']
            
            print(f"{benchmark:<20} {secret_key:<15} {stat['num_items']:<10} "
                  f"{stat['avg_green_proportion']:.4f}         "
                  f"{stat['total_green_tokens']:,} / {stat['total_scored_tokens']:,}")
            
            total_items += stat['num_items']
            total_green_tokens += stat['total_green_tokens']
            total_scored_tokens += stat['total_scored_tokens']
    
    print("-"*80)
    print(f"{'TOTAL':<20} {'':<15} {total_items:<10} "
          f"{total_green_tokens/total_scored_tokens if total_scored_tokens > 0 else 0:.4f}         "
          f"{total_green_tokens:,} / {total_scored_tokens:,}")
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(
        description="Organize watermarked benchmarks into training-ready structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize all benchmarks from a watermarking run
  python organize_watermarked_benchmarks.py \\
      --source /path/to/watermarked/output \
      --dest /path/to/benchmarks_organized
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing watermark experiment outputs'
    )
    
    parser.add_argument(
        '--dest',
        type=str,
        required=True,
        help='Destination directory for organized benchmarks'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source):
        print(f"❌ Error: Source directory does not exist: {args.source}")
        return 1
    
    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)
    
    print("="*80)
    print("WATERMARKED BENCHMARK ORGANIZATION")
    print("="*80)
    print(f"Source: {args.source}")
    print(f"Destination: {args.dest}")
    print("="*80)
    
    # Organize benchmarks
    stats = organize_benchmarks(
        args.source,
        args.dest,
        verbose=not args.quiet
    )
    
    # Print summary
    if stats:
        print_summary(stats)
        print(f"\n✅ Organization complete! Output saved to: {args.dest}")
    else:
        print("\n⚠️  No benchmarks were processed. Check your source directory and filters.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
