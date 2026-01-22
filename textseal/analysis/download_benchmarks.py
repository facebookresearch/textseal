"""
Download and prepare benchmark datasets for contamination experiments.

This script downloads evaluation benchmarks from HuggingFace and prepares them
in JSONL format for watermarking and contamination detection experiments.

Usage:
    python apps/analysis/download_benchmarks.py --output_dir assets/benchmarks
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def prepare_mmlu(output_dir: str, num_samples: int = 5000):
    """Download and prepare MMLU dataset."""
    print(f"Downloading MMLU dataset...")
    mmlu = load_dataset("cais/mmlu", "all")
    
    mmlu_dir = os.path.join(output_dir, 'mmlu')
    os.makedirs(mmlu_dir, exist_ok=True)
    
    # Sample test items
    mmlu_test_items = list(mmlu['test'])
    random.shuffle(mmlu_test_items)
    mmlu_test_subset = mmlu_test_items[:num_samples]
    
    output_path = os.path.join(mmlu_dir, 'mmlu.chunk.0.jsonl')
    with open(output_path, 'w') as f:
        for item in mmlu_test_subset:
            answer_idx = item['answer']
            answer_text = item['choices'][answer_idx]
            item['text'] = f"Question: {item['question']}\nAnswer: {answer_text}"
            json.dump(item, f)
            f.write('\n')
    
    print(f"✓ MMLU: {len(mmlu_test_subset)} samples saved to {output_path}")
    return len(mmlu_test_subset)


def prepare_arc_easy(output_dir: str):
    """Download and prepare ARC-Easy dataset."""
    print(f"Downloading ARC-Easy dataset...")
    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    
    arc_easy_dir = os.path.join(output_dir, 'arc_easy')
    os.makedirs(arc_easy_dir, exist_ok=True)
    
    output_path = os.path.join(arc_easy_dir, 'arc_easy.chunk.0.jsonl')
    count = 0
    with open(output_path, 'w') as f:
        for item in arc_easy['test']:
            answer_key = item['answerKey']
            answer_idx = item['choices']['label'].index(answer_key)
            answer_text = item['choices']['text'][answer_idx]
            item['text'] = f"Question: {item['question']}\nAnswer: {answer_text}"
            json.dump(item, f)
            f.write('\n')
            count += 1
    
    print(f"✓ ARC-Easy: {count} samples saved to {output_path}")
    return count


def prepare_arc_challenge(output_dir: str):
    """Download and prepare ARC-Challenge dataset."""
    print(f"Downloading ARC-Challenge dataset...")
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    
    arc_challenge_dir = os.path.join(output_dir, 'arc_challenge')
    os.makedirs(arc_challenge_dir, exist_ok=True)
    
    output_path = os.path.join(arc_challenge_dir, 'arc_challenge.chunk.0.jsonl')
    count = 0
    with open(output_path, 'w') as f:
        for item in arc_challenge['test']:
            answer_key = item['answerKey']
            answer_idx = item['choices']['label'].index(answer_key)
            answer_text = item['choices']['text'][answer_idx]
            item['text'] = f"Question: {item['question']}\nAnswer: {answer_text}"
            json.dump(item, f)
            f.write('\n')
            count += 1
    
    print(f"✓ ARC-Challenge: {count} samples saved to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare benchmark datasets for contamination experiments"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/benchmarks",
        help="Output directory for benchmark files (default: assets/benchmarks)"
    )
    parser.add_argument(
        "--mmlu_samples",
        type=int,
        default=5000,
        help="Number of MMLU test samples to use (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading benchmarks to: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Download and prepare each benchmark
    counts = {}
    counts['mmlu'] = prepare_mmlu(args.output_dir, args.mmlu_samples)
    counts['arc_easy'] = prepare_arc_easy(args.output_dir)
    counts['arc_challenge'] = prepare_arc_challenge(args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"MMLU:          {counts['mmlu']:>6} samples (~325,000 tokens)")
    print(f"ARC-Easy:      {counts['arc_easy']:>6} samples (~112,000 tokens)")
    print(f"ARC-Challenge: {counts['arc_challenge']:>6} samples (~64,000 tokens)")
    print(f"{'='*60}")
    print(f"Total:         {sum(counts.values()):>6} samples (~501,000 tokens)")
    print(f"{'='*60}\n")
    print(f"✓ All benchmarks downloaded successfully!")

if __name__ == "__main__":
    main()
