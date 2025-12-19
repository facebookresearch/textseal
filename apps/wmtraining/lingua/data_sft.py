# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
import numpy as np
import torch
from apps.wmtraining.lingua.tokenizer import build_tokenizer, TokenizerArgs

logger = logging.getLogger()


# SFTDataArgs removed - now using DataArgs from apps.wmtraining.lingua.data directly
    

@dataclass
class SFTBatch:
    """SFT batch containing tokens, labels, authors, and Q&A masks."""
    input_ids: torch.Tensor  # (batch_size, seq_len)
    labels: torch.Tensor     # (batch_size, seq_len) 
    authors: torch.Tensor    # (batch_size, seq_len) - author ID for each token
    qa_mask: torch.Tensor    # (batch_size, seq_len) - 1 for answer tokens, 0 for question
    attention_mask: torch.Tensor  # (batch_size, seq_len) - 1 for real tokens, 0 for padding


class SFTDataLoader:
    """Data loader for SFT with author-specific watermarking."""
    
    def __init__(self, args, rank: int = 0, world_size: int = 1):  # Accept DataArgs from apps.wmtraining.lingua.data
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # Build tokenizer
        self.tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
        
        # Load all authors and their data
        self.authors, self.author_to_id, self.all_examples = self._load_authors()
        
        # Group examples by author for balanced sampling
        self.examples_by_author = self._group_examples_by_author()
        
        # Setup sampling weights
        self._setup_author_weights()
        
        # Create RNG for reproducible sampling
        self.rng = np.random.RandomState(42 + rank)
        
        logger.info(f"Loaded {len(self.all_examples)} examples from {len(self.authors)} authors")
        for author, author_id in self.author_to_id.items():
            count = len(self.examples_by_author[author_id])
            logger.info(f"  {author}: {count} examples (ID: {author_id})")
    
    def _load_authors(self) -> Tuple[List[str], Dict[str, int], List[Dict]]:
        """Load author chunk.0.jsonl files from subfolders and assign author IDs."""
        data_dir = Path(self.args.root_dir) if self.args.root_dir else Path(".")
        author_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        author_files = []
        authors = []
        for author_dir in sorted(author_dirs):
            # Look for chunk.0.jsonl in each author subfolder
            chunk_files = list(author_dir.glob("*.chunk.0.jsonl"))
            if chunk_files:
                # Use the first chunk.0.jsonl file found
                author_files.append(chunk_files[0])
                authors.append(author_dir.name)
        if not author_files:
            raise ValueError(f"No chunk.0.jsonl files found in any subfolder of {data_dir}")

        # Optionally filter to specific authors
        sft_authors = getattr(self.args, 'sft_authors', None)
        if sft_authors is not None:
            allowed_authors = set(sft_authors)
            filtered_files = []
            filtered_authors = []
            for file_path, author_name in zip(author_files, authors):
                if author_name in allowed_authors:
                    filtered_files.append(file_path)
                    filtered_authors.append(author_name)
            author_files = filtered_files
            authors = filtered_authors
            logger.info(f"Filtering to {len(author_files)} authors: {sft_authors}")
        if not author_files:
            raise ValueError(f"No matching author chunk.0.jsonl files found. Available: {authors}")

        author_to_id = {author: i for i, author in enumerate(authors)}
        all_examples = []
        for i, (file_path, author_name) in enumerate(zip(author_files, authors)):
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        example = json.loads(line.strip())
                        example['author_id'] = i
                        example['file_path'] = str(file_path)
                        example['line_num'] = line_num
                        all_examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON in {file_path} line {line_num}: {e}")
        return authors, author_to_id, all_examples
    
    def _group_examples_by_author(self) -> Dict[int, List[Dict]]:
        """Group examples by author ID for balanced sampling."""
        examples_by_author = {}
        for author_id in range(len(self.authors)):
            examples_by_author[author_id] = []
        
        for example in self.all_examples:
            author_id = example['author_id']
            examples_by_author[author_id].append(example)
            
        return examples_by_author
    
    def _setup_author_weights(self):
        """Setup author sampling weights."""
        author_weights = getattr(self.args, 'author_weights', None)
        balanced_sampling = getattr(self.args, 'balanced_sampling', False)
        # Priority: user-provided weights override balanced sampling
        if author_weights is not None:
            # User-provided weights take precedence
            self.author_weights = np.array([
                author_weights.get(author, 1.0) 
                for author in self.authors
            ])
            logger.info("Using user-provided author weights")
        elif balanced_sampling:
            # Compute inverse frequency weights for balanced sampling
            author_counts = np.array([len(self.examples_by_author[i]) for i in range(len(self.authors))])
            # Inverse weights: authors with fewer examples get higher weights
            self.author_weights = 1.0 / (author_counts + 1e-8)  # Add small epsilon to avoid division by zero
            logger.info("Using balanced sampling with inverse frequency weights:")
            for i, (author, count) in enumerate(zip(self.authors, author_counts)):
                logger.info(f"  {author}: {count} examples, weight: {self.author_weights[i]:.4f}")
        else:
            # Uniform sampling
            self.author_weights = np.ones(len(self.authors))
            logger.info("Using uniform author sampling")
        
        # Normalize weights
        self.author_weights = self.author_weights / self.author_weights.sum()
        
    
    def _tokenize_qa_pair(self, question: str, answer: str, author_id: int) -> Dict[str, torch.Tensor]:
        """Tokenize a Q&A pair and create masks."""
        # Tokenize question and answer separately to track boundaries
        # The tokenizer.encode method handles BOS/EOS tokens automatically
        question_tokens = self.tokenizer.encode(question, add_bos=self.args.add_bos, add_eos=False)
        answer_tokens = self.tokenizer.encode(answer, add_bos=False, add_eos=self.args.add_eos)
        
        # Combine tokens
        all_tokens = question_tokens + answer_tokens
        seq_len = len(all_tokens)
        
        # Truncate if too long
        if seq_len > self.args.seq_len:
            # Truncate the answer to fit
            question_len = len(question_tokens)
            max_answer_len = self.args.seq_len - question_len
            if max_answer_len <= 0:
                # Question is too long, truncate question too
                question_tokens = question_tokens[:self.args.seq_len//2]
                answer_tokens = answer_tokens[:self.args.seq_len - len(question_tokens)]
            else:
                answer_tokens = answer_tokens[:max_answer_len]
            all_tokens = question_tokens + answer_tokens
            seq_len = len(all_tokens)
        
        # Create tensors
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        
        # Labels: shift input_ids by 1 for causal LM, mask question tokens to -100
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = -100  # Mask last token (no next token to predict)
        
        # Mask question tokens in labels (don't compute loss on them)
        question_len = len(question_tokens)
        labels[:question_len-1] = -100  # Keep the last question token to predict first answer token
        
        # Create Q&A mask: 1 for answer tokens, 0 for question tokens
        qa_mask = torch.zeros_like(input_ids)
        qa_mask[question_len:] = 1
        
        # Author tensor (same author for entire sequence)
        authors = torch.full_like(input_ids, author_id)
        
        # Attention mask (all 1s for now, will pad later)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'authors': authors,
            'qa_mask': qa_mask,
            'attention_mask': attention_mask,
            'seq_len': seq_len
        }
    
    def _pad_batch(self, examples: List[Dict[str, torch.Tensor]]) -> SFTBatch:
        """Pad examples to create a batch."""
        batch_size = len(examples)
        max_len = self.args.seq_len
        
        # Initialize tensors
        # Use EOS token as padding (common practice for Llama)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.tokenizer.eos_id)
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        authors = torch.zeros((batch_size, max_len), dtype=torch.long)
        qa_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Fill in the data
        for i, example in enumerate(examples):
            seq_len = example['seq_len']
            input_ids[i, :seq_len] = example['input_ids']
            labels[i, :seq_len] = example['labels']
            authors[i, :seq_len] = example['authors']
            qa_mask[i, :seq_len] = example['qa_mask']
            attention_mask[i, :seq_len] = example['attention_mask']
        
        return SFTBatch(
            input_ids=input_ids,
            labels=labels,
            authors=authors,
            qa_mask=qa_mask,
            attention_mask=attention_mask
        )
    
    def __iter__(self) -> Iterator[SFTBatch]:
        """Iterate over batches."""
        while True:
            batch_examples = []
            shuffle = getattr(self.args, 'shuffle', True)
            
            for i in range(self.args.batch_size):
                if shuffle:
                    # Sample author first using author weights, then sample example from that author
                    author_id = self.rng.choice(len(self.authors), p=self.author_weights)
                    author_examples = self.examples_by_author[author_id]
                    if len(author_examples) > 0:
                        example_idx = self.rng.choice(len(author_examples))
                        example = author_examples[example_idx]
                    else:
                        # Fallback to global sampling if author has no examples
                        idx = self.rng.choice(len(self.all_examples))
                        example = self.all_examples[idx]
                else:
                    # Deterministic iteration (for eval)
                    idx = i % len(self.all_examples)
                    example = self.all_examples[idx]
                
                # Tokenize the Q&A pair
                tokenized = self._tokenize_qa_pair(
                    example['question'], 
                    example['answer'],
                    example['author_id']
                )
                
                batch_examples.append(tokenized)
            
            # Create padded batch
            batch = self._pad_batch(batch_examples)
            yield batch


def build_sft_dataloader(args, rank: int = 0, world_size: int = 1) -> SFTDataLoader:
    """Build SFT data loader."""
    return SFTDataLoader(args, rank, world_size)