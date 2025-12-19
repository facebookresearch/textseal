# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Main watermarker module for post-hoc watermarking.

This module contains the refactored PostHocWatermarker class that uses
the modular components for text processing, evaluation, and chunking.
"""

import time
# warnings.filterwarnings('ignore')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from apps.common.watermark.core import WatermarkConfig
from apps.posthoc.config import ModelConfig, ProcessingConfig, EvaluationConfig, PromptConfig
from apps.posthoc.chunking import DocumentChunker
from apps.posthoc.evaluation import WatermarkEvaluator
from apps.posthoc.text_processing import TextProcessor
from apps.posthoc.generator import build_generator
from apps.posthoc.detector import build_detector


def load_tokenizer(model_config: ModelConfig):
    """
    Load tokenizer using HuggingFace AutoTokenizer.
    Args:
        model_config: Model configuration containing model name and cache directory
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name, 
        trust_remote_code=True,
        cache_dir=model_config.cache_dir
    )
    print("✓ Tokenizer loaded successfully")
    return tokenizer


def load_model(model_config: ModelConfig):
    """
    Load model using HuggingFace AutoModelForCausalLM.
    Args:
        model_config: Model configuration containing model settings
    Returns:
        Loaded model
    """
    # Prepare model loading kwargs
    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "cache_dir": model_config.cache_dir
    }
    
    # Add Flash Attention if requested and available
    if model_config.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, **model_kwargs)
    print(f"✓ Model {model_config.model_name} loaded successfully, with mesh:\n{model.hf_device_map}")
    
    # Compile the model for faster inference if requested
    if model_config.compile_model:
        try:
            model = torch.compile(model, mode="default", dynamic=True)
            print("✓ Model compilation completed")
        except Exception as e:
            print(f"Model compilation failed, continuing without compilation: {e}")
    
    return model


class PostHocWatermarker:
    """
    post-hoc watermarking system that performs post-hoc watermarking by:
    1. Taking an original document as input
    2. Rephrasing it using a large language model
    3. Applying watermarking techniques during the rephrasing process
    4. Evaluating the quality and detectability of the watermark
    """
    
    def __init__(
        self,
        watermark_config: WatermarkConfig = None,
        model_config: ModelConfig = None,
        processing_config: ProcessingConfig = None,
        evaluation_config: EvaluationConfig = None,
        prompt_config: PromptConfig = None
    ):
        """
        Args:
            watermark_config: Watermark-specific configuration
            model_config: Model loading and optimization configuration
            processing_config: Text processing configuration
            evaluation_config: Evaluation configuration
            prompt_config: Prompt and instruction configuration
        """
        self.watermark_config = watermark_config or WatermarkConfig()
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or ProcessingConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.prompt_config = prompt_config or PromptConfig()
        
        # Load tokenizer and model
        print(f"Loading tokenizer and model: {self.model_config.model_name}")
        self.tokenizer = load_tokenizer(self.model_config)
        self.model = None
        
        # Determine if we need to load the model
        # Need model if: (1) not detection-only mode, OR (2) detection-only but using entropy thresholds
        has_entropy_thresholds = False
        if self.evaluation_config.test_entropy_thresholds is not None:
            if isinstance(self.evaluation_config.test_entropy_thresholds, str):
                stripped = self.evaluation_config.test_entropy_thresholds.strip()
                has_entropy_thresholds = stripped != "" and stripped.lower() != "none"
            elif isinstance(self.evaluation_config.test_entropy_thresholds, list):
                has_entropy_thresholds = len(self.evaluation_config.test_entropy_thresholds) > 0
        
        needs_model = (not self.evaluation_config.enable_detection_only) or \
                     (self.evaluation_config.enable_detection_only and has_entropy_thresholds)
        
        if needs_model:
            # Initialize model
            self.model = load_model(self.model_config)
        
        if not self.evaluation_config.enable_detection_only:
            # Initialize text modules
            self.text_processor = TextProcessor(self.tokenizer, self.model_config.model_name, self.prompt_config)
            self.chunker = DocumentChunker(
                target_size=self.processing_config.target_chunk_size,
                overlap_ratio=self.processing_config.overlap_ratio,
            )
            # Initialize watermark generator
            self.generator = build_generator(
                model=self.model,
                tokenizer=self.tokenizer,
                wm_args=self.watermark_config,
                beam_width=self.processing_config.beam_width,
                candidates_per_beam=self.processing_config.candidates_per_beam,
                stochastic_beam=self.processing_config.stochastic_beam,
                use_biased_for_scoring=self.processing_config.use_biased_for_scoring,
            )
        # Initialize watermark detector
        self.detector = build_detector(self.tokenizer, self.watermark_config, self.model)
        
        # Load quality evaluation model if perplexity is enabled and it's different from watermarking model
        quality_model = None
        quality_tokenizer = None
        if self.evaluation_config.enable_perplexity:
            if self.evaluation_config.quality_model_name != self.model_config.model_name:
                print(f"Loading separate quality model for perplexity evaluation: {self.evaluation_config.quality_model_name}")
                quality_model_config = ModelConfig()
                quality_model_config.model_name = self.evaluation_config.quality_model_name
                quality_model_config.cache_dir = self.model_config.cache_dir
                quality_model_config.use_flash_attention = False  # Not critical for evaluation
                quality_model_config.compile_model = False
                quality_tokenizer = load_tokenizer(quality_model_config)
                quality_model = load_model(quality_model_config)
            else:
                print(f"Using same model for perplexity evaluation as watermarking model")
                quality_model = self.model
                quality_tokenizer = self.tokenizer
        
        # Initialize evaluator with quality model
        self.evaluator = WatermarkEvaluator(
            self.evaluation_config,
            quality_model=quality_model,
            quality_tokenizer=quality_tokenizer
        )
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            return 0
    
    def rephrase_with_watermark(
        self,
        text: str,
        max_gen_len: int = None,
        temperature: float = None,
        top_p: float = None,
        context_chunks: list = None
    ) -> str:
        """
        Rephrase text using watermarked generation.
        
        Args:
            text: Original text to rephrase
            max_gen_len: Maximum generation length (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            context_chunks: Optional list of previously rephrased chunks for context
        """        
        # Create rephrasing prompt with optional context.
        prompt = self.text_processor.create_rephrasing_prompt(text, context_chunks=context_chunks)
        # Configure generation parameters.
        max_gen_len = max_gen_len or self.processing_config.max_gen_len
        generation_limit = min(max_gen_len, len(text) + 500)
        # Generate watermarked text.
        watermarked_text = self.generator.generate(
            prompts = [prompt],
            max_gen_len = generation_limit,
            temperature = temperature or self.processing_config.temperature,
            top_p = top_p or self.processing_config.top_p,
        )[0]
        # Clean up the response by removing assistant prefixes and extra text.
        watermarked_text = self.text_processor.clean_generated_text(watermarked_text)
        # Optional code post-processing
        if self.evaluation_config.enable_code_evaluation:
            watermarked_text = self.text_processor.post_process_code(watermarked_text)
        return watermarked_text

    def process_text(
        self,
        original_text: str,
        max_gen_len: int = None,
        temperature: float = None,
        top_p: float = None,
        aux_data: dict = None,
        context_chunks: list = None,
    ) -> dict:
        """
        Process a document with post-hoc watermarking and evaluation.
        
        Args:
            original_text: Original text to watermark
            output_dir: Directory to save results
            max_gen_len: Maximum generation length (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            aux_data: Optional auxiliary data to use in results
            context_chunks: Optional list of previously rephrased chunks for context
            
        Returns:
            Dictionary with all results
        """
        start_time = time.time()
        # Step 1: Rephrase with watermark
        watermarked_text = self.rephrase_with_watermark(
            original_text,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            context_chunks=context_chunks
        )
        t1 = time.time()
        # Step 2: Evaluate watermark
        watermark_eval = self.evaluator.evaluate_watermark(
            watermarked_text,
            self.detector,
            self.watermark_config.watermark_type,
            self.watermark_config.scoring_method,
            entropy_threshold=self.evaluation_config.entropy_threshold,
            tokenizer=self.tokenizer,
            wm_config=self.watermark_config,
            model=self.model
        )
        t2 = time.time()
        # Step 3: Evaluate quality
        quality_eval = self.evaluator.evaluate_quality(
            original_text, 
            watermarked_text
        )
        t3 = time.time()

        if self.evaluation_config.enable_code_evaluation:
            test = aux_data["test"]
            orig_passed = self.evaluator.evaluate_code(original_text, test)
            wm_passed = self.evaluator.evaluate_code(watermarked_text, test)
            quality_eval["code_eval"] = {
                "orig_passed": orig_passed,
                "wm_passed": wm_passed,
            }

        # Timings
        time_1 = t1 - start_time
        time_2 = t2 - t1
        time_3 = t3 - t2
        total_time = t3 - start_time
        # Compute lightweight stats (uniform naming)
        orig_toks = self.count_tokens(original_text)
        wm_toks = self.count_tokens(watermarked_text)

        # Compile results (uniform, short keys)
        results = {
            "wm_text": watermarked_text,
            "orig_text": original_text,
            "wm_eval": watermark_eval,
            "quality": quality_eval,
            "times": {
                "t_rephrase": float(time_1),
                "tps": float(wm_toks / time_1) if time_1 > 0 else 0.0,
                "t_wm_eval": float(time_2),
                "t_quality": float(time_3),
                "t_total": float(total_time),
            },
            "stats": {
                "orig_len": len(original_text),
                "wm_len": len(watermarked_text),
                "orig_toks": orig_toks,
                "wm_toks": wm_toks,
                "tok_ratio": float(wm_toks / orig_toks) if orig_toks > 0 else 0.0,
            },
        }
        return results

    def process_large_text(
        self,
        original_text: str,
        max_gen_len: int = None,
        temperature: float = None,
        top_p: float = None,
        aux_data: dict = None,
        verbose: bool = False,
    ) -> dict:
        """
        Simpler large-text processing: split into chunks, reuse process_text per chunk,
        and aggregate chunk-wise and global metrics.
        """
        # Resolve defaults
        max_gen_len = max_gen_len or 512  # smaller per chunk by default
        temperature = temperature or self.processing_config.temperature
        top_p = top_p or self.processing_config.top_p

        start_time = time.time()

        # Chunk the document
        chunks = self.chunker.create_smart_chunks(original_text)
        num_chunks = len(chunks)
        watermarked_chunks = []
        chunk_results = []

        for i, chunk_info in enumerate(chunks):
            chunk_text = chunk_info['text']
            if verbose:
                print(f"Chunk {i+1}/{num_chunks}: {len(chunk_text)} chars (/{len(original_text)})")

            # Prepare context chunks if enabled
            context_chunks = None
            if self.processing_config.use_context_chunks and i > 0:
                # Get previous N chunks as context
                start_idx = max(0, i - self.processing_config.num_context_chunks)
                context_chunks = watermarked_chunks[start_idx:i]
                
                # Truncate context to max_context_tokens if needed
                total_context_tokens = sum(self.count_tokens(c) for c in context_chunks)
                while total_context_tokens > self.processing_config.max_context_tokens and len(context_chunks) > 1:
                    context_chunks.pop(0)  # Remove oldest chunk
                    total_context_tokens = sum(self.count_tokens(c) for c in context_chunks)

            # Reuse single-text pipeline for each chunk with context
            res = self.process_text(
                chunk_text,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                context_chunks=context_chunks,
            )

            wm_text = res.get('wm_text', '')
            watermarked_chunks.append(wm_text)

            # Collect concise per-chunk metrics
            chunk_results.append({
                'idx': i,
                'orig_len': len(chunk_text),
                'wm_len': len(wm_text),
                'orig_toks': self.count_tokens(chunk_text),
                'wm_toks': self.count_tokens(wm_text),
                'wm_eval': res.get('wm_eval'),
                'quality': res.get('quality'),
                'times': res.get('times'),
            })

        # Merge chunks and compute global metrics
        watermarked_text = self.chunker.intelligent_merge(watermarked_chunks, chunks)

        watermark_eval = self.evaluator.evaluate_watermark(
            watermarked_text,
            self.detector,
            self.watermark_config.watermark_type,
            self.watermark_config.scoring_method,
            entropy_threshold=self.evaluation_config.entropy_threshold
        )
        quality_eval = self.evaluator.evaluate_quality(
            original_text, 
            watermarked_text
        )
        if self.evaluation_config.enable_code_evaluation:
            test = aux_data["test"]
            orig_passed = self.evaluator.evaluate_code(original_text, test)
            wm_passed = self.evaluator.evaluate_code(watermarked_text, test)
            quality_eval["code_eval"] = {
                "orig_passed": orig_passed,
                "wm_passed": wm_passed,
            }
            
        total_time = time.time() - start_time
        total_original_tokens = self.count_tokens(original_text)
        total_watermarked_tokens = self.count_tokens(watermarked_text)

        results = {
            'wm_text': watermarked_text,
            'orig_text': original_text,
            'wm_eval': watermark_eval,
            'quality': quality_eval,
            'chunk_results': chunk_results,
            'stats': {
                'orig_len': len(original_text),
                'wm_len': len(watermarked_text),
                'orig_toks': total_original_tokens,
                'wm_toks': total_watermarked_tokens,
                'tok_ratio': float(total_watermarked_tokens / total_original_tokens) if total_original_tokens > 0 else 0.0,
            },
            'times': {
                't_total': float(total_time),
                'tps': float(total_watermarked_tokens / total_time) if total_time > 0 else 0.0,
            },
        }
        return results
