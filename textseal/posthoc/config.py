# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Configuration dataclasses for post-hoc watermarking system.
"""

from dataclasses import dataclass
from typing import Optional

MODEL_NAMES = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"  # HuggingFace model name
    use_flash_attention: bool = False  # use Flash Attention 2
    compile_model: bool = False  # use torch.compile
    cache_dir: Optional[str] = None  # HuggingFace cache dir
    
    assert model_name in MODEL_NAMES, "Model name not in supported MODEL_NAMES."


@dataclass
class ProcessingConfig:
    max_gen_len: int = 1024  # max generation length
    temperature: float = 0.8  # sampling temperature
    top_p: float = 0.95  # top-p sampling
    target_chunk_size: int = 2000  # target chunk size (chars)
    max_chunk_size: int = 1024  # max chunk size (tokens)
    overlap_ratio: float = 0.15  # chunk overlap ratio (0.0-1.0)
    
    # Context-aware chunking parameters
    use_context_chunks: bool = False  # Include previous chunks as context for coherence
    num_context_chunks: int = 4  # Number of previous chunks to include as context
    max_context_tokens: int = 1000  # Maximum tokens for context from previous chunks
    
    # Beam search parameters (None = standard sampling)
    beam_width: Optional[int] = None  # Number of beams (None or 1 = standard sampling)
    candidates_per_beam: Optional[int] = None  # Candidates per beam (default: beam_width)
    stochastic_beam: bool = False  # Stochastic (True) vs deterministic (False) beam search
    use_biased_for_scoring: bool = False  # Score with watermarked (True) vs original (False) model


@dataclass
class PromptConfig:
    system_message: str = (
        "You are a text rephrasing assistant. "
        "You must rephrase the given text while strictly preserving its original meaning, style, and structure. "
        "You must output only the rephrased text, with no explanations or commentary. "
    )  # for rephrasing
    user_message_template: str = "Please rephrase the following text:\n\n{text}"  # use {text} placeholder
    prefill_answer: str = ""  # prefill answer in the prompt
    custom_instruction: str = ""  # optional extra instruction
    preserve_style: bool = True  # emphasize style preservation
    preserve_length: bool = False  # preserve approximate length
    preserve_format: bool = True  # preserve formatting


@dataclass
class EvaluationConfig:
    enable_semantic_similarity: bool = True  # compute semantic similarity
    enable_rouge: bool = True  # compute ROUGE
    enable_bleu: bool = True  # compute BLEU
    enable_perplexity: bool = False  # compute perplexity with separate quality model
    quality_model_name: str = "mistralai/Mistral-7B-v0.3"  # Base model for perplexity evaluation (use base, not instruct; must be different from watermarking models)
    detection_threshold: float = 1e-3  # p-value threshold
    enable_code_evaluation: bool = False  # evaluate functional correctness for code
    entropy_threshold: Optional[float] = None  # if set, only score tokens with entropy < threshold (backward compat)
    enable_detection_only: bool = False  # only do watermark detection
    
    # Multi-test support: Run multiple statistical tests per rephrasing
    # Can be comma-separated strings (e.g., "none,1.5,2.0") or lists
    test_entropy_thresholds: Optional[str] = None  # Comma-separated entropy thresholds (e.g., "none,1.5,2.0,2.5")
    test_watermark_types: Optional[str] = None  # Comma-separated watermark types (e.g., "synthid,synthid-weighted")
    
    def __post_init__(self):
        """Parse comma-separated strings into lists."""
        # Parse entropy thresholds
        if isinstance(self.test_entropy_thresholds, str):
            parts = [p.strip() for p in self.test_entropy_thresholds.split(',')]
            self.test_entropy_thresholds = [
                None if p.lower() in ['none', 'null', ''] else float(p) 
                for p in parts
            ]
        
        # Parse watermark types
        if isinstance(self.test_watermark_types, str):
            self.test_watermark_types = [
                p.strip() for p in self.test_watermark_types.split(',')
            ]
