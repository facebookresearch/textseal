# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Configuration dataclasses for post-hoc watermarking system.
"""

from dataclasses import dataclass, field
from typing import Optional


DUAL_KEY_OFFSET = 12345


MODEL_NAMES = [
    # HF
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    # Gemma-3
    "google/gemma-3-4b-it",
    "google/gemma-3-27b-it",
    # Llama 3
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    # Qwen3
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    # Reasoning Models 
    # DeepSeek-R1 distilled
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # QwenQ models
    "Qwen/QwQ-32B-Preview",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # Qwen 3.5
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    # Mistral reasoning models
    "mistralai/Ministral-3-14B-Reasoning-2512",
    "mistralai/Ministral-3-3B-Reasoning-2512",
    # Microsoft reasoning models
    "microsoft/Phi-4-reasoning-plus",
    # OpenAI GPT-OSS (open-weight reasoning models with harmony format)
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


@dataclass
class WatermarkConfig:
    """Unified watermark configuration for both training and post-hoc watermarking."""

    # Core watermark parameters
    secret_key: int = 42
    ngram: int = 1

    # Watermark type and method
    watermark_type: str = "gumbelmax"  # "greenlist", "gumbelmax", "dipmark", "synthid", "watermax", "textseal", "none"
    method: str = "uniform"  # "binary", "uniform": the pseudorandom function method

    # Generation-specific parameters
    gamma: float = 0.5  # greenlist fraction
    delta: float = 2.0  # strength parameter in Green-list/Red-list
    alpha: float = 0.2  # interval parameter in DiPMark
    depth: int = 30  # number of tournaments in SynthID
    after_topp: bool = True  # For greenlist: apply bias after top-p filtering if True
    k_morphmark: float = 1.30  # For MorphMark
    p_0: float = 0.15  # For MorphMark

    # WaterMax-specific parameters
    chunk_size: int = 4  # L: number of tokens per chunk in WaterMax
    num_drafts: int = 4  # m: number of draft sequences to generate per chunk
    base_watermark: str = "greenlist"  # Base watermark for WaterMax scoring ("greenlist", "gumbelmax", "synthid")

    # Training-specific parameters
    attenuation: str = "interpolate"
    attenuation_weight: float = 0.0
    backprop_on_nonzero: bool = False

    # Detection parameters
    scoring_method: str = "v2"  # "v1" (dedup by window), "v2" (dedup by window+target), "none" or others (no dedup)

    # TextSeal-specific parameters
    secret_key_b: int = -1  # Key B (-1 = auto: secret_key + DUAL_KEY_OFFSET)
    mixing_alpha: float = 0.5  # Probability of using Key A per token (0.5 = equal)

    @property
    def key_a(self) -> int:
        return self.secret_key

    @property
    def key_b(self) -> int:
        if self.secret_key_b >= 0:
            return self.secret_key_b
        return self.secret_key + DUAL_KEY_OFFSET


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  # HuggingFace model name
    use_flash_attention: bool = False  # use Flash Attention 2
    compile_model: bool = False  # use torch.compile
    cache_dir: Optional[str] = None  # HuggingFace cache dir


@dataclass
class ReasoningConfig:
    """Configuration for reasoning mode (model thinks before answering)."""
    enabled: bool = False  # Enable reasoning mode (uses enable_thinking in chat template)
    start_token: str = "<think>"  # Token marking start of reasoning trace
    end_token: str = "</think>"  # Token marking end of reasoning trace (for splitting output)
    max_tokens: int = 0  # Max reasoning tokens before forcing end_token (0 = unlimited)


@dataclass
class ProcessingConfig:
    max_gen_len: int = 1024  # max generation length
    temperature: float = 0.9  # sampling temperature
    top_p: float = 0.95  # top-p sampling
    generation_mode: bool = False  # False = rephrase mode, True = generation mode (use text as prompt)
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
    
    # Reasoning
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)


@dataclass
class PromptConfig:
    system_message: str = ""  # Custom system message override (auto-generated per mode if empty)
    prefill_answer: str = ""  # prefill answer in the prompt
    preserve_style: bool = True  # emphasize style preservation
    preserve_length: bool = False  # preserve approximate length
    preserve_format: bool = True  # preserve formatting


# Predefined attack prompts by strength level
ATTACK_PROMPTS = {
    "mild": {
        "system": (
            "You are a careful text editor. "
            "Make minimal changes to the text - only fix grammar, punctuation, or word choice issues. "
            "Preserve the original wording as much as possible. "
            "Output only the edited text, with no explanations."
        ),
        "user": "Please lightly edit the following text, making minimal changes:\n\n{text}",
        "temperature": 0.5,
    },
    "moderate": {
        "system": (
            "You are a text rephrasing assistant. "
            "Rephrase the given text while preserving its meaning and style. "
            "Use different words and sentence structures where appropriate. "
            "Output only the rephrased text, with no explanations."
        ),
        "user": "Please rephrase the following text:\n\n{text}",
        "temperature": 0.8,
    },
    "aggressive": {
        "system": (
            "You are an expert paraphrasing tool. "
            "Completely rewrite the text using entirely different words, phrases, and sentence structures. "
            "Change the writing style while keeping the core meaning. "
            "Be creative with your rewording. "
            "Output only the rewritten text, with no explanations."
        ),
        "user": "Completely rewrite the following text using different words and structure:\n\n{text}",
        "temperature": 1.0,
    },
    "extreme": {
        "system": (
            "You are a heavy-duty text transformation tool. "
            "Aggressively rewrite the text - change everything possible while retaining only the essential meaning. "
            "Use completely different vocabulary, restructure all sentences, and alter the tone. "
            "The result should look nothing like the original. "
            "Output only the transformed text, with no explanations."
        ),
        "user": "Heavily transform the following text, making it completely different while keeping the core meaning:\n\n{text}",
        "temperature": 1.2,
    },
}


@dataclass
class AttackConfig:
    """Configuration for attack simulation on watermarked text."""
    enable_attack: bool = False  # Enable rephrasing attack simulation
    attack_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"  # Model to use for attacks
    attack_strengths: Optional[str] = None  # Comma-separated strengths to run: "mild,moderate,aggressive,extreme" or "all" (default: all if attacks enabled)
    attack_strength: str = "moderate"  # Single attack strength (used if attack_strengths not set)
    attack_temperature: Optional[float] = None  # Temperature override (uses strength default if None)
    attack_top_p: float = 0.95  # Top-p for attack rephrasing
    attack_max_gen_len: int = 1024  # Max generation length for attack
    attack_system_message: Optional[str] = None  # Custom system message (overrides strength preset)
    attack_user_message_template: Optional[str] = None  # Custom user template (overrides strength preset)
    
    def get_attack_strengths_list(self) -> list:
        """Get list of attack strengths to run."""
        if self.attack_strengths is None:
            # Default: run all if attacks enabled, otherwise just the single strength
            return ["mild", "moderate", "aggressive", "extreme"]
        if self.attack_strengths.lower() == "all":
            return ["mild", "moderate", "aggressive", "extreme"]
        # Handle empty or whitespace-only string explicitly to avoid returning ['']
        if self.attack_strengths.strip() == "":
            return []
        # Split on commas, strip whitespace, and drop any empty entries
        return [s.strip() for s in self.attack_strengths.split(",") if s.strip()]
    
    def _validate_strength(self, strength: str) -> None:
        """Validate that the attack strength is valid."""
        if strength not in ATTACK_PROMPTS:
            valid_strengths = list(ATTACK_PROMPTS.keys())
            raise ValueError(f"Invalid attack strength '{strength}'. Must be one of {valid_strengths}")
    
    def get_system_message(self, strength: Optional[str] = None) -> str:
        """Get the system message based on strength or custom setting."""
        if self.attack_system_message is not None:
            return self.attack_system_message
        strength = strength or self.attack_strength
        self._validate_strength(strength)
        return ATTACK_PROMPTS[strength]["system"]
    
    def get_user_template(self, strength: Optional[str] = None) -> str:
        """Get the user message template based on strength or custom setting."""
        if self.attack_user_message_template is not None:
            return self.attack_user_message_template
        strength = strength or self.attack_strength
        self._validate_strength(strength)
        return ATTACK_PROMPTS[strength]["user"]
    
    def get_temperature(self, strength: Optional[str] = None) -> float:
        """Get the temperature based on strength or custom setting."""
        if self.attack_temperature is not None:
            return self.attack_temperature
        strength = strength or self.attack_strength
        self._validate_strength(strength)
        return ATTACK_PROMPTS[strength]["temperature"]


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
    evaluate_attack: bool = False  # Also evaluate watermark detection on attacked (rephrased) text
    
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
