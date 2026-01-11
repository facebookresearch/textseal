# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
TextSeal: Comprehensive toolkit for LLM watermarking and contamination detection.

Basic usage:
    >>> from textseal import PostHocWatermarker, WatermarkConfig
    >>> watermarker = PostHocWatermarker()
    >>> result = watermarker.process_text("Your text here")
"""

from textseal.posthoc import PostHocWatermarker
from textseal.posthoc.config import (
    ModelConfig,
    ProcessingConfig,
    EvaluationConfig,
    PromptConfig,
)
from textseal.common.watermark import WatermarkConfig

__version__ = "0.0.4"

__all__ = [
    "PostHocWatermarker",
    "WatermarkConfig",
    "ModelConfig",
    "ProcessingConfig",
    "EvaluationConfig",
    "PromptConfig",
]
