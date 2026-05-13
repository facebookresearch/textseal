# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Post-hoc watermarking through LLM rephrasing."""

from textseal.watermarking.watermarker import PostHocWatermarker
from textseal.watermarking.config import (
    ModelConfig,
    ProcessingConfig,
    EvaluationConfig,
    PromptConfig,
)
from textseal.watermarking.detector import WmDetector
from textseal.watermarking.evaluation import WatermarkEvaluator

__all__ = [
    "PostHocWatermarker",
    "ModelConfig",
    "ProcessingConfig",
    "EvaluationConfig",
    "PromptConfig",
    "WmDetector",
    "WatermarkEvaluator",
]
