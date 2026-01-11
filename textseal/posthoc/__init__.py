# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Post-hoc watermarking through LLM rephrasing."""

from textseal.posthoc.watermarker import PostHocWatermarker
from textseal.posthoc.config import (
    ModelConfig,
    ProcessingConfig,
    EvaluationConfig,
    PromptConfig,
)
from textseal.posthoc.detector import WmDetector
from textseal.posthoc.evaluation import WatermarkEvaluator

__all__ = [
    "PostHocWatermarker",
    "ModelConfig",
    "ProcessingConfig",
    "EvaluationConfig",
    "PromptConfig",
    "WmDetector",
    "WatermarkEvaluator",
]
