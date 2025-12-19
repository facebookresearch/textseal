# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from apps.wmtraining.lingua.transformer import cross_entropy
from apps.common.watermark.core import WatermarkArgs


# Note: Watermark training with loss modification has been removed from this codebase.
# For watermarking capabilities, please use the post-hoc watermarking system in apps/posthoc/.
# This module now provides standard cross-entropy loss functions for training.


def wm_cross_entropy(
    logits: torch.Tensor,
    token_values: torch.Tensor,
    targets: torch.Tensor,
    sources: torch.Tensor,
    watermark: WatermarkArgs,
    **kwargs
) -> torch.Tensor:
    """
    Computes standard cross-entropy loss without watermarking.
    
    Note: Watermark training with loss modification has been removed.
    For watermarking, use post-hoc methods (see apps/posthoc/).

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size).
        token_values (torch.Tensor): Token values of shape (batch_size, seq_len).
        targets (torch.Tensor): Target token indices of shape (batch_size, seq_len).
        sources (torch.Tensor): Source identifiers (unused, kept for API compatibility).
        watermark (WatermarkArgs): Watermarking arguments (unused, kept for API compatibility).
        **kwargs: Additional keyword arguments for torch.nn.functional.nll_loss.

    Returns:
        torch.Tensor: Scalar tensor representing the cross-entropy loss.
    """
    # Always use standard cross-entropy loss
    return cross_entropy(logits, targets)


def sft_wm_cross_entropy(
    logits: torch.Tensor,
    token_values: torch.Tensor,
    targets: torch.Tensor,
    authors: torch.Tensor,
    qa_mask: torch.Tensor,
    watermark: WatermarkArgs,
    **kwargs
) -> torch.Tensor:
    """
    Computes standard SFT cross-entropy loss without watermarking.
    
    Note: Watermark training with loss modification has been removed.
    For watermarking, use post-hoc methods (see apps/posthoc/).

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size).
        token_values (torch.Tensor): Token values of shape (batch_size, seq_len).
        targets (torch.Tensor): Target token indices of shape (batch_size, seq_len).
        authors (torch.Tensor): Author identifiers (unused, kept for API compatibility).
        qa_mask (torch.Tensor): Q&A mask (unused, kept for API compatibility).
        watermark (WatermarkArgs): Watermarking arguments (unused, kept for API compatibility).
        **kwargs: Additional keyword arguments for torch.nn.functional.nll_loss.

    Returns:
        torch.Tensor: Scalar tensor representing the cross-entropy loss.
    """
    # Always use standard SFT loss (respects -100 masking in targets)
    return cross_entropy(logits, targets)


