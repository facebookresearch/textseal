# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core watermark functions shared across apps.
These are the fundamental mathematical operations for watermarking.
"""

from dataclasses import dataclass
import torch
from typing import Optional, Literal
from .pseudorandom import prf_binary, prf_uniform

@dataclass
class WatermarkConfig:
    """Unified watermark configuration for both training and post-hoc watermarking."""
    # Core watermark parameters
    secret_key: int = 42
    ngram: int = 1
    gamma: float = 0.5  # greenlist fraction
    
    # Watermark type and method
    watermark_type: str = "greenlist"  # "greenlist" (or "maryland"), "gumbelmax" (or "openai"), "dipmark", "synthid", "watermax", "none"
    method: str = "binary"  # "binary", "uniform": the pseudorandom function method
    
    # Generation-specific parameters
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
    
    def __post_init__(self):
        """Normalize watermark_type to support both old and new naming."""
        # Map old names to new names for backward compatibility
        type_aliases = {
            "maryland": "greenlist",
            "openai": "gumbelmax",
        }
        if self.watermark_type in type_aliases:
            self.watermark_type = type_aliases[self.watermark_type]
        
        # Also normalize base_watermark for WaterMax
        if self.base_watermark in type_aliases:
            self.base_watermark = type_aliases[self.base_watermark]

WatermarkArgs = WatermarkConfig # backward compatibility alias


def _load_filter_cache(filter_path: str, filter_keys: str) -> tuple[list[str], dict]:
    """Load and cache filter data to avoid repeated file I/O.
    
    filter_path: Path to the filter file. should be a JSON file with keys mapping to frequency dicts.
    filter_keys: Comma-separated keys to load from the filter file.
    
    """
    global _filter_cache
    if '_filter_cache' not in globals():
        _filter_cache = {}
    
    cache_key = (filter_path, filter_keys)
    if cache_key not in _filter_cache:
        import json
        import ast
        with open(filter_path, "r") as f:
            filter_dict = json.load(f)
        filter_key_list = filter_keys.split(",") # Expected to be comma-separated
        filter_freqs = {}
        for key in filter_key_list:
            if key in filter_dict:
                filter_freqs[key] = {ast.literal_eval(k): v for k, v in filter_dict[key].items()}
            else:
                filter_freqs[key] = {}
        _filter_cache[cache_key] = (filter_key_list, filter_freqs)
    
    return _filter_cache[cache_key]


def _apply_filter(
    wm_windows: torch.Tensor,
    targets: torch.Tensor,
    sources: torch.Tensor,
    filter_key_list: list[str],
    filter_freqs: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter sequences based on pre-computed frequency data.
    
    Args:
        wm_windows: Tensor of shape (..., ngram) representing the context windows
        targets: Tensor of shape (...) representing the target tokens
        sources: Tensor of shape (...) representing source identifiers
        filter_key_list: List of keys corresponding to filter frequencies
        filter_freqs: Dictionary mapping keys to frequency sets

    Returns:
        tuple: (filtered_windows, filtered_targets, filtered_sources)
    """
    valid_mask = torch.zeros_like(targets, dtype=torch.bool)
    
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            window_tuple = tuple(wm_windows[i][j].tolist())
            target_val = targets[i][j].item()
            pair = window_tuple + (target_val,)
            
            source_idx = sources[i][j].item() if sources is not None else 0
            if source_idx < len(filter_key_list):
                key = filter_key_list[source_idx]
                if key in filter_freqs and pair in filter_freqs[key]:
                    valid_mask[i][j] = True
    
    filtered_windows = wm_windows[valid_mask]
    filtered_targets = targets[valid_mask]
    filtered_sources = sources[valid_mask] if sources is not None else None
    
    return filtered_windows, filtered_targets, filtered_sources


def _get_dedup_key(wm_window: torch.Tensor, target: int, scoring_method: str) -> tuple:
    """Generate deduplication key based on scoring method.
    
    Args:
        wm_window: Tensor of shape (ngram,) representing the context window
        target: Target token id
        scoring_method: "v1" (deduplicate by window only) or "v2" (deduplicate by window+target)
    
    Returns:
        tuple: Key for deduplication
    """
    window_tuple = tuple(wm_window.tolist())
    if scoring_method == "v1":
        return window_tuple
    elif scoring_method == "v2":
        return window_tuple + (target,)
    else:
        # No deduplication for other methods
        return None


def _apply_deduplication(
    wm_windows: torch.Tensor,
    targets: torch.Tensor,
    sources: torch.Tensor,
    wm_args: WatermarkConfig,
    filter_key_list: list[str] = None,
    seen_windows: set = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Apply deduplication based on scoring method.
    
    Args:
        seen_windows: Optional set to track seen windows (for v1 deduplication).
                     If None, a new set will be created (no persistence across calls).
    
    Returns:
        tuple: (filtered_windows, filtered_targets, filtered_sources, dedup_stats)
    """
    # Use provided seen_windows set or create a new one
    if seen_windows is None:
        seen_windows = set()
    
    # Get device from input tensors
    device = wm_windows.device
    valid_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
    total_count = 0
    duplicate_count = 0

    for i in range(len(targets)):
        for j in range(len(targets[i])):
            total_count += 1
            window = wm_windows[i][j]
            target_val = targets[i][j].item()
            
            # Get deduplication key based on scoring method
            dedup_key = _get_dedup_key(window, target_val, wm_args.scoring_method)
            
            if dedup_key is None:
                # No deduplication
                valid_mask[i][j] = True
                continue
            
            # Check if we've seen this key before
            if dedup_key not in seen_windows:
                seen_windows.add(dedup_key)
                valid_mask[i][j] = True
            else:
                duplicate_count += 1

    filtered_windows = wm_windows[valid_mask]
    filtered_targets = targets[valid_mask]
    filtered_sources = sources[valid_mask.to(sources.device)] if sources is not None else None
    
    dedup_stats = {
        'total': total_count,
        'duplicates': duplicate_count,
        'kept': valid_mask.sum().item(),
        'dedup_rate': duplicate_count / total_count if total_count > 0 else 0.0
    }
    
    return filtered_windows, filtered_targets, filtered_sources, dedup_stats


def score_batch(
    wm_windows: torch.Tensor, 
    targets: torch.Tensor, 
    wm_args: WatermarkConfig,
    sources: torch.Tensor = None,
    use_filter: bool = False,
    filter_path: str = "",
    filter_keys: str = "",
    seen_windows: set = None
) -> tuple[list[int], float]:
    """
    Computes the watermark scores for a batch of token sequences.
    
    Args:
        wm_windows (torch.Tensor): Token sequences with shape (..., ngram).
        targets (torch.Tensor): targets tokens with shape (...).
        wm_args (WatermarkConfig): Watermark arguments.
        sources (torch.Tensor, optional): Source identifiers.
        use_filter (bool, optional): Whether to use a filter during scoring.
        filter_path (str, optional): Path to the filter file.
        filter_keys (str, optional): Comma-separated keys for the filter.
        seen_windows (set, optional): Set to track seen windows for v1 deduplication.
                                      If provided, persists across calls for the same source.
    
    Returns:
        tuple: (wm_mask as list of int, agg_score as float)
    """
    
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
        assert wm_windows.dim() == 2, "If targets is 1-D, wm_windows must be 2-D."
        wm_windows = wm_windows.unsqueeze(0)
        if sources is not None:
            assert sources.dim() == 1, "If targets is 1-D, sources must be 1-D."
            sources = sources.unsqueeze(0)

    sk = wm_args.secret_key + sources if sources is not None else wm_args.secret_key

    if use_filter:
        filter_key_list, filter_freqs = _load_filter_cache(filter_path, filter_keys)
        wm_windows, targets, sources = _apply_filter(
            wm_windows, targets, sources, filter_key_list, filter_freqs
        )
        if sources is not None:
            sk = wm_args.secret_key + sources

    if wm_args.scoring_method in ["v1", "v2"]:
        wm_windows, targets, sources, dedup_stats = _apply_deduplication(
            wm_windows, targets, sources, wm_args, filter_key_list if use_filter else None,
            seen_windows=seen_windows
        )
        if sources is not None:
            sk = wm_args.secret_key + sources
    
    if wm_args.method == "binary":
        wm_mask = prf_binary(wm_windows, targets, sk, gamma=wm_args.gamma)
        mean_val = wm_mask.float().mean().item()
        agg_score = 0.0 if torch.isnan(torch.tensor(mean_val)) else mean_val
    elif wm_args.method == "uniform":
        wm_mask = prf_uniform(wm_windows, targets, sk)
        agg_score = -(1 - wm_mask).log().mean().item() 
    elif wm_args.method == "none":
        wm_mask = torch.zeros_like(targets)
        agg_score = 0.0
    else:
        raise ValueError(f"Unknown watermarking method: {wm_args.method}.")
    
    return wm_mask.flatten().tolist(), agg_score


def score_tokens(tokens: list[int] | torch.Tensor, wm_args: WatermarkConfig) -> tuple[list[int], float]:
    """
    Computes the predicted watermark mask for a sequence of token IDs using the pseudorandom functions.
    
    Args:
        tokens (list of int or torch.Tensor): Sequence of token IDs.
        wm_args (WatermarkConfig): Arguments for watermarking.
    
    Returns:
        tuple:
            - wm_mask (list of int): Mask indicating watermarked scores in the sequence.
            - agg_score (float): Proportion of positions predicted as watermarked.
    
    Example:
        >>> tokens = tokenizer.encode("Hello world", add_bos=False, add_eos=False)
        >>> wm_mask, prop_pred_watermarked = score_tokens(tokens, wm_args)
        >>> print(f"Proportion watermarked: {prop_pred_watermarked:.4f}")
    """
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    wm_windows = torch.stack(
        [tokens.roll(i + 1, dims=-1) for i in range(wm_args.ngram)],
        dim=-1,
    ).flip(-1)  # ... x ngram
    targets = tokens
    return score_batch(
        wm_windows,
        targets,
        wm_args,
    )


def score_listed_tokens(
    wm_windows: torch.Tensor,
    wm_args: WatermarkArgs,
    listed_tokens: list[int] | torch.Tensor
) -> torch.Tensor:
    """
    Computes watermark scores for a provided list of next-token ids for each window in the batch.
    Args:
        wm_windows (torch.Tensor): Tensor of shape (batch_size, ngram) representing the watermark window for each example.
        wm_args (WatermarkArgs): Watermark arguments.
        listed_tokens (list[int] or torch.Tensor): 1-D list/tensor of token ids to score.
    Returns:
        torch.Tensor: Watermark scores of shape (batch_size, n_listed).
    TODO: potentially make it such that listed_tokens can be different for each batch element
    """
    batch_size = wm_windows.shape[0]
    device = wm_windows.device
    # ensure listed_tokens is a 1-D torch tensor on correct device
    if not isinstance(listed_tokens, torch.Tensor):
        listed_tokens = torch.tensor(listed_tokens, dtype=torch.long, device=device)
    else:
        listed_tokens = listed_tokens.to(device).long()
    if listed_tokens.dim() != 1:
        raise ValueError("listed_tokens must be a 1-D tensor or list of ints")
    n_listed = listed_tokens.numel()
    # Expand windows and listed tokens for batch computation
    wm_windows_exp = wm_windows.unsqueeze(1).expand(batch_size, n_listed, wm_windows.shape[1])  # b x n_listed x ngram
    listed_tokens_exp = listed_tokens.unsqueeze(0).expand(batch_size, n_listed)  # b x n_listed
    # Compute scores for each listed next token
    method = wm_args.method.lower()
    if method.startswith("bin"):  # binary
        wm_mask = prf_binary(wm_windows_exp, listed_tokens_exp, wm_args.secret_key, gamma=wm_args.gamma)
        scores = wm_mask.float()
    elif method.startswith("uni"):  # uniform
        wm_mask = prf_uniform(wm_windows_exp, listed_tokens_exp, wm_args.secret_key)
        scores = wm_mask.float()
    elif method == "none":
        scores = torch.zeros((batch_size, n_listed), device=device)
    else:
        raise ValueError(f"Unknown watermarking method: {wm_args.method}.")
    return scores


def score_all_next_tokens(
    wm_windows: torch.Tensor,
    wm_args: WatermarkConfig,
    vocab_size: int = 128256
) -> torch.Tensor:
    """
    Computes watermark scores for all possible next tokens for each window in the batch.
    Args:
        wm_windows (torch.Tensor): Tensor of shape (batch_size, ngram) representing the watermark window for each example.
        wm_args (WatermarkConfig): Watermark arguments.
        vocab_size (int): Vocabulary size (default: 128256).
    Returns:
        torch.Tensor: Watermark scores of shape (batch_size, vocab_size).
    Example:
        >>> wm_windows = torch.tensor([[1, 2], [3, 4]])  # Example batch of windows
        >>> scores = score_all_next_tokens(wm_windows, wm_args)
        >>> print(scores.shape)  # (2, 128256)
    """
    all_next_tokens = torch.arange(vocab_size, device=wm_windows.device)
    return score_listed_tokens(wm_windows, wm_args, all_next_tokens)
