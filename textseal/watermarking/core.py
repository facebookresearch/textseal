# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run with:
    python -m textseal.watermarking.core
"""

import ast
import json

import numpy as np
import torch

from textseal.watermarking.config import WatermarkConfig


# Shared primes for hashing.
# We use a fixed list of primes to compute a weighted sum of the input window w.
# The rest of the hashing function uses modular arithmetic to mix this weighted sum with the input token and secret key.
_PRIMES = [10000019, 10000247, 10000439, 10000643, 10000747, 10000867, 10000993, 10001213, 10001357, 10001501]
_P2 = 100000007
_P3 = 500001713
_P4 = 15485863
_M = 2**13 - 1

_MIXING_PRIME = 40499
_MIXING_SHIFT = 13


def _get_primes(k_dim: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(_PRIMES[:k_dim], dtype=torch.long, device=device)


def _weighted_sum(w: torch.Tensor) -> torch.Tensor:
    """
    Computes a weighted sum of w using unique primes for each position in k.
    """
    k_dim = w.shape[-1]
    primes = _get_primes(k_dim, w.device)
    return (w.long() * primes).sum(dim=-1)


def _hash(w_weighted, x_k, sk, device):
    """
    Hashes the weighted sum, x_k, and secret key into a pseudorandom integer.
    """
    if isinstance(sk, int):
        sk_tensor = torch.full_like(x_k, sk, dtype=torch.long, device=device)
    elif isinstance(sk, torch.Tensor):
        assert sk.shape == x_k.shape, (
            f"sk tensor must have the same shape as x_k. Got sk.shape={sk.shape}, x_k.shape={x_k.shape}"
        )
        sk_tensor = sk.to(device).long()
    h = (w_weighted + _P2 * x_k.long() + _P3 * sk_tensor) * _P4
    h = h * _MIXING_PRIME
    h = h ^ (h >> _MIXING_SHIFT)
    return h % _M


def prf_uniform(w: torch.Tensor, x_k: torch.Tensor, sk: int | torch.Tensor) -> torch.Tensor:
    """
    Generates a pseudorandom float tensor uniformly distributed between 0 and 1.
    Now sensitive to the order of elements in w along the k dimension.

    Args:
        w (torch.Tensor): A batch of windows (token sequences), shape (bsz, k).
        x_k (torch.Tensor): A batch of tokens to score, shape (bsz, seq_len) or (bsz).
        sk (int | torch.Tensor): Secret key, either a single integer or a tensor with the same shape as x_k.

    Returns:
        torch.Tensor: A tensor of shape (bsz,) with float values
                      uniformly distributed in [0, 1).
    """
    device = w.device
    x_k = x_k.to(device)
    w_weighted = _weighted_sum(w)
    hashed_values = _hash(w_weighted, x_k, sk, device)
    return hashed_values.float() / _M


def prf_dual(
    w: torch.Tensor,
    x_k: torch.Tensor,
    sk_a: int | torch.Tensor,
    sk_b: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = w.device
    x_k = x_k.to(device)
    w_weighted = _weighted_sum(w)
    scores_a = _hash(w_weighted, x_k, sk_a, device).float() / _M
    scores_b = _hash(w_weighted, x_k, sk_b, device).float() / _M
    return scores_a, scores_b


def prf_binary(w: torch.Tensor, x_k: torch.Tensor, sk: int | torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Generates a pseudorandom binary tensor using modular arithmetic.

    Args:
        w (torch.Tensor): A batch of windows (token sequences), shape (bsz, seq_len, k) or (bsz, k).
        x_k (torch.Tensor): A batch of tokens to score, shape (bsz, seq_len) or (bsz).
        sk (int | torch.Tensor): Secret key, either a single integer or a tensor with the same shape as x_k.
        gamma (float): The desired proportion of outputs that should be 1 (0.0 to 1.0).

    Returns:
        torch.Tensor: A tensor of shape (bsz, seq_len) or (bsz) with binary values (0 or 1).
    """
    hashed_uniform = prf_uniform(w, x_k, sk)  # b s or b
    result = (hashed_uniform < gamma).to(torch.int8)  # green (true) if value < γ
    return result


@torch.compile(fullgraph=True)
def _prf_core_compiled(w_weighted: torch.Tensor, token_ids: torch.Tensor, sk: torch.Tensor) -> torch.Tensor:
    h = (w_weighted.unsqueeze(-1) + _P2 * token_ids.long() + _P3 * sk) * (_P4 * _MIXING_PRIME)
    h = h ^ (h >> _MIXING_SHIFT)
    return (h % _M).float() / _M


@torch.compile(fullgraph=True)
def _prf_dual_compiled(
    w_weighted: torch.Tensor,
    token_ids: torch.Tensor,
    sk_a: torch.Tensor,
    sk_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = w_weighted.unsqueeze(-1) + _P2 * token_ids.long()
    constant = _P4 * _MIXING_PRIME
    h_a = (base + _P3 * sk_a) * constant
    h_a = h_a ^ (h_a >> _MIXING_SHIFT)
    h_b = (base + _P3 * sk_b) * constant
    h_b = h_b ^ (h_b >> _MIXING_SHIFT)
    return (h_a % _M).float() / _M, (h_b % _M).float() / _M


def fast_prf_scores(w: torch.Tensor, token_ids: torch.Tensor, sk: int) -> torch.Tensor:
    w_weighted = _weighted_sum(w)
    sk_t = torch.tensor(sk, dtype=torch.long, device=w.device)
    return _prf_core_compiled(w_weighted, token_ids, sk_t)


def fast_prf_dual(
    w: torch.Tensor,
    token_ids: torch.Tensor,
    sk_a: int,
    sk_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    w_weighted = _weighted_sum(w)
    sk_a_t = torch.tensor(sk_a, dtype=torch.long, device=w.device)
    sk_b_t = torch.tensor(sk_b, dtype=torch.long, device=w.device)
    return _prf_dual_compiled(w_weighted, token_ids, sk_a_t, sk_b_t)


def _load_filter_cache(filter_path: str, filter_keys: str) -> tuple[list[str], dict]:
    """Load and cache filter data to avoid repeated file I/O.

    filter_path: Path to the filter file. should be a JSON file with keys mapping to frequency dicts.
    filter_keys: Comma-separated keys to load from the filter file.

    """
    global _filter_cache
    if "_filter_cache" not in globals():
        _filter_cache = {}

    cache_key = (filter_path, filter_keys)
    if cache_key not in _filter_cache:
        with open(filter_path, "r") as f:
            filter_dict = json.load(f)
        filter_key_list = filter_keys.split(",")  # Expected to be comma-separated
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
    filter_freqs: dict,
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
    seen_windows: set = None,
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
        "total": total_count,
        "duplicates": duplicate_count,
        "kept": valid_mask.sum().item(),
        "dedup_rate": duplicate_count / total_count if total_count > 0 else 0.0,
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
    seen_windows: set = None,
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
        wm_windows, targets, sources = _apply_filter(wm_windows, targets, sources, filter_key_list, filter_freqs)
        if sources is not None:
            sk = wm_args.secret_key + sources

    if wm_args.scoring_method in ["v1", "v2"]:
        wm_windows, targets, sources, dedup_stats = _apply_deduplication(
            wm_windows, targets, sources, wm_args, filter_key_list if use_filter else None, seen_windows=seen_windows
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
    wm_windows: torch.Tensor, wm_args: WatermarkConfig, listed_tokens: list[int] | torch.Tensor
) -> torch.Tensor:
    """
    Computes watermark scores for a provided list of next-token ids for each window in the batch.
    Args:
        wm_windows (torch.Tensor): Tensor of shape (batch_size, ngram) representing the watermark window for each example.
        wm_args (WatermarkConfig): Watermark arguments.
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


def score_all_next_tokens(wm_windows: torch.Tensor, wm_args: WatermarkConfig, vocab_size: int = 128256) -> torch.Tensor:
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


if __name__ == "__main__":
    test_uniform = True
    test_binary = True

    if test_binary:
        # Example Usage
        bsz = 10
        seq_len = 3
        k = 5
        vocab_size = 128_000

        # Create some example tensors
        w_example = torch.randint(0, vocab_size, (bsz, seq_len, k))
        x_k_example = torch.randint(
            0,
            vocab_size,
            (
                bsz,
                seq_len,
            ),
        )
        sk_example = 12345
        gamma = 0.5

        # Generate the pseudorandom binary tensor
        output = prf_binary(w_example, x_k_example, sk_example, gamma)

        print("\n--- Example PRF Output: ---")
        print("Input w:\n", w_example)
        print("\nInput x_k:\n", x_k_example)
        print("\nSecret Key sk:", sk_example)
        print("\nGamma:", gamma)
        print("\nOutput tensor:\n", output)
        print("\nProportion of 1s:", (output.sum() / bsz / seq_len).item())

        # --- Test for uniformity when varying one factor ---

        # 1. Varying sk
        print("\n--- Testing uniformity by varying sk ---")
        n_trials = 10000
        w_fixed = torch.randint(0, vocab_size, (1, k))
        x_k_fixed = torch.randint(0, vocab_size, (1,))

        results_sk = []
        for i in range(n_trials):
            sk_varying = 10000 + i * 7
            res = prf_binary(w_fixed.repeat(1, 1), x_k_fixed.repeat(1), sk_varying, gamma)
            results_sk.append(res.item())

        print(f"Proportion of 1s over {n_trials} trials: {sum(results_sk) / n_trials}")

        # 2. Varying x_k
        print("\n--- Testing uniformity by varying x_k ---")
        sk_fixed = 54321
        results_xk = []
        for i in range(n_trials):
            x_k_varying = torch.tensor([100 + i])
            res = prf_binary(w_fixed.repeat(1, 1), x_k_varying, sk_fixed, gamma)
            results_xk.append(res.item())

        print(f"Proportion of 1s over {n_trials} trials: {sum(results_xk) / n_trials}")

        # 3. Varying w
        print("\n--- Testing uniformity by varying w ---")
        results_w = []
        for i in range(n_trials):
            w_varying = torch.randint(0, vocab_size, (1, k))
            res = prf_binary(w_varying, x_k_fixed.repeat(1), sk_fixed, gamma)
            results_w.append(res.item())

        print(f"Proportion of 1s over {n_trials} trials: {sum(results_w) / n_trials}")

    if test_uniform:
        # --- Testing prf_uniform ---
        print("\n\n--- Testing prf_uniform ---")

        # Example Usage
        bsz = 10
        k = 5
        w_example = torch.randint(0, vocab_size, (bsz, k))
        x_k_example = torch.randint(0, vocab_size, (bsz,))
        sk_example = 12345

        # Generate the pseudorandom float tensor
        output_uniform = prf_uniform(w_example, x_k_example, sk_example)

        print("\n--- Example prf_uniform Output: ---")
        print("Input w:\n", w_example)
        print("\nInput x_k:\n", x_k_example)
        print("\nSecret Key sk:", sk_example)
        print("\nOutput tensor (uniform float):\n", output_uniform)

        # --- Test for uniform distribution properties ---
        print("\n--- Testing statistical properties of prf_uniform output ---")
        n_samples = 100000000
        w_large = torch.randint(0, vocab_size, (n_samples, k))
        x_k_large = torch.randint(0, vocab_size, (n_samples,))
        sk_fixed = 98765

        uniform_samples = prf_uniform(w_large, x_k_large, sk_fixed)

        mean = torch.mean(uniform_samples).item()
        std = torch.std(uniform_samples).item()

        expected_mean = 0.5
        expected_std = np.sqrt(1 / 12)  # Theoretical std dev of U(0,1)

        print(f"Generated {n_samples} samples.")
        print(f"Mean: {mean:.4f} (Expected: {expected_mean:.4f})")
        print(f"Std Dev: {std:.4f} (Expected: {expected_std:.4f})")

        # Check if values are within [0, 1)
        is_in_range = (uniform_samples.min() >= 0) and (uniform_samples.max() < 1)
        print(f"All values in [0, 1): {is_in_range}")
