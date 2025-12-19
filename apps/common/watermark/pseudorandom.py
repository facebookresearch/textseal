# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run with:
    python -m apps.common.watermark.pseudorandom
"""

import torch
import numpy as np

# Shared primes for hashing
_PRIMES = [
    10000019, 10000247, 10000439, 10000643, 10000747, 
    10000867, 10000993, 10001213, 10001357, 10001501
]
_P2 = 100000007
_P3 = 500001713
_P4 = 15485863
_M = 2**13 - 1

_MIXING_PRIME = 40499
_MIXING_SHIFT = 13

def _weighted_sum(w: torch.Tensor) -> torch.Tensor:
    """
    Computes a weighted sum of w using unique primes for each position in k.
    """
    k_dim = w.shape[-1]
    primes = torch.tensor(_PRIMES[:k_dim], dtype=torch.long, device=w.device)
    return (w.long() * primes).sum(dim=-1)

def _hash(w_weighted, x_k, sk, device):
    """
    Hashes the weighted sum, x_k, and secret key into a pseudorandom integer.
    """
    if isinstance(sk, int):
        sk_tensor = torch.full_like(x_k, sk, dtype=torch.long, device=device)
    elif isinstance(sk, torch.Tensor):
        assert sk.shape == x_k.shape, f"sk tensor must have the same shape as x_k. Got sk.shape={sk.shape}, x_k.shape={x_k.shape}"
        sk_tensor = sk.to(device).long()
    h = (w_weighted + _P2 * x_k.long() + _P3 * sk_tensor) * _P4
    h = h * _MIXING_PRIME
    h = h ^ (h >> _MIXING_SHIFT)
    return h % _M

def prf_uniform(
    w: torch.Tensor, 
    x_k: torch.Tensor, 
    sk: int | torch.Tensor
) -> torch.Tensor:
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

def prf_binary(
    w: torch.Tensor, 
    x_k: torch.Tensor, 
    sk: int | torch.Tensor, 
    gamma: float
) -> torch.Tensor:
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
    result = (hashed_uniform < gamma).to(torch.int8) # green (true) if value < γ
    return result


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
        x_k_example = torch.randint(0, vocab_size, (bsz, seq_len,))
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
            res = prf_binary(w_fixed.repeat(1,1), x_k_fixed.repeat(1), sk_varying, gamma)
            results_sk.append(res.item())
            
        print(f"Proportion of 1s over {n_trials} trials: {sum(results_sk) / n_trials}")

        # 2. Varying x_k
        print("\n--- Testing uniformity by varying x_k ---")
        sk_fixed = 54321
        results_xk = []
        for i in range(n_trials):
            x_k_varying = torch.tensor([100 + i])
            res = prf_binary(w_fixed.repeat(1,1), x_k_varying, sk_fixed, gamma)
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
        expected_std = np.sqrt(1/12) # Theoretical std dev of U(0,1)

        print(f"Generated {n_samples} samples.")
        print(f"Mean: {mean:.4f} (Expected: {expected_mean:.4f})")
        print(f"Std Dev: {std:.4f} (Expected: {expected_std:.4f})")

        # Check if values are within [0, 1)
        is_in_range = (uniform_samples.min() >= 0) and (uniform_samples.max() < 1)
        print(f"All values in [0, 1): {is_in_range}")
