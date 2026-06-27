# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Localized watermark detection — adaptive ensemble.

Implements Section 3.3 of the TextSeal paper:
  log10 p_final = min(log10 p_global, log10 p_single, log10 p_multi) + log10(k)

where:
  - p_global = full-text moment-matched Gamma (weighted if entropies provided)
  - p_single = best single dyadic window, Bonferroni-corrected by M ≈ 4n/L_min
  - p_multi  = best k-region greedy aggregate, Bonferroni-corrected by C(M,k)*Y_max
  - k = 3 (or 4 if entropy weights present, so global has both weighted+unweighted)

The greedy multi-region path repeatedly finds the most significant residual
window and masks it (set scores to zero) until adding more regions stops
improving the corrected p-value, up to Y_max.

Public surface:
  - adaptive_ensemble_detection(scores, ...) -> AdaptiveEnsembleResult
  - AdaptiveEnsembleResult dataclass

Everything else is private. All math is numpy + scipy.special.gammaincc.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special


# ----------------------------------------------------------------------
# Public dataclass
# ----------------------------------------------------------------------

@dataclass
class AdaptiveEnsembleResult:
    final_pvalue: float        # min(p_strategies) * k, capped at 1
    final_log_pvalue: float    # log of the above (for numerical stability)
    detection: bool            # final_pvalue < alpha
    global_pvalue: float       # best full-text p (weighted or unweighted)
    localized_pvalue: float    # best of {single-window, multi-region} p
    winner: str                # one of "global_weighted", "global_unweighted",
                               #        "localized_single", "localized_multi"
    regions: list              # [(start, end, p), ...] for the localized winner
    n_regions: int             # len(regions) for the multi-region path
    token_labels: list         # per-token 0/1 annotation (Stage-2 smoother)


# ----------------------------------------------------------------------
# Prefix sums and z-scores (the O(1) window-query primitives)
# ----------------------------------------------------------------------

def _prefix(scores: np.ndarray) -> np.ndarray:
    """Plain prefix sum: prefix[i] = sum(scores[:i])."""
    p = np.zeros(len(scores) + 1, dtype=np.float64)
    np.cumsum(scores, out=p[1:])
    return p


def _weighted_prefixes(scores: np.ndarray, weights: np.ndarray) -> tuple:
    """Three prefix arrays for weighted z-score: sum(s*w), sum(w), sum(w^2)."""
    n = len(scores)
    P_sw = np.zeros(n + 1)
    P_w = np.zeros(n + 1)
    P_w2 = np.zeros(n + 1)
    np.cumsum(scores * weights, out=P_sw[1:])
    np.cumsum(weights, out=P_w[1:])
    np.cumsum(weights ** 2, out=P_w2[1:])
    return P_sw, P_w, P_w2


def _window_zscore(
    P: np.ndarray, start: int, end: int, null_mean: float, base_var: float,
) -> float:
    """Raw z-score: (sum - n*mu) / sqrt(n*var)."""
    n = end - start
    if n <= 0:
        return float("-inf")
    raw_sum = P[end] - P[start]
    return (raw_sum - n * null_mean) / math.sqrt(n * base_var)


def _window_weighted_zscore(
    P_sw: np.ndarray, P_w: np.ndarray, P_w2: np.ndarray,
    start: int, end: int, base_var: float,
) -> float:
    """Weighted z-score: (sum_sw - sum_w) / sqrt(base_var * sum_w2)."""
    if end <= start:
        return float("-inf")
    sum_sw = P_sw[end] - P_sw[start]
    sum_w = P_w[end] - P_w[start]
    sum_w2 = P_w2[end] - P_w2[start]
    if sum_w2 <= 0:
        return float("-inf")
    return (sum_sw - sum_w) / math.sqrt(base_var * sum_w2)


# ----------------------------------------------------------------------
# Geometric cover: dyadic windows at L/2 stride
# ----------------------------------------------------------------------

def _dyadic_windows(n: int, min_length: int) -> list:
    """Enumerate (start, end) candidate windows. Same schedule used for both
    the search and the Bonferroni count, so the FWER bound stays valid."""
    out = []
    if n < min_length:
        return out
    max_power = int(math.floor(math.log2(n)))
    min_power = max(0, int(math.ceil(math.log2(min_length))))
    for power in range(min_power, max_power + 1):
        L = 1 << power
        if L < min_length or L > n:
            continue
        stride = max(1, L // 2)
        for start in range(0, n - L + 1, stride):
            out.append((start, start + L))
    return out


def _count_windows(n: int, min_length: int) -> int:
    """M = # of geometric-cover windows for the Bonferroni correction."""
    return max(1, len(_dyadic_windows(n, min_length)))


def _best_window(
    scores: np.ndarray, weights: Optional[np.ndarray],
    min_length: int, null_mean: float, base_var: float,
) -> tuple:
    """Return (start, end, zscore) maximizing the appropriate z-score.

    Uses weighted z-score when weights are provided, raw otherwise.
    """
    n = len(scores)
    windows = _dyadic_windows(n, min_length)
    if not windows:
        return 0, n, float("-inf")

    if weights is not None:
        P_sw, P_w, P_w2 = _weighted_prefixes(scores, weights)
        zfn = lambda s, e: _window_weighted_zscore(P_sw, P_w, P_w2, s, e, base_var)
    else:
        P = _prefix(scores)
        zfn = lambda s, e: _window_zscore(P, s, e, null_mean, base_var)

    best = (0, n, float("-inf"))
    for s, e in windows:
        z = zfn(s, e)
        if z > best[2]:
            best = (s, e, z)
    return best


# ----------------------------------------------------------------------
# P-values: global / window with optional entropy weighting + dedup
# ----------------------------------------------------------------------

def _unweighted_gamma_p(score_sum: float, n: int, base_var: float,
                        eps: float = 1e-300) -> float:
    """P(sum >= score_sum) under H0: sum ~ Gamma(n/base_var, base_var)."""
    if n <= 0:
        return 1.0
    return max(float(special.gammaincc(n / base_var, score_sum / base_var)), eps)


def _weighted_gamma_p(scores: np.ndarray, weights: np.ndarray, base_var: float,
                      eps: float = 1e-300) -> float:
    """Moment-matched Gamma p-value for weighted sum sum(w_i * s_i).

    Under H0, s_i has mean 1 and var base_var, so:
      mean[sum(w*s)] = sum(w), var = base_var * sum(w^2)
    Match Gamma(shape, scale): shape = mu^2/var, scale = var/mu.
    """
    if len(scores) == 0:
        return 1.0
    w_sum = float(np.sum(weights * scores))
    mu = float(np.sum(weights))
    var = float(np.sum(weights ** 2)) * base_var
    if var <= 1e-10 or mu <= 0:
        return 1.0
    shape = mu ** 2 / var
    scale = var / mu
    return max(float(special.gammaincc(shape, w_sum / scale)), eps)


def _dedup_window(
    tokens: Optional[list], scores: np.ndarray,
    weights: Optional[np.ndarray], start: int, end: int,
    ngram: int, scoring_method: str,
) -> tuple:
    """Return (scores_in_window, weights_in_window) after v1/v2 dedup.

    Dedup is window-local: at most one occurrence of each (n-gram window)
    or each (n-gram window, target) tuple, depending on scoring_method.
    """
    if tokens is None or scoring_method == "none":
        return scores[start:end], (weights[start:end] if weights is not None else None)

    out_scores = []
    out_weights = [] if weights is not None else None
    seen = set()
    for pos in range(start, end):
        # pos in `scores` corresponds to token at index pos+ngram+1 in `tokens`
        # (because scores skip the first ngram+1 positions during generation)
        tok_pos = pos + ngram + 1
        if tok_pos - ngram < 0 or tok_pos >= len(tokens):
            continue
        ctx = tuple(tokens[tok_pos - ngram:tok_pos])
        tok = tokens[tok_pos]
        if scoring_method == "v1":
            key = ctx
        elif scoring_method == "v2":
            key = ctx + (tok,)
        else:
            key = None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out_scores.append(scores[pos])
        if weights is not None:
            out_weights.append(weights[pos])
    return (
        np.asarray(out_scores, dtype=np.float64),
        np.asarray(out_weights, dtype=np.float64) if weights is not None else None,
    )


def _verified_pvalue(
    tokens: Optional[list], scores: np.ndarray, weights: Optional[np.ndarray],
    start: int, end: int, ngram: int, scoring_method: str,
    base_var: float, num_tests_for_bonferroni: int,
) -> float:
    """Phase B: window-local dedup → weighted-or-unweighted Gamma p → Bonferroni."""
    deduped_s, deduped_w = _dedup_window(
        tokens, scores, weights, start, end, ngram, scoring_method
    )
    n_eff = len(deduped_s)
    if n_eff == 0:
        return 1.0

    if deduped_w is not None and len(deduped_w) == n_eff:
        raw = _weighted_gamma_p(deduped_s, deduped_w, base_var)
    else:
        raw = _unweighted_gamma_p(float(np.sum(deduped_s)), n_eff, base_var)

    return min(1.0, raw * max(num_tests_for_bonferroni, 1))


# ----------------------------------------------------------------------
# Multi-region greedy extraction
# ----------------------------------------------------------------------

def _mask_region(scores: np.ndarray, weights: Optional[np.ndarray],
                 start: int, end: int, null_mean: float = 1.0) -> None:
    """In-place: set window's scores to the null mean so it can't re-win;
    zero out weights so the weighted search treats it as uninformative."""
    scores[start:end] = null_mean
    if weights is not None:
        weights[start:end] = 0.0


def _greedy_multi_regions(
    scores: np.ndarray, weights: Optional[np.ndarray],
    min_length: int, null_mean: float, base_var: float,
    max_regions: int, tokens: Optional[list], ngram: int,
    scoring_method: str, alpha: float, num_tests_for_bonferroni: int,
    hybrid_mode: bool,
) -> list:
    """Iteratively find the best residual window, verify, accept if p<alpha,
    mask it out, and repeat. Returns [(start, end, p), ...].

    Uses raw-score search when hybrid_mode=True (better localization) even
    when weights are supplied; verification still uses the weighted Gamma.
    """
    n = len(scores)
    scores_work = scores.copy()
    weights_work = weights.copy() if weights is not None else None
    found = []
    masks: list = []

    for _ in range(max_regions):
        search_weights = None if hybrid_mode else weights_work
        s, e, _ = _best_window(scores_work, search_weights, min_length, null_mean, base_var)

        # Clip overlap with any earlier masked region (dyadic windows at different
        # scales can still partially overlap a masked region).
        for m_s, m_e in masks:
            if s < m_e and e > m_s:
                left, right = m_s - s, e - m_e
                if left >= right and left >= min_length:
                    e = m_s
                elif right >= min_length:
                    s = m_e
                else:
                    s = e = 0
                    break

        if e <= s or (e - s) < min_length:
            break

        p = _verified_pvalue(
            tokens, scores, weights, s, e, ngram, scoring_method,
            base_var, num_tests_for_bonferroni,
        )
        if p >= alpha:
            break

        found.append((s, e, p))
        _mask_region(scores_work, weights_work, s, e, null_mean)
        masks.append((s, e))

    return found


def _multi_region_corrected_log_p(
    regions: list, scores: np.ndarray, weights: Optional[np.ndarray],
    tokens: Optional[list], ngram: int, scoring_method: str,
    base_var: float, n: int, min_length: int, y_max: int,
) -> tuple:
    """Aggregate y regions: their combined dedup'd p × Bonferroni C(M,y)*Y_max.

    Returns (best_log_p, best_y, best_regions).
    """
    if not regions:
        return 0.0, 0, []

    # Sort by individual p ascending so 'most significant first' greedy works.
    regions = sorted(regions, key=lambda r: r[2])
    M = _count_windows(n, min_length)
    log_ymax = math.log10(max(y_max, 1))
    best_log_p, best_y, best_regs = 0.0, 0, []

    # Document-level dedup across the union of selected regions.
    doc_seen: set = set()
    union_scores: list = []
    union_weights: list = [] if weights is not None else None

    for y, (s, e, _) in enumerate(regions[:y_max], start=1):
        # Accumulate region [s,e) with cross-region dedup.
        for pos in range(s, e):
            tok_pos = pos + ngram + 1
            if tokens is not None and 0 <= tok_pos - ngram and tok_pos < len(tokens):
                ctx = tuple(tokens[tok_pos - ngram:tok_pos])
                tok = tokens[tok_pos]
                if scoring_method == "v1":
                    key = ctx
                elif scoring_method == "v2":
                    key = ctx + (tok,)
                else:
                    key = ("nodedup", pos)
                if key in doc_seen:
                    continue
                doc_seen.add(key)
            union_scores.append(scores[pos])
            if weights is not None:
                union_weights.append(weights[pos])

        n_eff = len(union_scores)
        if n_eff == 0:
            continue
        if weights is not None and len(union_weights) == n_eff:
            raw = _weighted_gamma_p(
                np.asarray(union_scores), np.asarray(union_weights), base_var
            )
        else:
            raw = _unweighted_gamma_p(float(sum(union_scores)), n_eff, base_var)

        # Bonferroni for choosing y windows out of M, then union-bound across y values.
        log_cmy = float(special.gammaln(M + 1) - special.gammaln(y + 1)
                        - special.gammaln(M - y + 1)) / math.log(10)
        log_corrected = math.log10(max(raw, 1e-300)) + log_cmy + log_ymax
        if log_corrected < best_log_p or best_y == 0:
            best_log_p = log_corrected
            best_y = y
            best_regs = list(regions[:y])

    return best_log_p, best_y, best_regs


# ----------------------------------------------------------------------
# Stage-2 token labeling (moving average smoother)
# ----------------------------------------------------------------------

def _smooth_labels(
    scores: np.ndarray, weights: Optional[np.ndarray],
    window: int, threshold: float, null_mean: float,
    min_weight_sum: float = 0.1,
) -> list:
    """Per-token 0/1 labels via a (weighted) moving average vs threshold."""
    n = len(scores)
    if n == 0:
        return []
    half = window // 2
    if weights is None or len(weights) != n:
        padded = np.pad(scores, (half, half), mode="edge")
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="valid")[:n]
    else:
        smoothed = np.empty(n)
        for t in range(n):
            a, b = max(0, t - half), min(n, t + half + 1)
            w_sum = float(np.sum(weights[a:b]))
            if w_sum < min_weight_sum:
                smoothed[t] = null_mean
            else:
                smoothed[t] = float(np.sum(scores[a:b] * weights[a:b])) / w_sum
    return (smoothed > (threshold * null_mean)).astype(int).tolist()


# ----------------------------------------------------------------------
# Public entry point: adaptive ensemble
# ----------------------------------------------------------------------

def adaptive_ensemble_detection(
    scores,
    *,
    alpha: float = 0.01,
    min_length: int = 50,
    null_mean: float = 1.0,
    max_regions: int = 5,
    tokens: Optional[list] = None,
    ngram: int = 1,
    scoring_method: str = "v2",
    weights=None,
    hybrid_mode: bool = True,
    enable_stage2: bool = True,
    smoother_window: int = 20,
    smoother_threshold: float = 1.2,
    base_variance: float = 1.0,
) -> AdaptiveEnsembleResult:
    """Adaptive ensemble detection — Section 3.3 of the TextSeal paper.

    Three strategies are computed in parallel; the minimum p-value wins, with a
    flat Bonferroni-k correction (k = 3 without entropy weights, k = 4 with).

    Args:
        scores: per-token fused dual-key Gumbel scores (mean ~1 under H0).
        alpha: detection significance threshold.
        min_length: minimum dyadic-window length L_min (default 50).
        null_mean: E[score] under H0 (1.0 for Gumbel-max).
        max_regions: Y_max for the multi-region greedy extraction.
        tokens / ngram / scoring_method: needed for window-local dedup ("v2" is
            the strictest and what the paper uses).
        weights: per-token entropy weights (paper Eq. 4). If None, falls back
            to unweighted Gamma everywhere.
        hybrid_mode: search with raw scores, verify with weighted Gamma.
        enable_stage2: if detection succeeds, run the smoother to label tokens.
        smoother_window, smoother_threshold: Stage-2 smoother params.
        base_variance: variance of per-token score under H0. Use 0.5 for
            single-key Exp(1), or alpha^2 + (1-alpha)^2 for dual-key fused.

    Returns:
        AdaptiveEnsembleResult.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)
    if n == 0:
        return AdaptiveEnsembleResult(
            final_pvalue=1.0, final_log_pvalue=0.0, detection=False,
            global_pvalue=1.0, localized_pvalue=1.0, winner="global",
            regions=[], n_regions=0, token_labels=[],
        )

    weights_arr = None
    if weights is not None and len(weights) > 0:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != n:
            w = np.resize(w, n)
        weights_arr = w
    use_weighted = weights_arr is not None

    # ---- Strategy 1: global ----
    global_unweighted = _unweighted_gamma_p(float(np.sum(scores_arr)), n, base_variance)
    if use_weighted:
        global_weighted = _weighted_gamma_p(scores_arr, weights_arr, base_variance)
        global_pvalue = min(global_weighted, global_unweighted)
        global_winner = "global_weighted" if global_weighted <= global_unweighted else "global_unweighted"
    else:
        global_weighted = global_unweighted
        global_pvalue = global_unweighted
        global_winner = "global_unweighted"

    # ---- Strategy 2: single best window ----
    M = _count_windows(n, min_length)
    s, e, _ = _best_window(
        scores_arr,
        None if hybrid_mode else weights_arr,
        min_length, null_mean, base_variance,
    )
    if e > s and (e - s) >= min_length:
        single_pvalue = _verified_pvalue(
            tokens, scores_arr, weights_arr, s, e, ngram, scoring_method,
            base_variance, num_tests_for_bonferroni=M,
        )
        single_regions = [(s, e, single_pvalue)]
    else:
        single_pvalue = 1.0
        single_regions = []

    # ---- Strategy 3: multi-region greedy ----
    # Phase A: find disjoint candidate windows; Phase B: aggregate up to Y_max.
    raw_regions = _greedy_multi_regions(
        scores_arr, weights_arr, min_length, null_mean, base_variance,
        max_regions, tokens, ngram, scoring_method, alpha,
        num_tests_for_bonferroni=M, hybrid_mode=hybrid_mode,
    )
    if len(raw_regions) > 1:
        multi_log_p, multi_y, multi_regs = _multi_region_corrected_log_p(
            raw_regions, scores_arr, weights_arr, tokens, ngram, scoring_method,
            base_variance, n, min_length, max_regions,
        )
    else:
        multi_log_p, multi_y, multi_regs = 0.0, 0, []
    multi_pvalue = min(1.0, 10 ** multi_log_p) if multi_log_p < 0 else 1.0

    # ---- Ensemble: min over k strategies + Bonferroni-k correction ----
    localized_pvalue = min(single_pvalue, multi_pvalue)
    if single_pvalue <= multi_pvalue:
        localized_winner, localized_regs = "localized_single", single_regions
    else:
        localized_winner, localized_regs = "localized_multi", multi_regs

    k = 4 if use_weighted else 3
    candidates_log = {
        "global_weighted": math.log10(max(global_weighted, 1e-300)),
        "global_unweighted": math.log10(max(global_unweighted, 1e-300)),
        "localized_single": math.log10(max(single_pvalue, 1e-300)),
        "localized_multi": math.log10(max(multi_pvalue, 1e-300)),
    }
    if not use_weighted:
        del candidates_log["global_weighted"]

    final_log_pvalue = min(candidates_log.values()) + math.log10(k)
    final_log_pvalue = min(0.0, final_log_pvalue)  # cap at log10(1)
    final_pvalue = 10 ** final_log_pvalue
    winner = min(candidates_log, key=candidates_log.get)
    if winner.startswith("global"):
        emitted_regions = []
    else:
        emitted_regions = localized_regs

    # ---- Stage 2: per-token labels (only if detection fired) ----
    detection = final_pvalue < alpha
    if enable_stage2 and detection:
        labels = _smooth_labels(
            scores_arr, weights_arr,
            smoother_window, smoother_threshold, null_mean,
        )
    else:
        labels = [0] * n

    return AdaptiveEnsembleResult(
        final_pvalue=final_pvalue,
        final_log_pvalue=final_log_pvalue,
        detection=detection,
        global_pvalue=global_pvalue,
        localized_pvalue=localized_pvalue,
        winner=winner,
        regions=emitted_regions,
        n_regions=len(emitted_regions),
        token_labels=labels,
    )
