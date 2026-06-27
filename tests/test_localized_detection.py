# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Offline unit tests for the localized (adaptive ensemble) watermark detector.

These tests exercise ``adaptive_ensemble_detection`` directly on synthetic
score sequences, so they require **no model, no network, and no demo server** —
they are fast, deterministic (seeded), and suitable for CI.

Under H0 each per-token score has mean 1 and variance ``base_variance``; we
model the null as Exp(1) (mean 1, var 1) and inject a positive boost on
"watermarked" positions so their mean is clearly above 1.

Run directly:    python tests/test_localized_detection.py
Run via pytest:  pytest tests/test_localized_detection.py
"""

import numpy as np

from textseal.watermarking.localized_detection import (
    AdaptiveEnsembleResult,
    adaptive_ensemble_detection,
)

ALPHA = 0.01
MIN_LEN = 50
BASE_VAR = 1.0


def _rng(seed):
    return np.random.default_rng(seed)


def _null(rng, n):
    """Unwatermarked scores: Exp(1) (mean 1, var 1)."""
    return rng.exponential(1.0, n)


def _wm(rng, n, boost=1.8):
    """Watermarked scores: mean clearly above the null mean of 1."""
    return rng.exponential(1.0, n) + boost


def test_null_text_not_detected():
    r = adaptive_ensemble_detection(
        _null(_rng(0), 400), alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert not r.detection
    assert r.final_pvalue > ALPHA


def test_watermarked_text_detected():
    r = adaptive_ensemble_detection(
        _wm(_rng(1), 400), alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert r.detection
    assert r.final_pvalue < ALPHA
    assert len(r.token_labels) == 400
    assert sum(r.token_labels) > 300  # most tokens flagged as watermarked


def test_localized_region_recovered():
    """A watermarked block embedded in null scores is detected and localized."""
    rng = _rng(2)
    s = _null(rng, 600)
    s[250:450] = _wm(rng, 200)
    r = adaptive_ensemble_detection(
        s, alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert r.detection
    assert r.regions, "expected at least one localized region"
    start, end = r.regions[0][0], r.regions[0][1]
    overlap = max(0, min(end, 450) - max(start, 250))
    assert overlap >= 100, f"region [{start},{end}) barely overlaps truth [250,450)"


def test_multi_region_detected():
    """Two disjoint watermarked blocks are still detected."""
    rng = _rng(3)
    s = _null(rng, 700)
    s[100:200] = _wm(rng, 100)
    s[400:500] = _wm(rng, 100)
    r = adaptive_ensemble_detection(
        s, alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert r.detection


def test_short_input_no_crash():
    """Input shorter than min_length: only the global test applies, no crash."""
    r = adaptive_ensemble_detection(
        _null(_rng(4), 30), alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert not r.detection


def test_empty_input():
    r = adaptive_ensemble_detection(
        [], alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert r.final_pvalue == 1.0
    assert not r.detection
    assert r.token_labels == []


def test_weighted_path_detects():
    """Entropy weights present (k=4 path): watermarked block still detected."""
    rng = _rng(5)
    s = _null(rng, 600)
    s[250:450] = _wm(rng, 200)
    w = np.full(600, 0.2)
    w[250:450] = 1.0  # high entropy on the watermarked region
    r = adaptive_ensemble_detection(
        s, alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR, weights=w
    )
    assert r.detection


def test_false_positive_rate_controlled():
    """Across many null draws, detections stay rare at alpha=0.01."""
    rng = _rng(6)
    fp = sum(
        adaptive_ensemble_detection(
            _null(rng, 400), alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
        ).detection
        for _ in range(200)
    )
    assert fp <= 8, f"too many false positives: {fp}/200 at alpha={ALPHA}"


def test_dual_key_variance():
    """base_variance=0.5 (alpha=0.5 dual-key fused) null is not detected."""
    rng = _rng(7)
    # construct mean ~1, var ~0.5
    s = rng.exponential(1.0, 400) * np.sqrt(0.5) + (1.0 - np.sqrt(0.5))
    r = adaptive_ensemble_detection(s, alpha=ALPHA, min_length=MIN_LEN, base_variance=0.5)
    assert not r.detection


def test_result_dataclass_shape():
    r = adaptive_ensemble_detection(
        _wm(_rng(8), 200), alpha=ALPHA, min_length=MIN_LEN, base_variance=BASE_VAR
    )
    assert isinstance(r, AdaptiveEnsembleResult)
    assert isinstance(r.final_pvalue, float)
    assert isinstance(r.detection, bool)
    assert isinstance(r.regions, list)
    assert isinstance(r.token_labels, list)
    assert r.winner in {
        "global_weighted", "global_unweighted", "localized_single", "localized_multi",
    }


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print("Running localized detection unit tests...")
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print()
    if failed:
        print(f"✗ {failed}/{len(tests)} localized detection tests failed")
        return 1
    print(f"✓ All {len(tests)} localized detection tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
