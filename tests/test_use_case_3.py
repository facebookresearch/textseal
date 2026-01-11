# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys

def test_use_case_3():
    """Test detection-only mode."""
    print("Testing Use Case 3: Detection Only...")
    
    try:
        from textseal import PostHocWatermarker, WatermarkConfig, EvaluationConfig
        
        print("  - Creating detector (detection-only mode)...")
        detector = PostHocWatermarker(
            watermark_config=WatermarkConfig(watermark_type="gumbelmax", secret_key=42),
            evaluation_config=EvaluationConfig(enable_detection_only=True),
        )
        print("    ✓ Detector created successfully")
        
        print("  - Detecting watermark in test text...")
        # Test text (may or may not be watermarked - we're just testing API)
        test_text = "The quick brown fox jumps over the lazy dog."
        
        wm_eval = detector.evaluate_watermark(test_text)
        print("    ✓ Watermark detection completed successfully")
        
        print("  - Validating result structure...")
        assert isinstance(wm_eval, dict), f"Expected dict, got {type(wm_eval)}"
        assert "p_value" in wm_eval, "Missing 'p_value' in wm_eval"
        assert "det" in wm_eval, "Missing 'det' in wm_eval"
        assert "score" in wm_eval, "Missing 'score' in wm_eval"
        print(f"    ✓ Result has required keys: p_value={wm_eval['p_value']:.4f}, det={wm_eval['det']}, score={wm_eval['score']:.4f}")
        
        print("\n✓ Use Case 3 test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Use Case 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_use_case_3())
