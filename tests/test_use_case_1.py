# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys

def test_use_case_1():
    """Test watermarking with automatic detection."""
    print("Testing Use Case 1: Watermarking + Detection...")
    
    try:
        from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig, ProcessingConfig
        
        print("  - Creating PostHocWatermarker...")
        watermarker = PostHocWatermarker(
            watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
            model_config=ModelConfig(model_name="HuggingFaceTB/SmolLM2-135M-Instruct"),
            processing_config=ProcessingConfig(temperature=0.8),
        )
        print("    ✓ Watermarker created successfully")
        
        print("  - Processing test text...")
        result = watermarker.process_text("The quick brown fox jumps over the lazy dog.")
        print("    ✓ Text processed successfully")
        
        print("  - Validating result structure...")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "wm_text" in result, "Missing 'wm_text' in result"
        assert "wm_eval" in result, "Missing 'wm_eval' in result"
        print("    ✓ Result has required keys: wm_text, wm_eval")
        
        print("  - Validating wm_eval structure...")
        wm_eval = result["wm_eval"]
        assert isinstance(wm_eval, dict), f"Expected wm_eval to be dict, got {type(wm_eval)}"
        assert "p_value" in wm_eval, "Missing 'p_value' in wm_eval"
        assert "det" in wm_eval, "Missing 'det' in wm_eval"
        assert "score" in wm_eval, "Missing 'score' in wm_eval"
        print(f"    ✓ wm_eval has required keys: p_value={wm_eval['p_value']:.4f}, det={wm_eval['det']}, score={wm_eval['score']:.4f}")
        
        print("\n✓ Use Case 1 test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Use Case 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_use_case_1())
