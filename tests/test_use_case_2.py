# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys

def test_use_case_2():
    """Test watermarking without automatic detection."""
    print("Testing Use Case 2: Watermarking Only...")
    
    try:
        from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig
        
        print("  - Creating PostHocWatermarker...")
        watermarker = PostHocWatermarker(
            watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
            model_config=ModelConfig(model_name="HuggingFaceTB/SmolLM2-135M-Instruct"),
        )
        print("    ✓ Watermarker created successfully")
        
        print("  - Watermarking test text...")
        watermarked_text = watermarker.rephrase_with_watermark("The quick brown fox jumps over the lazy dog.")
        print("    ✓ Text watermarked successfully")
        
        print("  - Validating result...")
        assert isinstance(watermarked_text, str), f"Expected str, got {type(watermarked_text)}"
        assert len(watermarked_text) > 0, "Watermarked text is empty"
        print(f"    ✓ Result is non-empty string (length: {len(watermarked_text)})")
        print(f"    ✓ Watermarked text: {watermarked_text[:100]}...")
        
        print("\n✓ Use Case 2 test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Use Case 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_use_case_2())
