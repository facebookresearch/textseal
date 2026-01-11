# Copyright (c) Meta Platforms, Inc. and affiliates.


import sys

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Test public API imports
        print("  - Testing public API imports...")
        from textseal import (
            PostHocWatermarker,
            WatermarkConfig,
            ModelConfig,
            ProcessingConfig,
            EvaluationConfig,
            PromptConfig,
        )
        print("    ✓ All 6 public APIs imported successfully")
        
        # Test critical internal import (this was the blocker)
        print("  - Testing critical internal import...")
        from textseal.common.utils.config import cfg_from_cli
        print("    ✓ textseal.common.utils.config imported successfully")
        
        # Test posthoc modules
        print("  - Testing posthoc modules...")
        from textseal.posthoc.detector import WmDetector
        from textseal.posthoc.evaluation import WatermarkEvaluator
        print("    ✓ Posthoc modules imported successfully")
        
        print("\n✓ All import tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
