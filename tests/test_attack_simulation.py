# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Test for attack simulation (no-box RephrasingAttack). Covers:
1. Basic attack on watermarked text
2. Attacks at several temperatures
3. Batched attack on multiple chunks
4. Integration testing - Attack simulation with watermarking pipeline
"""

import sys


def _attack(temperature=1.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from textseal.attacks.oracle import NoBox
    from textseal.attacks.rephrasing import RephrasingAttack
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    attack = RephrasingAttack(model, AutoTokenizer.from_pretrained(model_name), NoBox())
    attack.temperature = temperature
    return attack

def test_attack_simulator_basic():
    """Test basic AttackSimulator functionality."""
    print("Testing AttackSimulator basic attack...")
    
    try:
        print("  - Creating RephrasingAttack...")
        attack = _attack(temperature=0.7)
        print("    ✓ RephrasingAttack created successfully")
        
        print("  - Performing attack on watermarked text...")
        watermarked_text = "The quick brown fox jumps over the lazy dog."
        result = attack.run_batch([watermarked_text], max_new_tokens=100)[0]
        print("    ✓ Attack completed successfully")
        
        print("  - Validating result structure...")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "text" in result, "Missing 'text' in result"
        print("    ✓ Result has required key: text")
        
        attacked_text = result["text"]
        assert isinstance(attacked_text, str), f"Expected str for attacked_text, got {type(attacked_text)}"
        assert len(attacked_text) > 0, "Attacked text is empty"
        print(f"    ✓ Attacked text is non-empty (length: {len(attacked_text)})")
        
        print("\n✓ AttackSimulator basic test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_simulator_all_strengths():
    """Test attacks at several temperatures."""
    print("Testing attacks at several temperatures...")
    
    try:
        print("  - Creating RephrasingAttack...")
        attack = _attack()
        print("    ✓ RephrasingAttack created successfully")
        
        print("  - Performing attacks at all temperatures...")
        watermarked_text = "The quick brown fox jumps over the lazy dog."
        results = {}
        for temperature in (0.5, 0.8):
            attack.temperature = temperature
            results[temperature] = attack.run_batch([watermarked_text], max_new_tokens=100)[0]
        print("    ✓ All attacks completed successfully")
        
        for temperature, result in results.items():
            print(f"  - Validating temperature {temperature} attack result...")
            assert "text" in result, f"Missing 'text' in temperature {temperature} result"
            assert result["text"], f"Empty attacked text at temperature {temperature}"
            print(f"    ✓ temperature {temperature} result valid")
        
        print("\n✓ AttackSimulator all strengths test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator all strengths test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_simulator_chunks():
    """Test batched attack on multiple chunks."""
    print("Testing batched attack on chunks...")
    
    try:
        print("  - Creating RephrasingAttack...")
        attack = _attack()
        print("    ✓ RephrasingAttack created successfully")
        
        print("  - Performing attack on multiple chunks...")
        chunks = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "How much wood would a woodchuck chuck?"
        ]
        results = attack.run_batch(chunks, max_new_tokens=50)
        print("    ✓ Chunk attacks completed successfully")
        
        print("  - Validating results...")
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) == len(chunks), f"Expected {len(chunks)} results, got {len(results)}"
        print(f"    ✓ Got expected number of results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  - Validating chunk {i} result...")
            assert "text" in result, f"Missing 'text' in chunk {i}"
            print(f"    ✓ Chunk {i} result valid")
        
        print("\n✓ AttackSimulator chunks test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator chunks test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_integration_with_watermarker():
    """Test attack simulation integrated with watermarking."""
    print("Testing attack integration with watermarking...")
    
    try:
        from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig
        
        print("  - Creating watermarker...")
        watermarker = PostHocWatermarker(
            watermark_config=WatermarkConfig(watermark_type="gumbelmax"),
            model_config=ModelConfig(model_name="HuggingFaceTB/SmolLM2-135M-Instruct"),
        )
        print("    ✓ Watermarker created successfully")
        
        print("  - Watermarking test text...")
        original_text = "The quick brown fox jumps over the lazy dog."
        watermarked_text = watermarker.rephrase_with_watermark(original_text)
        print("    ✓ Text watermarked successfully")
        
        print("  - Creating RephrasingAttack...")
        attack = _attack()
        print("    ✓ RephrasingAttack created successfully")
        
        print("  - Attacking watermarked text...")
        attacked_text = attack.run_batch([watermarked_text], max_new_tokens=100)[0]["text"]
        print("    ✓ Attack completed successfully")
        
        print("  - Evaluating watermark on original and attacked text...")
        wm_eval_original = watermarker.evaluate_watermark(watermarked_text)
        wm_eval_attacked = watermarker.evaluate_watermark(attacked_text)
        print("    ✓ Watermark evaluation completed successfully")
        
        print("  - Validating evaluations...")
        assert isinstance(wm_eval_original, dict), "Original evaluation should be dict"
        assert isinstance(wm_eval_attacked, dict), "Attacked evaluation should be dict"
        assert "p_value" in wm_eval_original, "Missing p_value in original evaluation"
        assert "p_value" in wm_eval_attacked, "Missing p_value in attacked evaluation"
        print(f"    ✓ Original p_value: {wm_eval_original['p_value']:.4f}")
        print(f"    ✓ Attacked p_value: {wm_eval_attacked['p_value']:.4f}")
        
        print("\n✓ Attack integration test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Attack integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_all_tests():
    """Run all attack simulation tests."""
    print("="*60)
    print("Running Attack Simulation Tests")
    print("="*60 + "\n")
    
    tests = [
        ("AttackSimulator basic", test_attack_simulator_basic),
        ("AttackSimulator all strengths", test_attack_simulator_all_strengths),
        ("AttackSimulator chunks", test_attack_simulator_chunks),
        ("Attack integration", test_attack_integration_with_watermarker),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print("\n" + "-"*60)
        result = test_func()
        if result == 0:
            passed += 1
        else:
            failed += 1
        print("-"*60)
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
