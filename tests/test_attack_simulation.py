# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Test for attack simulation. Covers:
1. AttackConfig - Configuration for attack parameters and strengths
2. AttackSimulator - Basic attack operations on watermarked text
3. AttackSimulator.attack_all_strengths() - Multiple strength attacks
4. AttackSimulator.attack_chunks() - Batch attack processing
5. Integration testing - Attack simulation with watermarking pipeline
"""

import sys

def test_attack_simulator_basic():
    """Test basic AttackSimulator functionality."""
    print("Testing AttackSimulator basic attack...")
    
    try:
        from textseal.watermarking.attack import AttackSimulator
        from textseal.watermarking.config import AttackConfig
        
        print("  - Creating AttackSimulator...")
        attack_config = AttackConfig(
            attack_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            attack_temperature=0.7,
            attack_max_gen_len=100
        )
        simulator = AttackSimulator(attack_config=attack_config)
        print("    ✓ AttackSimulator created successfully")
        
        print("  - Performing attack on watermarked text...")
        watermarked_text = "The quick brown fox jumps over the lazy dog."
        result = simulator.attack(watermarked_text)
        print("    ✓ Attack completed successfully")
        
        print("  - Validating result structure...")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "attacked_text" in result, "Missing 'attacked_text' in result"
        assert "attack_stats" in result, "Missing 'attack_stats' in result"
        print("    ✓ Result has required keys: attacked_text, attack_stats")
        
        attacked_text = result["attacked_text"]
        assert isinstance(attacked_text, str), f"Expected str for attacked_text, got {type(attacked_text)}"
        assert len(attacked_text) > 0, "Attacked text is empty"
        print(f"    ✓ Attacked text is non-empty (length: {len(attacked_text)})")
        
        attack_stats = result["attack_stats"]
        assert isinstance(attack_stats, dict), f"Expected dict for attack_stats, got {type(attack_stats)}"
        assert "orig_wm_tokens" in attack_stats, "Missing 'orig_wm_tokens' in attack_stats"
        assert "attacked_tokens" in attack_stats, "Missing 'attacked_tokens' in attack_stats"
        print(f"    ✓ Attack stats valid: orig_tokens={attack_stats['orig_wm_tokens']}, attacked_tokens={attack_stats['attacked_tokens']}")
        
        print("\n✓ AttackSimulator basic test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_simulator_all_strengths():
    """Test AttackSimulator with all attack strengths."""
    print("Testing AttackSimulator all strengths...")
    
    try:
        from textseal.watermarking.attack import AttackSimulator
        from textseal.watermarking.config import AttackConfig
        
        print("  - Creating AttackSimulator with multiple strengths...")
        attack_config = AttackConfig(
            attack_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            attack_strengths="mild,moderate",
            attack_max_gen_len=100
        )
        simulator = AttackSimulator(attack_config=attack_config)
        print("    ✓ AttackSimulator created successfully")
        
        print("  - Performing attacks with all strengths...")
        watermarked_text = "The quick brown fox jumps over the lazy dog."
        results = simulator.attack_all_strengths(watermarked_text, verbose=True)
        print("    ✓ All attacks completed successfully")
        
        print("  - Validating results structure...")
        assert isinstance(results, dict), f"Expected dict, got {type(results)}"
        assert "mild" in results, "Missing 'mild' strength in results"
        assert "moderate" in results, "Missing 'moderate' strength in results"
        print("    ✓ Results contain expected strengths: mild, moderate")
        
        for strength, result in results.items():
            print(f"  - Validating {strength} attack result...")
            assert "attacked_text" in result, f"Missing 'attacked_text' in {strength} result"
            assert "attack_stats" in result, f"Missing 'attack_stats' in {strength} result"
            assert "strength" in result, f"Missing 'strength' in {strength} result"
            assert "temperature" in result, f"Missing 'temperature' in {strength} result"
            assert result["strength"] == strength, f"Strength mismatch: expected {strength}, got {result['strength']}"
            print(f"    ✓ {strength} result valid (temp={result['temperature']})")
        
        print("\n✓ AttackSimulator all strengths test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator all strengths test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_simulator_chunks():
    """Test AttackSimulator with multiple chunks."""
    print("Testing AttackSimulator with chunks...")
    
    try:
        from textseal.watermarking.attack import AttackSimulator
        from textseal.watermarking.config import AttackConfig
        
        print("  - Creating AttackSimulator...")
        attack_config = AttackConfig(
            attack_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            attack_max_gen_len=50
        )
        simulator = AttackSimulator(attack_config=attack_config)
        print("    ✓ AttackSimulator created successfully")
        
        print("  - Performing attack on multiple chunks...")
        chunks = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "How much wood would a woodchuck chuck?"
        ]
        results = simulator.attack_chunks(chunks, verbose=True)
        print("    ✓ Chunk attacks completed successfully")
        
        print("  - Validating results...")
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) == len(chunks), f"Expected {len(chunks)} results, got {len(results)}"
        print(f"    ✓ Got expected number of results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  - Validating chunk {i} result...")
            assert "attacked_text" in result, f"Missing 'attacked_text' in chunk {i}"
            assert "attack_stats" in result, f"Missing 'attack_stats' in chunk {i}"
            assert "chunk_idx" in result, f"Missing 'chunk_idx' in chunk {i}"
            assert result["chunk_idx"] == i, f"Chunk index mismatch: expected {i}, got {result['chunk_idx']}"
            print(f"    ✓ Chunk {i} result valid")
        
        print("\n✓ AttackSimulator chunks test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackSimulator chunks test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_config_strengths():
    """Test AttackConfig strength parsing and temperature mapping."""
    print("Testing AttackConfig strengths...")
    
    try:
        from textseal.watermarking.config import AttackConfig
        
        print("  - Testing default strength...")
        config1 = AttackConfig()
        strengths = config1.get_attack_strengths_list()
        assert isinstance(strengths, list), f"Expected list, got {type(strengths)}"
        assert len(strengths) > 0, "No default strengths configured"
        print(f"    ✓ Default strengths: {strengths}")
        
        print("  - Testing custom strength string...")
        config2 = AttackConfig(attack_strengths="mild,aggressive")
        strengths = config2.get_attack_strengths_list()
        assert strengths == ["mild", "aggressive"], f"Expected ['mild', 'aggressive'], got {strengths}"
        print(f"    ✓ Custom strengths parsed correctly: {strengths}")
        
        print("  - Testing temperature mapping...")
        for strength in ["mild", "moderate", "aggressive", "extreme"]:
            temp = config2.get_temperature(strength)
            assert isinstance(temp, (int, float)), f"Expected numeric temperature, got {type(temp)}"
            assert temp > 0, f"Expected positive temperature, got {temp}"
            print(f"    ✓ {strength} -> temperature={temp}")
        
        print("\n✓ AttackConfig strengths test passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ AttackConfig strengths test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_attack_integration_with_watermarker():
    """Test attack simulation integrated with watermarking."""
    print("Testing attack integration with watermarking...")
    
    try:
        from textseal import PostHocWatermarker, WatermarkConfig, ModelConfig, AttackConfig
        from textseal.watermarking.attack import AttackSimulator
        
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
        
        print("  - Creating attack simulator...")
        attack_config = AttackConfig(
            attack_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            attack_max_gen_len=100
        )
        simulator = AttackSimulator(attack_config=attack_config)
        print("    ✓ Attack simulator created successfully")
        
        print("  - Attacking watermarked text...")
        result = simulator.attack(watermarked_text)
        attacked_text = result["attacked_text"]
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
        ("AttackConfig strengths", test_attack_config_strengths),
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
