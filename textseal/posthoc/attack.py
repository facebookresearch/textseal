# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Attack simulation module for testing watermark robustness.

This module provides functionality to simulate attacks on watermarked text
by rephrasing it using a language model, then evaluating watermark detection
on both the original watermarked text and the attacked version.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from textseal.posthoc.config import AttackConfig, ModelConfig

# Length of user text to use as a marker for extracting attacked text from model output
USER_TEXT_MARKER_LENGTH = 50


def load_attack_model(attack_config: AttackConfig, cache_dir: str = None):
    """
    Load the attack model and tokenizer.
    
    Args:
        attack_config: Attack configuration
        cache_dir: HuggingFace cache directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading attack model: {attack_config.attack_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        attack_config.attack_model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "cache_dir": cache_dir
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        attack_config.attack_model_name,
        **model_kwargs
    )
    
    print(f"✓ Attack model loaded successfully")
    return model, tokenizer


class AttackSimulator:
    """Simulates attacks on watermarked text by rephrasing."""
    
    def __init__(
        self,
        attack_config: AttackConfig,
        cache_dir: str = None,
        model=None,
        tokenizer=None
    ):
        """
        Initialize attack simulator.
        
        Args:
            attack_config: Attack configuration
            cache_dir: HuggingFace cache directory
            model: Pre-loaded attack model (optional, will load if not provided)
            tokenizer: Pre-loaded attack tokenizer (optional, will load if not provided)
        """
        self.attack_config = attack_config
        
        # Load or use provided model/tokenizer
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_attack_model(attack_config, cache_dir)
        else:
            self.model = model
            self.tokenizer = tokenizer
            print(f"✓ Using provided attack model: {attack_config.attack_model_name}")
        
        # Log attack strengths
        strengths = attack_config.get_attack_strengths_list()
        print(f"  Attack strengths to run: {strengths}")
    
    def attack(
        self,
        watermarked_text: str,
        max_gen_len: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        strength: Optional[str] = None
    ) -> dict:
        """
        Perform rephrasing attack on watermarked text.
        
        Args:
            watermarked_text: The watermarked text to attack
            max_gen_len: Maximum generation length (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            strength: Attack strength to use (uses config default if None)
            
        Returns:
            Dictionary containing:
                - attacked_text: The rephrased (attacked) text
                - attack_stats: Statistics about the attack
        """
        # Use config defaults if not specified
        if max_gen_len is None:
            max_gen_len = self.attack_config.attack_max_gen_len
        temperature = temperature or self.attack_config.get_temperature(strength)
        if top_p is None:
            top_p = self.attack_config.attack_top_p
        
        # Build the attack prompt using system and user messages
        messages = [
            {"role": "system", "content": self.attack_config.get_system_message(strength)},
            {"role": "user", "content": self.attack_config.get_user_template(strength).format(text=watermarked_text)}
        ]
        
        # Use chat template to format the prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate attacked text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the output
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the rephrased text (remove the prompt part)
        # The output contains both the prompt and the response, so we remove the prompt
        if full_output.startswith(prompt):
            attacked_text = full_output[len(prompt):].strip()
        else:
            # Fallback: try to find where the response starts after the last user message
            # Look for the text after the user's input
            user_text_marker = watermarked_text[:USER_TEXT_MARKER_LENGTH]
            if user_text_marker in full_output:
                # Find everything after the user's text
                idx = full_output.find(user_text_marker)
                remaining = full_output[idx + len(watermarked_text):].strip()
                attacked_text = remaining if remaining else full_output
            else:
                # Last resort: just use the full output
                attacked_text = full_output.strip()
        
        # Compute statistics
        attack_stats = {
            "orig_wm_tokens": len(self.tokenizer.encode(watermarked_text, add_special_tokens=False)),
            "attacked_tokens": len(self.tokenizer.encode(attacked_text, add_special_tokens=False)),
        }
        
        return {
            "attacked_text": attacked_text,
            "attack_stats": attack_stats
        }
    
    def attack_all_strengths(
        self,
        watermarked_text: str,
        max_gen_len: Optional[int] = None,
        top_p: Optional[float] = None,
        verbose: bool = False
    ) -> dict:
        """
        Perform attacks with all configured strengths.
        
        Args:
            watermarked_text: The watermarked text to attack
            max_gen_len: Maximum generation length
            top_p: Top-p sampling parameter
            verbose: Print progress
            
        Returns:
            Dictionary with results for each strength:
            {
                "mild": {"attacked_text": ..., "attack_stats": ...},
                "moderate": {"attacked_text": ..., "attack_stats": ...},
                ...
            }
        """
        strengths = self.attack_config.get_attack_strengths_list()
        results = {}
        
        for strength in strengths:
            if verbose:
                temp = self.attack_config.get_temperature(strength)
                print(f"  Running {strength} attack (temp={temp})...")
            
            result = self.attack(
                watermarked_text,
                max_gen_len=max_gen_len,
                top_p=top_p,
                strength=strength
            )
            result["strength"] = strength
            result["temperature"] = self.attack_config.get_temperature(strength)
            results[strength] = result
        
        return results
    
    def attack_chunks(
        self,
        watermarked_chunks: list,
        max_gen_len: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        verbose: bool = False
    ) -> list:
        """
        Perform attack on multiple chunks.
        
        Args:
            watermarked_chunks: List of watermarked text chunks
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            verbose: Print progress
            
        Returns:
            List of dictionaries with attacked chunks and stats
        """
        attacked_chunks = []
        
        for i, chunk in enumerate(watermarked_chunks):
            if verbose:
                print(f"Attacking chunk {i+1}/{len(watermarked_chunks)}...")
            
            result = self.attack(chunk, max_gen_len, temperature, top_p)
            result["chunk_idx"] = i
            attacked_chunks.append(result)
        
        return attacked_chunks
