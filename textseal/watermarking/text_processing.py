# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Text processing utilities for post-hoc watermarking.

This module provides text formatting, instruction templates, and other
text processing utilities for working with different language models.
"""

import re

from .config import PromptConfig


class TextProcessor:
    """Handles text processing and formatting for different models."""
    
    def __init__(self, tokenizer, model_name: str, prompt_config: PromptConfig = None):
        """
        Initialize the text processor.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            model_name: Name of the model being used
            prompt_config: Configuration for prompts and instructions
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.prompt_config = prompt_config or PromptConfig()
    
    def get_instruction_template(self, system_message: str, user_message: str, enable_thinking: bool = False) -> str:
        """
        Format message using HuggingFace chat template or fallback format.
        Args:
            system_message: System instruction
            user_message: User input
            enable_thinking: If True, enable thinking mode in chat template (for models that support it)
        Returns:
            Properly formatted instruction prompt
        """
        # Try to use HuggingFace chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            model_name = self.model_name.lower()
            if "llama" in model_name:
                # Llama 2/3: system + user
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            elif "gemma" in model_name:
                # Gemma: system as part of user, roles are user/model
                messages = [
                    {"role": "user", "content": f"{system_message}\n\n{user_message}"}
                ]
            else:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_message})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # Tokenizer template doesn't support enable_thinking kwarg.
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        else:
            # Fallback: simple format
            print("Using fallback instruction template")
            return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
    
    def get_stop_tokens(self) -> list[str]:
        """Get common stop tokens that work across models."""
        # Return common stop tokens - the tokenizer will handle model-specific ones
        return ["</s>", "<|end_of_text|>", "<|eot_id|>", "<end_of_turn>"]
    
    def clean_generated_text(self, generated_text: str) -> str:
        """Clean up generated text by removing prompt artifacts and extracting assistant response."""
        # Try to split on common assistant tokens
        assistant_tokens = [
            "<|start_header_id|>assistant<|end_header_id|>\n\n", 
            "<start_of_turn>model\n", 
            "[/INST]", 
            "Assistant:",
            "<|im_start|>assistant"
        ]
        for token in assistant_tokens:
            if token in generated_text:
                generated_text = generated_text.split(token, 1)[-1]
                break
        
        # Remove EOS/end-of-turn tokens that may appear anywhere in text.
        eos_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<end_of_turn>",
        ]
        for token in eos_tokens:
            generated_text = generated_text.replace(token, "")
        
        return generated_text.strip()

    def post_process_code(self, code: str) -> str:
        """
        Post-process generated code to fix common issues:
        - Extract contents of fenced code blocks (``` ... ```), joining them if multiple.
        - Remove fence markers if no fenced blocks are present.
        - Decode escaped unicode sequences.
        """
        # Clean up any escaped unicode sequences (causes issues with some models)
        code = code.encode("utf-8").decode("unicode_escape", errors="ignore")

        # Extract contents of fenced code blocks (``` ... ```) if present.
        matches = re.findall(r'```(?:[^\n]*)\n(.*?)```', code, re.DOTALL)
        if matches:
            # Strip surrounding blank lines from each fenced block and join with double newline
            cleaned_parts = [m.strip("\n") for m in matches]
            code = "\n\n".join(cleaned_parts)
        else:
            # Fallback: remove any standalone fence marker lines (preserve other content)
            lines = code.splitlines()
            cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
            code = "\n".join(cleaned_lines)

        return code
    
    
    def create_prompt(
        self,
        text: str,
        generation_mode: bool = False,
        context_chunks: list = None,
        enable_thinking: bool = False,
    ) -> str:
        """
        Create prompt for generation or rephrase mode.
        
        Args:
            text: Text to rephrase OR prompt/question to answer
            generation_mode: True = answer prompts, False = rephrase text
            context_chunks: Previous chunks for context (rephrase mode only)
            enable_thinking: If True, enable thinking mode in chat template
        """
        if generation_mode:
            system_message = self.prompt_config.system_message  # Empty by default for generation
            user_message = text  # Use input text directly as prompt for generation
        else:
            # Rephrase mode: rewrite existing text
            system_message = self.prompt_config.system_message or (
                "You are a text rephrasing assistant. "
                "Rephrase the given text while preserving its original meaning, style, and structure. "
                "Output only the rephrased text, with no explanations."
            )
            # Add preservation instructions
            extras = []
            if self.prompt_config.preserve_style:
                extras.append("Preserve the original writing style and tone.")
            if self.prompt_config.preserve_length:
                extras.append("Keep approximately the same length.")
            if self.prompt_config.preserve_format:
                extras.append("Maintain the original formatting.")
            if extras:
                system_message += " " + " ".join(extras)
            
            # Handle context chunks for long documents
            if context_chunks:
                system_message += " Use the previous chunks for context, but only rephrase the current chunk."
                context_text = "\n\n---\n\n".join([f"Previous chunk {i+1}:\n{c}" for i, c in enumerate(context_chunks)])
                user_message = f"{context_text}\n\n---\n\nTEXT TO REPHRASE:\n{text}"
            else:
                user_message = f"Please rephrase the following text:\n\n{text}"
        
        prompt = self.get_instruction_template(system_message, user_message, enable_thinking=enable_thinking)
        if self.prompt_config.prefill_answer and not enable_thinking:
            prompt += self.prompt_config.prefill_answer
        return prompt