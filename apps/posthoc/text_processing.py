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
    
    def get_instruction_template(self, system_message: str, user_message: str) -> str:
        """
        Format message using HuggingFace chat template or fallback format.
        Args:
            system_message: System instruction
            user_message: User input
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
    
    def create_rephrasing_prompt(self, text: str, context_chunks: list = None) -> str:
        """
        Create a standardized rephrasing prompt for the given text.
        Args:
            text: Text to be rephrased
            context_chunks: Optional list of previously rephrased chunks for context
        Returns:
            Formatted prompt for rephrasing
        """
        # Build system message from config
        system_message = self.prompt_config.system_message
        
        # Add custom instruction if provided
        if self.prompt_config.custom_instruction:
            system_message += f" {self.prompt_config.custom_instruction}"
        
        # Add style/format preservation instructions
        additional_instructions = []
        if self.prompt_config.preserve_style:
            additional_instructions.append("Preserve the original writing style and tone.")
        if self.prompt_config.preserve_length:
            additional_instructions.append("Keep the rephrased text approximately the same length as the original.")
        if self.prompt_config.preserve_format:
            additional_instructions.append("Maintain the original formatting, including line and paragraph breaks and structure.")
        
        if additional_instructions:
            system_message += " " + " ".join(additional_instructions)
        
        # Add context-aware instruction if context chunks are provided
        if context_chunks and len(context_chunks) > 0:
            system_message += " Use the previously rephrased chunks below as context to maintain narrative coherence, but only rephrase the current chunk marked as 'TEXT TO REPHRASE'."
        
        # Format user message using template
        if context_chunks and len(context_chunks) > 0:
            # Include context chunks in the prompt
            context_text = "\n\n---\n\n".join([f"Previous chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
            user_message = f"{context_text}\n\n---\n\nTEXT TO REPHRASE:\n{text}\n\n---\n\nPlease rephrase only the text marked as 'TEXT TO REPHRASE' above, using the previous chunks for context to maintain coherence:"
        else:
            user_message = self.prompt_config.user_message_template.format(text=text)
        
        # Create final
        prompt = self.get_instruction_template(system_message, user_message)
        prompt += self.prompt_config.prefill_answer

        return prompt