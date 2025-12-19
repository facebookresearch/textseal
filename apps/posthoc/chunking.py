# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Document chunking utilities for post-hoc watermarking.

This module provides intelligent document chunking strategies that preserve
context and maintain coherence across chunk boundaries for large documents.
"""


class DocumentChunker:
    """Handles intelligent document chunking for large texts."""
    
    def __init__(self, target_size: int = 2000, overlap_ratio: float = 0.15):
        """
        Args:
            target_size: Target size for each chunk in characters
            overlap_ratio: Ratio of overlap between chunks (0.0-1.0)
        """
        self.target_size = target_size
        self.overlap_ratio = overlap_ratio
    
    def create_smart_chunks(self, text: str) -> list[dict]:
        """
        Create chunks with smart boundaries and context preservation.
        Args:
            text: Input text to chunk
        Returns:
            list of chunk dictionaries with text, context, and metadata
        """
        chunks = []
        # First, split into paragraphs (double newlines) for natural boundaries.
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            # Fallback: treat entire text as one paragraph
            paragraphs = [text.strip()]
        
        current_chunk = ""
        current_paragraphs = []
        for ii, paragraph in enumerate(paragraphs):
            # Test if adding this paragraph would exceed target size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            if len(test_chunk) <= self.target_size or current_chunk == "":
                # Add paragraph to current chunk
                current_chunk = test_chunk
                current_paragraphs.append(paragraph)
            # Current chunk is ready, save it and start new one
            else:
                # Get context from previous chunk if available
                overlap_size = int(len(current_chunk) * self.overlap_ratio)
                context_prefix = ""
                if chunks and overlap_size > 0:
                    prev_chunk_text = chunks[-1]['text']
                    context_prefix = prev_chunk_text[-overlap_size:] if len(prev_chunk_text) > overlap_size else prev_chunk_text
                chunks.append({
                    'text': current_chunk,
                    'context_prefix': context_prefix,
                    'paragraph_count': len(current_paragraphs),
                    'chunk_index': len(chunks)
                })
                # Start new chunk with current paragraph
                current_chunk = paragraph
                current_paragraphs = [paragraph]
        
        # Add final chunk if there's remaining content
        if current_chunk:
            overlap_size = int(len(current_chunk) * self.overlap_ratio)
            context_prefix = ""
            if chunks and overlap_size > 0:
                prev_chunk_text = chunks[-1]['text']
                context_prefix = prev_chunk_text[-overlap_size:] if len(prev_chunk_text) > overlap_size else prev_chunk_text
            chunks.append({
                'text': current_chunk,
                'context_prefix': context_prefix,
                'paragraph_count': len(current_paragraphs),
                'chunk_index': len(chunks)
            })
        
        return chunks

    def intelligent_merge(self, chunks: list[str], chunk_info: list[dict]) -> str:
        """
        Intelligently merge watermarked chunks back together.
        Args:
            chunks: list of watermarked chunk texts
            chunk_info: list of chunk metadata
        Returns:
            Merged document text
        """
        merged = chunks[0]
        for ii in range(1, len(chunks)):
            chunk = chunks[ii]
            if merged and not merged.endswith('\n\n'):
                if merged.endswith('\n'):
                    merged += '\n' + chunk
                else:
                    merged += '\n\n' + chunk
            else:
                merged += chunk
        
        return merged.strip()

    def recommend_processing_method(self, input_path: str) -> str:
        """
        Recommend processing method based on document characteristics.
        
        Args:
            input_path: Path to the input document
            
        Returns:
            Recommended processing method: "full", "adaptive", or "paragraphs"
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        length = len(text)
        
        # For small documents, use full processing
        if length < 3000:
            return "full"
        
        # Analyze document structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else length
        
        # Check for cross-references indicating document coherence
        cross_ref_words = ['mentioned', 'above', 'earlier', 'previously', 'furthermore', 
                          'however', 'therefore', 'thus', 'consequently', 'moreover']
        cross_refs = sum(1 for p in paragraphs 
                        for word in cross_ref_words 
                        if word in p.lower())
        cross_ref_density = cross_refs / len(paragraphs) if paragraphs else 0
        
        # Decision logic
        if cross_ref_density > 0.3:
            return "adaptive"  # High cross-reference density needs context preservation
        elif avg_paragraph_length > 800:
            return "adaptive"  # Long paragraphs need chunking
        elif length > 10000:
            return "adaptive"  # Very long documents benefit from adaptive chunking
        else:
            return "paragraphs"  # Medium documents with independent paragraphs
