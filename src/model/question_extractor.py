"""
Question Extraction Utilities for Query-Aware Routing

This module provides utilities to extract question portions from QA datasets
for use with QueryAwareRouter.

Supports:
- NarrativeQA: Format "Context: ... Question: ... Answer: ..."
- Natural Questions: Format "Question: ... Answer: ..."
- SQuAD: Format "Context: ... Question: ... Answer: ..."
"""

import torch
import re
from typing import Optional, Tuple


class QuestionExtractor:
    """
    Extract question tokens from formatted QA text.
    
    Uses keyword matching to identify question boundaries.
    """
    
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: HuggingFace tokenizer used for the dataset
        """
        self.tokenizer = tokenizer
        
        # Common question markers
        self.question_markers = [
            "Question:",
            "question:",
            "QUESTION:",
            "Q:",
            "q:",
        ]
        
        # Answer markers (end of question)
        self.answer_markers = [
            "Answer:",
            "answer:",
            "ANSWER:",
            "A:",
            "a:",
        ]
    
    def extract_question_mask_from_text(
        self,
        text: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find question span in text.
        
        Args:
            text: Full text string
        
        Returns:
            (start_pos, end_pos): Character positions of question, or (None, None)
        """
        # Find question start
        question_start = None
        for marker in self.question_markers:
            pos = text.find(marker)
            if pos != -1:
                question_start = pos + len(marker)
                break
        
        if question_start is None:
            return None, None
        
        # Find question end (answer start or end of text)
        question_end = len(text)
        for marker in self.answer_markers:
            pos = text.find(marker, question_start)
            if pos != -1:
                question_end = pos
                break
        
        return question_start, question_end
    
    def extract_question_mask_from_ids(
        self,
        input_ids: torch.Tensor,
        texts: list,
    ) -> torch.Tensor:
        """
        Create boolean mask for question tokens.
        
        Args:
            input_ids: (batch, seqlen) token IDs
            texts: List of original text strings (batch size)
        
        Returns:
            question_mask: (batch, seqlen) boolean mask, True for question tokens
        """
        batch_size, seqlen = input_ids.shape
        question_mask = torch.zeros(batch_size, seqlen, dtype=torch.bool)
        
        for i, text in enumerate(texts):
            # Find question span in text
            q_start_char, q_end_char = self.extract_question_mask_from_text(text)
            
            if q_start_char is None:
                # No question found - use heuristic (last 20% of tokens)
                question_start_token = int(0.8 * seqlen)
                question_mask[i, question_start_token:] = True
                continue
            
            # Convert character positions to token positions
            # Tokenize and find corresponding tokens
            tokens_before = self.tokenizer.encode(
                text[:q_start_char],
                add_special_tokens=False
            )
            tokens_question = self.tokenizer.encode(
                text[q_start_char:q_end_char],
                add_special_tokens=False
            )
            
            q_start_token = len(tokens_before)
            q_end_token = q_start_token + len(tokens_question)
            
            # Clamp to sequence length
            q_start_token = min(q_start_token, seqlen)
            q_end_token = min(q_end_token, seqlen)
            
            question_mask[i, q_start_token:q_end_token] = True
        
        return question_mask
    
    def extract_from_batch(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Extract question mask from a batch dict (from DataLoader).
        
        Args:
            batch: Dict with keys 'input_ids' and optionally 'question' or 'text'
        
        Returns:
            question_mask: (batch, seqlen) boolean mask
        """
        input_ids = batch['input_ids']
        
        # If batch has explicit question text, use it
        if 'question' in batch:
            texts = batch.get('context', [''] * len(batch['question']))
            questions = batch['question']
            
            # Reconstruct full text
            full_texts = []
            for ctx, q in zip(texts, questions):
                if ctx:
                    full_text = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
                else:
                    full_text = f"Question: {q}\n\nAnswer:"
                full_texts.append(full_text)
            
            return self.extract_question_mask_from_ids(input_ids, full_texts)
        
        # If batch has full text, decode and extract
        elif 'text' in batch:
            texts = batch['text']
            return self.extract_question_mask_from_ids(input_ids, texts)
        
        # Fallback: no explicit question, return None
        # Router will use heuristic
        else:
            return None


class CachedQuestionExtractor:
    """
    Caches question representations to avoid recomputation.
    
    For training efficiency, we can pre-compute question embeddings
    and cache them.
    """
    
    def __init__(self, tokenizer, cache_size: int = 10000):
        self.extractor = QuestionExtractor(tokenizer)
        self.cache = {}
        self.cache_size = cache_size
    
    def get_question_mask(
        self,
        input_ids: torch.Tensor,
        texts: Optional[list] = None,
        batch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get question mask with caching.
        
        Args:
            input_ids: (batch, seqlen)
            texts: Optional list of texts
            batch_idx: Optional batch index for cache key
        
        Returns:
            question_mask: (batch, seqlen)
        """
        # Try cache
        if batch_idx is not None and batch_idx in self.cache:
            return self.cache[batch_idx]
        
        # Compute
        if texts is not None:
            mask = self.extractor.extract_question_mask_from_ids(input_ids, texts)
        else:
            # No text available, return None (use heuristic)
            return None
        
        # Cache if not too large
        if len(self.cache) < self.cache_size and batch_idx is not None:
            self.cache[batch_idx] = mask
        
        return mask
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()


# Convenience function
def create_question_extractor(tokenizer, use_cache: bool = True):
    """
    Create a question extractor.
    
    Args:
        tokenizer: HuggingFace tokenizer
        use_cache: Whether to use caching
    
    Returns:
        QuestionExtractor or CachedQuestionExtractor
    """
    if use_cache:
        return CachedQuestionExtractor(tokenizer)
    else:
        return QuestionExtractor(tokenizer)


if __name__ == '__main__':
    # Test
    from transformers import AutoTokenizer
    
    print("Testing QuestionExtractor...")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    extractor = QuestionExtractor(tokenizer)
    
    # Test text
    text = "Context: The quick brown fox jumps over the lazy dog. Question: What color is the fox? Answer: Brown"
    
    # Find question span
    q_start, q_end = extractor.extract_question_mask_from_text(text)
    question_text = text[q_start:q_end]
    
    print(f"Full text: {text}")
    print(f"Question span: ({q_start}, {q_end})")
    print(f"Question: {question_text}")
    
    # Test with tokens
    input_ids = tokenizer.encode(text, return_tensors='pt')
    mask = extractor.extract_question_mask_from_ids(input_ids, [text])
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Question mask shape: {mask.shape}")
    print(f"Question tokens: {mask.sum().item()} / {mask.shape[1]}")
    print(f"Question ratio: {mask.float().mean().item():.2%}")
    
    print("\n✓ QuestionExtractor test passed!")