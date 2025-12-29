# text_utils.py
"""Generic text processing utilities."""

import re
from typing import List

# Token counting - try tiktoken, fall back to approximation
_tokenizer = None
_tokenizer_loaded = False


def get_tokenizer():
    """Get or initialize the tokenizer (lazy loading)."""
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer
    
    _tokenizer_loaded = True
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    except ImportError:
        _tokenizer = None
    
    return _tokenizer


def count_tokens(text: str, method: str = "tiktoken") -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text to count tokens for
        method: 'tiktoken' for accurate counting, 'words' for fast approximation
    
    Returns:
        Token count
    """
    if method == "words":
        # Approximation: ~1.3 tokens per word
        words = len(text.split())
        return int(words * 1.3)
    
    # Use tiktoken if available
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Fallback to word approximation
        words = len(text.split())
        return int(words * 1.3)


def split_text(
    text: str, 
    max_tokens: int = 100, 
    token_method: str = "tiktoken"
) -> List[str]:
    """
    Split text into chunks, preferring sentence/punctuation boundaries.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        token_method: Token counting method ('tiktoken' or 'words')
    
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return [""]
    
    # If text is short enough, return as single chunk
    if count_tokens(text, token_method) <= max_tokens:
        return [text]
    
    chunks = []
    remaining = text.strip()
    max_chars_search = max_tokens * 5  # Estimate max chars to search
    
    while remaining:
        if count_tokens(remaining, token_method) <= max_tokens:
            chunks.append(remaining)
            break
        
        # Try sentence boundaries first: . ! ?
        best_pos, best_chunk = _find_split_point(
            remaining, r'([.!?]+\s+)', max_chars_search, max_tokens, token_method
        )
        
        if best_pos > 0:
            remaining = remaining[best_pos:].strip()
            chunks.append(best_chunk)
            continue
        
        # Try punctuation: , ; :
        best_pos, best_chunk = _find_split_point(
            remaining, r'([,;:]\s+)', max_chars_search, max_tokens, token_method
        )
        
        if best_pos > 0:
            remaining = remaining[best_pos:].strip()
            chunks.append(best_chunk)
            continue
        
        # Try word boundaries
        best_pos, best_chunk = _find_split_point(
            remaining, r'(\s+)', max_chars_search, max_tokens, token_method, use_start=True
        )
        
        if best_pos > 0:
            remaining = remaining[best_pos:].strip()
            chunks.append(best_chunk)
        else:
            # Force split by words
            words = remaining.split()
            chunk_words = []
            for word in words:
                test = ' '.join(chunk_words + [word])
                if count_tokens(test, token_method) > max_tokens:
                    break
                chunk_words.append(word)
            
            if chunk_words:
                chunk = ' '.join(chunk_words)
                remaining = remaining[len(chunk):].strip()
                chunks.append(chunk)
            else:
                # Single word exceeds limit
                chunk = words[0] if words else remaining[:100]
                remaining = remaining[len(chunk):].strip()
                chunks.append(chunk)
    
    return chunks


def _find_split_point(
    text: str, 
    pattern: str, 
    max_chars: int, 
    max_tokens: int, 
    token_method: str,
    use_start: bool = False
) -> tuple:
    """Find best split point matching pattern within token limit."""
    best_pos = -1
    best_chunk = ""
    
    for m in re.finditer(pattern, text[:max_chars]):
        end_pos = m.start() if use_start else m.end()
        candidate = text[:end_pos].strip()
        if count_tokens(candidate, token_method) <= max_tokens:
            best_pos = m.end()
            best_chunk = candidate
    
    return best_pos, best_chunk


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
    
    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


__all__ = [
    'count_tokens',
    'split_text',
    'split_sentences',
    'get_tokenizer',
]
