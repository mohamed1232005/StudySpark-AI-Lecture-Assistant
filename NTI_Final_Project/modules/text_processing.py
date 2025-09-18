import re
from typing import Iterable, List

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """Very simple word-based chunking (works well with our HF pipelines)."""
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def split_sentences(text: str) -> List[str]:
    # lightweight sentence split
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]
