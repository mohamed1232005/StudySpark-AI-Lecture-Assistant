from typing import List, Tuple
import logging
from transformers import pipeline
from config.settings import (
    HF_DEVICE,
    SUMMARIZATION_MODEL,
    SUMMARY_MAXLEN,
    SUMMARY_MINLEN,
)
from .text_processing import chunk_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Single global HF pipeline. No Streamlit calls here.
_summarizer = pipeline(
    task="summarization",
    model=SUMMARIZATION_MODEL,   # e.g., "facebook/bart-large-cnn"
    device=HF_DEVICE,            # -1 CPU, 0 GPU
    use_safetensors=True,
)

def _adaptive_lengths(text: str, max_len: int, min_len: int) -> Tuple[int, int]:
    """Pick safe min/max based on input size to avoid HF errors."""
    wc = max(1, len(text.split()))
    dyn_min = max(8, min(min_len, wc // 3))
    dyn_max = max(dyn_min + 8, min(max_len, 160))   # keep outputs modest
    return dyn_min, dyn_max

def _fallback_summary(text: str, target_words: int = 80) -> str:
    """
    Ultra-safe fallback: return the first ~N words / 2 sentences.
    This guarantees we never return an empty string.
    """
    words = text.split()
    short = " ".join(words[:max(20, min(target_words, len(words)))])
    # light sentence cut to avoid mid-sentence truncation
    for sep in [". ", "! ", "? "]:
        if sep in short and len(short) > 60:
            short = short.split(sep)[0] + "."
            break
    return short.strip()

def summarize_chunk(text: str,
                    max_len: int = SUMMARY_MAXLEN,
                    min_len: int = SUMMARY_MINLEN) -> str:
    dyn_min, dyn_max = _adaptive_lengths(text, max_len, min_len)
    try:
        out = _summarizer(
            text,
            max_length=dyn_max,
            min_length=dyn_min,
            do_sample=False,
        )
        s = (out[0].get("summary_text") or "").strip()
        return s if s else _fallback_summary(text)
    except Exception as e:
        logger.exception(f"[summarize_chunk] falling back due to: {e}")
        return _fallback_summary(text)

def summarize_long_text(text: str,
                        max_len: int = SUMMARY_MAXLEN,
                        min_len: int = SUMMARY_MINLEN) -> str:
    """
    Chunk (smaller to be extra safe) → summarize each → refine;
    if refine fails, return concatenated parts. Never empty.
    """
    # use smaller chunks to avoid edge cases
    chunks: List[str] = chunk_text(text, max_tokens=220)
    if not chunks:
        return ""

    parts: List[str] = []
    for c in chunks:
        s = summarize_chunk(c, max_len=max_len, min_len=min_len)
        if s:
            parts.append(s)

    if not parts:
        # last resort: take a short slice of the original text
        return _fallback_summary(text, target_words=120)

    combined = " ".join(parts)
    try:
        refined = summarize_chunk(combined, max_len=max_len, min_len=min_len)
        return refined if refined else combined
    except Exception as e:
        logger.exception(f"[summarize_long_text] refine failed: {e}")
        return combined

def summarize_many_texts(texts: List[str]) -> List[str]:
    return [summarize_long_text(t) for t in texts]
