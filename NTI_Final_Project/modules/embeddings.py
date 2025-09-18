from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np

try:
    import faiss
except Exception as e:
    raise ImportError("FAISS is required. Install with: pip install faiss-cpu\n" + str(e))

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/e5-small-v2"


class VectorIndex:
    """
    Cosine search via normalized embeddings + FAISS IndexFlatIP.
    Keeps the original text chunks so RAG can reconstruct context.
    """
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.index: faiss.Index | None = None
        self.doc_ids: List[int] = []
        self._chunks: List[str] = []
        self._ntotal: int = 0

    # ---------- internal ----------
    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,       # enables cosine via IP
            show_progress_bar=False,
        )
        return emb.astype("float32")

    def _make_index(self, dim: int) -> faiss.Index:
        return faiss.IndexFlatIP(dim)        # inner product; with L2-normalized inputs = cosine

    # ---------- public ----------
    def build(self, chunks: List[str], doc_ids: List[int]) -> None:
        """Legacy builder (kept for backward compatibility)."""
        if len(chunks) == 0:
            self.index = None
            self.doc_ids = []
            self._chunks = []
            self._ntotal = 0
            return
        if len(chunks) != len(doc_ids):
            raise ValueError("chunks and doc_ids must have the same length")

        corpus = [f"passage: {c}" for c in chunks]  # E5 corpus prefix
        embs = self._encode(corpus)
        self.index = self._make_index(embs.shape[1])
        self.index.add(embs)
        self.doc_ids = list(doc_ids)
        self._chunks = list(chunks)
        self._ntotal = len(chunks)

    def build_index(self, chunks: List[str], doc_ids: List[int]) -> None:
        """Preferred builder name used by the app."""
        self.build(chunks, doc_ids)

    def search(self, query: str, top_k: int = 3) -> Tuple[List[int], List[int]]:
        if self.index is None or self._ntotal == 0:
            return [], []
        q = f"query: {query}"                 # E5 query prefix
        _, I = self.index.search(self._encode([q]), min(top_k, self._ntotal))
        idxs = I[0].tolist()
        mapped = [self.doc_ids[i] for i in idxs]
        return idxs, mapped

    def get_chunks(self, indices: List[int]) -> List[str]:
        return [self._chunks[i] for i in indices]
