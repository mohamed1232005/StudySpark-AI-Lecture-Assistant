from transformers import pipeline
from .embeddings import VectorIndex
from config.settings import QA_MODEL

# Keep QA on CPU for maximum portability.
_qa = pipeline("question-answering", model=QA_MODEL, device=-1)


class RAGQA:
    """
    Retrieve top-k chunks with FAISS, concatenate into a bounded context,
    then run an extractive QA head to produce the final answer.
    """
    def __init__(self, vector_index: VectorIndex):
        self.vi = vector_index

    def answer(self, question: str, top_k: int = 4, max_ctx_chars: int = 2000):
        # 1) Retrieve indices & map back to text
        idxs, doc_ids = self.vi.search(question, top_k=top_k)
        ctx_chunks = self.vi.get_chunks(idxs)
        if not ctx_chunks:
            return "", "", []

        # 2) Assemble bounded context
        context = " ".join(ctx_chunks)[:max_ctx_chars]

        # 3) Extractive QA
        ans = _qa(question=question, context=context)
        return ans.get("answer", ""), context, doc_ids
