# Lecture Study Assistant (NLP Graduation Project)

**Features**
- PDF ingestion with OCR fallback (scanned slides supported)
- Abstractive summarization (BART/T5)
- Question bank generation (MCQ / True-False / Fill-Blank) from summaries
- RAG Q&A: Sentence-Transformers + FAISS + extractive QA
- Exports: Summary PDF, Questions CSV
- Streamlit UI

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install OS deps:
# - Poppler (for pdf2image)
# - Tesseract OCR
streamlit run pdf-qa-app/app.py
