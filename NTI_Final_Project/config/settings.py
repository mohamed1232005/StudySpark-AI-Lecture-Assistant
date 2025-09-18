from pathlib import Path
import torch

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "outputs"

for p in (UPLOAD_DIR, PROC_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Models
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # has safetensors

QA_MODEL = "distilbert-base-cased-distilled-squad"
# config/settings.py
EMBEDDING_MODEL = "intfloat/e5-small-v2"   # has safetensors
EMBED_DIM = 384                            # e5-small-v2 dimension


# Device helpers
CUDA_AVAILABLE = torch.cuda.is_available()
HF_DEVICE = 0 if CUDA_AVAILABLE else -1  # pipelines: 0=gpu, -1=cpu

# Chunking
CHUNK_TOKENS = 300  # ~ tokens≈words for our simple splitter
SUMMARY_MAXLEN = 120
SUMMARY_MINLEN = 40

# YAKE keyword settings
YAKE_TOPK = 50

# ---

# # ### how to run
# # 1) create/activate venv  
# # 2) `pip install -r requirements.txt`  
# # 3) ensure **Poppler** + **Tesseract** are installed (OS-level)  
# # 4) `streamlit run pdf-qa-app/app.py`

# # that’s it. you’ve got a full app.

# # **my opinion:** this layout is tight, production-like, and mirrors your Colab logic. if you hit any local OCR deps issue, test first with non-scanned PDFs (PyPDF2 path) — the app will still work and you can add Poppler/Tesseract after.
# # ::contentReference[oaicite:0]{index=0}
