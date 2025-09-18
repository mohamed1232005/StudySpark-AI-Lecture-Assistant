from typing import List
from pathlib import Path
import PyPDF2
import pytesseract
from PIL import Image
from io import BytesIO

# pdf2image needs poppler installed on system. If not available, we handle gracefully.
try:
    from pdf2image import convert_from_path
    _PDF2IMAGE = True
except Exception:
    _PDF2IMAGE = False

def _extract_text_pypdf2(pdf_path: Path) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
                text += t + "\n"
            except Exception:
                continue
    return text.strip()

def _ocr_pages_with_pdf2image(pdf_path: Path) -> str:
    if not _PDF2IMAGE:
        raise RuntimeError("pdf2image/poppler not found. Install poppler or use system package.")
    images: List[Image.Image] = convert_from_path(str(pdf_path))
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()

def extract_text_with_ocr(pdf_path: Path) -> str:
    """Try text extraction; if empty, fall back to OCR."""
    pdf_path = Path(pdf_path)
    text = _extract_text_pypdf2(pdf_path)
    if text:
        return text
    # Fallback to OCR
    return _ocr_pages_with_pdf2image(pdf_path)
