from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path
import pandas as pd

def save_text_as_pdf(text: str, out_path: Path):
    out_path = Path(out_path)
    c = canvas.Canvas(str(out_path), pagesize=letter)
    w, h = letter
    t = c.beginText(40, h-40)
    t.setFont("Helvetica", 12)
    for para in text.split("\n"):
        line = para.strip()
        while len(line) > 95:
            t.textLine(line[:95]); line = line[95:]
        t.textLine(line)
        t.textLine("")
    c.drawText(t); c.save()

def save_questions_csv(df: pd.DataFrame, out_path: Path):
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
