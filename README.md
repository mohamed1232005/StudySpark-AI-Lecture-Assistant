# StudySpark-AI-Lecture-Assistant

# StudySpark: AI-Powered Lecture Study Assistant

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)  
[![Hugging Face](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  

---

## 📌 Overview

**StudySpark** is a **comprehensive AI-powered lecture assistant** designed to help students, researchers, and professionals process lecture notes in PDF form. It combines **summarization, question bank generation, and Retrieval-Augmented Generation (RAG)-based Q&A** into a single streamlined app.

The application allows you to:

- Upload **multiple PDFs** (supports OCR for scanned docs).  
- Generate **clear, structured summaries** using `facebook/bart-large-cnn`.  
- Build a **Question Bank** with:
  - Multiple-Choice Questions (MCQs with distractors).  
  - True/False Questions.  
  - Fill-in-the-Blank (FIB) Questions.  
- Ask natural language questions about your notes with **RAG (FAISS + SentenceTransformers)**.  
- Export summaries and question banks to **PDF/CSV**.  
- Evaluate summarization quality with **ROUGE / BLEU metrics** against reference summaries.  

---

## 🚀 Features

| Feature                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **PDF Upload**         | Drag & drop multiple PDFs; OCR-enabled via `pdf2image` + `pytesseract`.     |
| **Summarization**      | Uses Hugging Face BART (`facebook/bart-large-cnn`) with chunk-based pipeline.|
| **Question Bank**      | Keyword-based question generation (MCQ, T/F, FIB) via YAKE + regex.         |
| **RAG Q&A**            | Embeds text with `intfloat/e5-small-v2`, indexed in FAISS, retrieved for QA.|
| **Export**             | Summaries → PDF (`reportlab`), Questions → CSV (`pandas`).                  |
| **Evaluation**         | Compute **ROUGE-1, ROUGE-2, ROUGE-L, BLEU** vs reference summaries.         |
| **UI**                 | Streamlit-based multi-tab UI: Upload | Summarize | QBank | Ask | Export.     |

---

## 📂 Project Structure

```
NTI_Final_Project/
├── config/
│   └── settings.py          # Paths, model configs, constants
├── modules/
│   ├── embeddings.py        # FAISS + SentenceTransformer RAG backend
│   ├── exporters.py         # PDF/CSV export utilities
│   ├── pdf_loader.py        # PDF text + OCR extraction
│   ├── question_generator.py# YAKE-based QBank generator
│   ├── rag_qa.py            # RAG pipeline (retrieval + QA)
│   ├── summarizer.py        # Summarization pipeline
│   ├── text_processing.py   # Cleaning, chunking, sentence splitting
│   └── utils.py             # Timestamp helpers
├── pdf-qa-app/
│   └── app.py               # Streamlit frontend
├── data/
│   ├── uploads/             # Raw uploaded PDFs
│   ├── processed/           # Intermediate files
│   └── outputs/             # Exported summaries / CSVs
└── requirements.txt         # Dependencies
```

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/StudySpark-AI-Lecture-Assistant.git
cd StudySpark-AI-Lecture-Assistant
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate     # Linux / Mac
.venv\Scripts\activate.ps1    # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies
- **Poppler** → required by `pdf2image` (PDF → images).  
- **Tesseract OCR** → required for OCR on scanned PDFs.  

**Ubuntu / Debian**
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

**MacOS (Homebrew)**
```bash
brew install poppler tesseract
```

**Windows**
- Install [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/).  
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).  

---

## ▶️ Running the App

From project root:

```bash
streamlit run pdf-qa-app/app.py --server.enableXsrfProtection=false --server.enableCORS=false
```

- Open in browser: [http://localhost:8501](http://localhost:8501)  

---
---

## 🔄 Full Project Flow

The following diagram and step-by-step explanation describe the **end-to-end workflow** of StudySpark.

### 📐 High-Level Architecture

```text
          ┌───────────────┐
          │   PDF Upload  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ OCR Extraction │───► [Tesseract + Poppler] (if scanned PDF)
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │  Text Cleaning│───► remove noise, normalize, preprocess
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Text Chunking │───► ~300 tokens per chunk
          └───────┬───────┘
                  │
                  ├────────────► (1) Summarization
                  │
                  ├────────────► (2) Question Bank Generation
                  │
                  └────────────► (3) Vector Indexing (FAISS)
```

---

### 1. **Document Ingestion**
- User uploads **1 or more PDFs**.  
- For **native PDFs** → extract text using `PyPDF2`.  
- For **scanned PDFs** → convert pages into images via `pdf2image` → run **OCR** with `pytesseract`.  
- All extracted text is normalized with `text_processing.clean_text()`.  

---

### 2. **Chunking**
- Long documents are split into **chunks (~300 tokens)** to avoid exceeding model context windows.  
- Each chunk is stored in memory with a reference to its **document ID**.  

Equation:
```math
chunks = ⌈ total_tokens / chunk_size ⌉
```

---

### 3. **Summarization**
- Each chunk is fed into **BART (`facebook/bart-large-cnn`)** summarizer.  
- Individual summaries are concatenated and **refined again** for a **cleaner overall summary**.  

Pipeline:
```text
Chunk → BART → Chunk Summary
...
All Chunk Summaries → BART → Overall Summary
```

---

### 4. **Question Bank Generation**
- From the overall summary:
  - **YAKE** extracts top keywords (`topk=50`).  
  - For each keyword, generate:
    - **MCQs** (distractors chosen from other keywords).  
    - **True/False** questions (random perturbations).  
    - **Fill-in-the-Blank** (keyword blanked).  
- Output → `pandas.DataFrame` with: *[Question | Type | Options | Answer]*.  

---

### 5. **Vector Indexing (RAG)**
- All chunks are embedded using `intfloat/e5-small-v2`.  
- Embeddings stored in **FAISS Index (IndexFlatIP, cosine similarity)**.  
- Each embedding keeps a pointer back to its **source chunk**.  

Equation:
```math
similarity(query, chunk) = (u · v) / (||u|| ||v||)
```

---

### 6. **Retrieval-Augmented Q&A**
- User types a question.  
- Query is embedded and searched against FAISS.  
- Top-k chunks are retrieved.  
- Retrieved chunks + question → passed into **DistilBERT (SQuAD fine-tuned)** for answer extraction.  

Pipeline:
```text
User Question → Encode → FAISS Search → Retrieve Chunks → DistilBERT → Answer
```

---

### 7. **Export**
- **Summaries** exported as PDF via `reportlab`.  
- **Question Bank** exported as CSV.  
- Filenames timestamped (`utils.ts()`).  

---

### 8. **Evaluation (Optional)**
- Summaries can be evaluated against human references using **ROUGE & BLEU**.  
- Metrics provide quantitative insights into summarization performance.  

---

✅ This modular design ensures **scalability**:  
- You can swap summarization models (e.g., T5, Pegasus).  
- Replace FAISS with **Pinecone / Weaviate** for cloud-scale retrieval.  
- Plug in GPT-style large language models for improved Q&A.  
 
## 🔍 Usage Flow

1. **Upload PDFs**  
   - Supports multiple documents.  
   - OCR fallback if no text is extracted.  

2. **Summarization**  
   - Per-document summary.  
   - Combined overall summary.  

3. **Question Bank**  
   - Auto-generated MCQs, T/F, and FIBs.  
   - Configurable number per type.  

4. **Ask (RAG)**  
   - Query your lecture notes.  
   - Uses FAISS + embeddings for retrieval.  
   - QA model extracts answers.  

5. **Export**  
   - Summaries → PDF.  
   - Questions → CSV.  

---

## 🧠 Technical Details

### Summarization
- Model: `facebook/bart-large-cnn`  
- Pipeline: Hugging Face `transformers.pipeline("summarization")`  
- Chunking: `chunk_text` splits into ~300 tokens.  
- Refinement: Multi-stage summarization (chunk → merge → refine).  

Equation for word count split:
```math
chunks = ⌈ len(words) / max_tokens ⌉
```

### Question Bank
- **Keyword Extraction:** YAKE (`KeywordExtractor(n=1, top=50)`).  
- **MCQ Generation:** Sentence with keyword blanked, distractors from other keywords.  
- **T/F Generation:** Randomly replace keyword with distractor for "False".  
- **FIB Generation:** Replace keyword with `"____"`.  

### RAG Q&A
- Embedding model: `intfloat/e5-small-v2` (SentenceTransformers).  
- Retrieval backend: FAISS `IndexFlatIP` (cosine similarity).  
- QA model: `distilbert-base-cased-distilled-squad`.  
- Query flow:
  ```text
  Question → Encode → Search FAISS → Retrieve Top-k Chunks → QA Model
  ```

Equation for similarity:
```math
cosine(u, v) = (u · v) / (||u|| ||v||)
```

---

## 📊 Evaluation (ROUGE / BLEU)

You can evaluate summarization with gold reference summaries.

### Metrics
| Metric    | Definition |
|-----------|------------|
| **ROUGE-1** | Overlap of unigrams between candidate and reference. |
| **ROUGE-2** | Overlap of bigrams. |
| **ROUGE-L** | Longest common subsequence (LCS). |
| **BLEU**    | Precision-based n-gram metric with brevity penalty. |

### Example Usage
```python
from datasets import load_metric

rouge = load_metric("rouge")
bleu = load_metric("bleu")

candidate = "Deep learning models are powerful."
reference = ["Deep learning is a powerful approach."]

rouge_score = rouge.compute(predictions=[candidate], references=[reference])
bleu_score = bleu.compute(predictions=[candidate.split()], references=[[reference[0].split()]])
print(rouge_score, bleu_score)
```

---

## 📦 Requirements

```
streamlit
torch
transformers
sentence-transformers
faiss-cpu
pandas
numpy
PyPDF2
pdf2image
pytesseract
Pillow
reportlab
yake
```

---

## 🧪 Example Run

### Uploaded PDF
📄 `intro_nlp.pdf` (Lecture notes on NLP basics)

### Output
- **Summary:**
```
Natural Language Processing (NLP) involves preprocessing, embeddings, and transformers...
```

- **Sample Question Bank:**
| Type | Question | Options | Answer |
|------|----------|---------|--------|
| MCQ  | NLP is about _____. Choose the best option. | [Text, Vision, Sound, NLP] | NLP |
| T/F  | True or False: NLP deals with language understanding. | True/False | True |
| FIB  | NLP involves _____. | Language | Language |

- **Q&A:**
```
Q: What is tokenization?
A: The process of splitting text into tokens such as words or subwords.
```



## 🙌 Acknowledgments

- Hugging Face `transformers`  
- SentenceTransformers library  
- Streamlit team  
- FAISS (Facebook AI Similarity Search)  
- YAKE (Yet Another Keyword Extractor)  
- ReportLab for PDF exports  
