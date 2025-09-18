# --- make project root importable ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------

import streamlit as st
st.set_page_config(page_title="StudySpark_Lecture Study Assistant", layout="wide")  # FIRST & ONLY

import pandas as pd

from config.settings import UPLOAD_DIR, OUT_DIR
from modules.pdf_loader import extract_text_with_ocr
from modules.text_processing import clean_text, chunk_text
from modules.summarizer import summarize_long_text
from modules.question_generator import build_question_bank_from_summary
from modules.embeddings import VectorIndex
from modules.rag_qa import RAGQA
from modules.exporters import save_text_as_pdf, save_questions_csv
from modules.utils import ts

st.title("📚 StudySpark")
st.caption("Where lectures spark understanding. Summarize smarter, learn faster, retain longer.")

# add "Evaluate" to your tabs tuple:
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Upload", "Summarize", "Question Bank", "Ask (RAG)", "Export", "Evaluate"]
)

# -------------------------
# Session state containers
# -------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []   # each: {name, text, chunks, summary?}
if "overall_summary" not in st.session_state:
    st.session_state.overall_summary = ""
if "qbank" not in st.session_state:
    st.session_state.qbank = pd.DataFrame()
if "vi" not in st.session_state:
    st.session_state.vi = None

# =======================
# 1) UPLOAD
# =======================
with tab1:
    st.header("1) Upload PDFs (OCR-enabled)")

    files = st.file_uploader(
        "Select one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Uploads") and files:
        st.session_state.docs.clear()

        for f in files:
            dest_path = UPLOAD_DIR / f.name
            dest_path.write_bytes(f.read())

            text = extract_text_with_ocr(dest_path)
            text = clean_text(text)

            chunks = chunk_text(text, max_tokens=300)
            st.session_state.docs.append({"name": f.name, "text": text, "chunks": chunks})

            st.caption(f"{f.name}: extracted {len(text)} chars, {len(chunks)} chunks")

        st.session_state.vi = None  # reset index when uploads change
        st.success(f"Processed {len(st.session_state.docs)} document(s). Go to 'Summarize' tab.")

# =======================
# 2) SUMMARIZE
# =======================
with tab2:
    st.header("2) Per-document + Overall Summaries")

    if not st.session_state.docs:
        st.info("Upload docs first.")
    else:
        # quick self-test (helps debug environments)
        with st.expander("Summarizer self-test (optional)"):
            if st.button("Run self-test"):
                demo = "Streamlit is an open-source Python library for building interactive data apps. It turns scripts into shareable web apps in minutes."
                st.write("Input:", demo)
                st.write("Output:", summarize_long_text(demo))

        overall_parts = []
        for d in st.session_state.docs:
            with st.spinner(f"Summarizing {d['name']} ..."):
                d["summary"] = summarize_long_text(d["text"])

            st.subheader(d["name"])
            st.write(d["summary"] if d["summary"].strip() else "_(no summary)_")
            if d["summary"].strip():
                overall_parts.append(d["summary"].strip())

        st.session_state.overall_summary = " ".join(overall_parts)

        if st.session_state.overall_summary:
            st.success("Overall summary ready below.")
            st.text_area("Overall Summary", st.session_state.overall_summary, height=220)
        else:
            st.warning(
                "No overall summary produced. If your PDFs are scanned, verify Poppler + Tesseract are installed for OCR, "
                "or try a text-based PDF. (The app now falls back to a safe preview if the model fails.)"
            )

# =======================
# 3) QUESTION BANK
# =======================
with tab3:
    st.header("3) Generate Question Bank (MCQ / True-False / Fill-Blank)")

    if not st.session_state.overall_summary:
        st.info("Create summaries first (see 'Summarize' tab).")
    else:
        max_per_type = st.slider("Max items per type", 5, 20, 10)
        if st.button("Build Question Bank"):
            st.session_state.qbank = build_question_bank_from_summary(
                st.session_state.overall_summary, max_per_type=max_per_type
            )
            st.success(f"Created {len(st.session_state.qbank)} questions.")
            st.dataframe(st.session_state.qbank.head(50), use_container_width=True)

# =======================
# 4) ASK (RAG)
# =======================
with tab4:
    st.header("4) Ask Your Lecture (RAG)")

    if not st.session_state.docs:
        st.info("Upload docs first.")
    else:
        if st.session_state.vi is None:
            vi = VectorIndex()
            all_chunks, doc_ids = [], []
            for i, d in enumerate(st.session_state.docs):
                all_chunks.extend(d["chunks"])
                doc_ids.extend([i] * len(d["chunks"]))
            vi.build_index(all_chunks, doc_ids)
            st.session_state.vi = vi

        st.caption(f"Index size: {st.session_state.vi._ntotal} chunks")

        question = st.text_input("Type your question")
        if st.button("Answer") and question.strip():
            rag = RAGQA(st.session_state.vi)
            answer, retrieved_ctx, src_doc_ids = rag.answer(question)

            st.subheader("Answer")
            st.write(answer if answer else "No answer generated.")

            with st.expander("Show retrieved context"):
                st.write(retrieved_ctx)

            src_names = sorted(set(st.session_state.docs[i]["name"] for i in src_doc_ids))
            st.caption(f"Sources: {', '.join(src_names) if src_names else 'N/A'}")

# =======================
# 5) EXPORT
# =======================
with tab5:
    st.header("5) Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.overall_summary:
            if st.button("Download Summary PDF"):
                out_pdf = OUT_DIR / f"summary_{ts()}.pdf"
                save_text_as_pdf(st.session_state.overall_summary, out_pdf)
                st.success(f"Saved: {out_pdf}")
                st.download_button("⬇️ Download PDF", data=open(out_pdf, "rb").read(),
                                   file_name=out_pdf.name, mime="application/pdf")
        else:
            st.info("No summary yet.")

    with col2:
        if not st.session_state.qbank.empty:
            if st.button("Download Question Bank CSV"):
                out_csv = OUT_DIR / f"question_bank_{ts()}.csv"
                save_questions_csv(st.session_state.qbank, out_csv)
                st.success(f"Saved: {out_csv}")
                st.download_button("⬇️ Download CSV", data=open(out_csv, "rb").read(),
                                   file_name=out_csv.name, mime="text/csv")
        else:
            st.info("No question bank yet.")

# =======================
# 6) EVALUATE
# =======================
with tab6:
    st.header("6) Evaluate Summaries (ROUGE / BLEU)")

    from modules.metrics import evaluate_many  # local import to avoid overhead

    if not st.session_state.docs:
        st.info("Upload and summarize documents first.")
    else:
        # 1) Collect reference summaries (two options)
        st.subheader("Provide reference summaries")

        refs_mode = st.radio(
            "Choose input mode",
            ["Type/paste for each document", "Upload CSV (filename, reference)"],
            horizontal=True,
        )

        references = {}
        filenames = [d["name"] for d in st.session_state.docs]

        if refs_mode == "Type/paste for each document":
            for i, d in enumerate(st.session_state.docs):
                key = f"ref_{i}"
                references[d["name"]] = st.text_area(
                    f"Reference for: {d['name']}",
                    value="",
                    height=120,
                    key=key
                )
        else:
            csv_file = st.file_uploader("Upload CSV with columns: filename, reference", type=["csv"])
            if csv_file:
                import pandas as pd
                ref_df = pd.read_csv(csv_file)
                # Normalize column names
                ref_df.columns = [c.strip().lower() for c in ref_df.columns]
                if not {"filename", "reference"}.issubset(set(ref_df.columns)):
                    st.error("CSV must include columns: filename, reference")
                else:
                    references = {row["filename"]: str(row["reference"]) for _, row in ref_df.iterrows()}

        # 2) Build paired lists (pred summaries vs references)
        preds, refs, doc_names = [], [], []
        for d in st.session_state.docs:
            pred = d.get("summary", "").strip()
            ref = references.get(d["name"], "").strip()
            if pred and ref:
                preds.append(pred)
                refs.append(ref)
                doc_names.append(d["name"])

        # 3) Evaluate
        if st.button("Compute Metrics"):
            if not preds:
                st.warning("Please provide at least one reference summary for a summarized document.")
            else:
                rows, macro = evaluate_many(preds, refs)

                # Pretty table
                import pandas as pd
                table = pd.DataFrame(rows, index=doc_names)
                st.subheader("Per-document scores")
                st.dataframe(table.style.format({
                    "rouge1_f": "{:.3f}", "rouge2_f": "{:.3f}", "rougeL_f": "{:.3f}",
                    "rouge1_p": "{:.3f}", "rouge1_r": "{:.3f}",
                    "rouge2_p": "{:.3f}", "rouge2_r": "{:.3f}",
                    "rougeL_p": "{:.3f}", "rougeL_r": "{:.3f}",
                    "bleu": "{:.2f}"
                }), use_container_width=True)

                st.subheader("Macro averages")
                st.write({
                    "ROUGE-1 F1": round(macro["rouge1_f"], 3),
                    "ROUGE-2 F1": round(macro["rouge2_f"], 3),
                    "ROUGE-L F1": round(macro["rougeL_f"], 3),
                    "BLEU": round(macro["bleu"], 2),
                })

                # Download
                csv_bytes = table.to_csv().encode("utf-8")
                st.download_button(
                    "⬇️ Download scores CSV",
                    data=csv_bytes,
                    file_name="evaluation_scores.csv",
                    mime="text/csv"
                )

        st.caption("Tip: write concise, human reference summaries (5–10 sentences) per lecture for fair comparison.")
