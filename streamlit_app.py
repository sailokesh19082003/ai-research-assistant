import streamlit as st
import os

# Load secrets (set in Streamlit Cloud's "Secrets" box) into environment
# variables BEFORE importing anything that reads config/settings.py.
for key in ["GEMINI_API_KEY", "OPENAI_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

import sys
sys.path.insert(0, os.path.dirname(__file__))

import uuid
from src.database.base import init_db, SessionLocal
from src.database.models import Document
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.vector_store.manager import get_vector_store
from src.ml.isolated_client import predict_category_isolated
from src.rag.qa_chain import RAGQuestionAnswering
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.analytics.metrics import get_system_analytics

st.set_page_config(page_title="DocuMind - AI Research Assistant", layout="wide")
init_db()

st.title("📚 DocuMind — AI Research & Knowledge Assistant")
st.caption("Upload PDFs, auto-classify them, search, ask questions with citations, summarize, and compare.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload", "❓ Ask / Search", "📝 Summarize", "⚖️ Compare", "📊 Analytics"])

with tab1:
    st.header("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Process Document"):
        doc_id = str(uuid.uuid4())
        os.makedirs("data/raw_documents", exist_ok=True)
        file_path = f"data/raw_documents/{doc_id}_{uploaded.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        db = SessionLocal()
        doc = Document(doc_id=doc_id, file_name=uploaded.name, file_path=file_path, processing_status="PENDING")
        db.add(doc); db.commit()

        with st.spinner("Parsing, classifying, chunking, indexing..."):
            parser = PDFParser()
            pages_data, total_pages = parser.extract_text_with_metadata(file_path, doc_id)
            vector_store = get_vector_store()
            full_text = "\n".join(p["text"] for p in pages_data)
            category, confidence = predict_category_isolated(full_text)
            chunker = DocumentChunker()
            chunks = chunker.create_chunks(pages_data, file_name=uploaded.name)
            vector_store.index_chunks(chunks)

            doc.total_pages = total_pages
            doc.total_chunks = len(chunks)
            doc.category = category
            doc.category_confidence = int(confidence * 100)
            doc.processing_status = "PROCESSED"
            db.add(doc); db.commit()
        db.close()
        st.success(f"Processed! Category: **{category}** ({confidence:.0%} confidence) — {total_pages} pages, {len(chunks)} chunks")

    st.divider()
    st.subheader("Uploaded Documents")
    db = SessionLocal()
    docs = db.query(Document).all()
    if docs:
        for d in docs:
            st.write(f"**{d.file_name}** — {d.processing_status} — {d.category} — {d.total_chunks} chunks — `{d.doc_id[:8]}...`")
    else:
        st.info("No documents uploaded yet.")
    db.close()

with tab2:
    st.header("Ask a question (RAG with citations)")
    query = st.text_input("Your question")
    if st.button("Ask") and query:
        db = SessionLocal()
        rag = RAGQuestionAnswering()
        with st.spinner("Retrieving and answering..."):
            result = rag.answer_question(db=db, query=query)
        db.close()
        st.write("**Answer:**")
        st.write(result["answer"])
        st.write("**Sources:**")
        for c in result["citations"]:
            st.write(f"- {c['document']} (page {c['page']}) — relevance {c['relevance_score']}")

with tab3:
    st.header("Summarize a document")
    db = SessionLocal()
    docs = db.query(Document).filter(Document.processing_status == "PROCESSED").all()
    options = {f"{d.file_name} ({d.doc_id[:8]})": d.doc_id for d in docs}
    db.close()
    if options:
        choice = st.selectbox("Choose a document", list(options.keys()))
        if st.button("Summarize"):
            summarizer = DocumentSummarizer()
            with st.spinner("Summarizing..."):
                result = summarizer.summarize_document(options[choice], choice)
            st.write(result.get("summary", result.get("error")))
    else:
        st.info("Upload and process a document first.")

with tab4:
    st.header("Compare documents")
    db = SessionLocal()
    docs = db.query(Document).filter(Document.processing_status == "PROCESSED").all()
    options = {f"{d.file_name} ({d.doc_id[:8]})": d.doc_id for d in docs}
    db.close()
    chosen = st.multiselect("Choose 2 or more documents", list(options.keys()))
    if st.button("Compare") and len(chosen) >= 2:
        comparator = DocumentComparator()
        doc_infos = [{"doc_id": options[c], "file_name": c} for c in chosen]
        with st.spinner("Comparing..."):
            result = comparator.compare_documents(doc_infos)
        st.write(result.get("comparison", result.get("error")))
    elif chosen:
        st.warning("Pick at least 2 documents.")

with tab5:
    st.header("System Analytics")
    db = SessionLocal()
    stats = get_system_analytics(db)
    db.close()
    st.json(stats)
