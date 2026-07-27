"""
PDF text extraction with page-level metadata preservation.
"""
from typing import List, Dict, Any
import fitz  # PyMuPDF


class PDFParser:
    """Extracts text page-by-page from a PDF, preserving page numbers."""

    def extract_text_with_metadata(self, pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        pages_data: List[Dict[str, Any]] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "doc_id": doc_id,
                    "page_number": page_num + 1,
                    "text": text,
                })

        total_pages = len(doc)
        doc.close()
        return pages_data, total_pages

    def get_full_text(self, pdf_path: str) -> str:
        """Convenience helper: returns the whole document as one string (used for ML classification)."""
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return full_text.strip()
