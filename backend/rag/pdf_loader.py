import re
from typing import List, Union
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

def clean_extracted_text(text: str) -> str:
    """
    Cleans raw PDF text extraction: removes multiple spaces, 
    normalizes line breaks, and strips non-printable characters.
    """
    if not text:
        return ""
    # Normalize unicode spaces and unusual control characters
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    # Fix hyphenated words at line breaks (e.g. "dia-\nbetes" -> "diabetes")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Normalize multiple newlines/spaces
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf_documents(pdf_source: Union[str, Path, object], filename: str = "document.pdf") -> List[Document]:
    """
    Extracts text from a PDF file per page, retaining rich metadata:
    page number, source filename, total pages, and character count.
    """
    reader = PdfReader(pdf_source)
    total_pages = len(reader.pages)
    documents = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_extracted_text(raw_text)
        
        if cleaned:
            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": filename,
                    "page": idx + 1,
                    "total_pages": total_pages,
                    "char_count": len(cleaned)
                }
            )
            documents.append(doc)

    return documents

def load_pdf(pdf_source: Union[str, Path, object]) -> str:
    """
    Backward-compatible single-string extractor.
    """
    docs = load_pdf_documents(pdf_source, filename="document.pdf")
    return "\n\n".join([doc.page_content for doc in docs])
