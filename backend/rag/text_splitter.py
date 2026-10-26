import hashlib
import re
from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def _normalize_chunk_text(text: str) -> str:
    """Normalizes text for hash-based deduplication."""
    return re.sub(r"\s+", " ", text.strip().lower())

def split_documents(
    docs: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    deduplicate: bool = True
) -> List[Document]:
    """
    Splits Document objects into semantic chunks while maintaining rich page/source metadata,
    generating unique chunk IDs, and deduplicating identical or near-identical text blocks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )

    all_chunks = []
    seen_hashes = set()

    for doc in docs:
        sub_chunks = splitter.split_text(doc.page_content)
        source = doc.metadata.get("source", "document.pdf")
        page = doc.metadata.get("page", 1)

        for chunk_idx, chunk_text in enumerate(sub_chunks):
            chunk_clean = chunk_text.strip()
            if not chunk_clean or len(chunk_clean) < 20:
                continue

            # Exact hash deduplication
            norm_text = _normalize_chunk_text(chunk_clean)
            content_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()[:16]

            if deduplicate and content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            meta = dict(doc.metadata)
            meta.update({
                "chunk_id": f"{source}_p{page}_c{chunk_idx+1}",
                "chunk_index": chunk_idx + 1,
                "chunk_size": len(chunk_clean),
                "content_hash": content_hash
            })

            all_chunks.append(Document(page_content=chunk_clean, metadata=meta))

    return all_chunks

def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> List[str]:
    """
    Backward-compatible string splitter with standard RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )
    return splitter.split_text(text)
