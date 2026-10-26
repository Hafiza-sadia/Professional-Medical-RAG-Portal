import os
import re
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from backend.rag.embeddings import get_embeddings
from backend.config import VECTORSTORE_DIR, MIN_RELEVANCE_THRESHOLD, DEFAULT_EMBEDDING_MODEL

_GLOBAL_VECTORSTORE = None
_GLOBAL_VECTORSTORE_MTIME = None
_GLOBAL_VECTORSTORE_MODEL = None

def l2_to_cosine_similarity(l2_distance: float) -> float:
    """
    Converts Euclidean (L2) distance of normalized embeddings to Cosine Similarity:
    Since ||u|| = ||v|| = 1: L2^2 = 2 - 2*cos(theta) => cos(theta) = 1 - (L2^2 / 2).
    """
    cosine_sim = 1.0 - (float(l2_distance) ** 2) / 2.0
    return round(max(0.0, min(1.0, cosine_sim)), 4)

def get_relevance_tier(similarity: float) -> str:
    """Classifies similarity score into an interpretable qualitative tier."""
    if similarity >= 0.70:
        return "High Relevance"
    elif similarity >= 0.45:
        return "Moderate Relevance"
    else:
        return "Low / Marginal"

def _jaccard_similarity(text1: str, text2: str) -> float:
    """Computes word-level Jaccard similarity between two texts."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def deduplicate_chunks(docs_and_scores: List[Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Filters out near-duplicate chunks (e.g., overlapping paragraphs or repeat pages)
    retaining the highest-scoring unique evidence passages.
    """
    unique_results = []

    for doc, score in docs_and_scores:
        content = doc.page_content.strip()
        metadata = dict(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}
        cosine_sim = l2_to_cosine_similarity(score)
        
        doc_name = metadata.get("source", "Medical Document")
        page_num = metadata.get("page", 1)
        chunk_id = metadata.get("chunk_id", f"{doc_name}_p{page_num}")

        # Check for near-duplicates among already selected results
        is_duplicate = False
        for existing in unique_results:
            # Check identical content hash or high lexical similarity (> 65%)
            if metadata.get("content_hash") and metadata.get("content_hash") == existing["metadata"].get("content_hash"):
                is_duplicate = True
                break
            
            sim = _jaccard_similarity(content, existing["content"])
            if sim > 0.65 or (content in existing["content"] or existing["content"] in content):
                is_duplicate = True
                # If from same document and different page, combine page info
                if doc_name == existing["document"] and page_num != existing["page"]:
                    if isinstance(existing["page"], list):
                        if page_num not in existing["page"]:
                            existing["page"].append(page_num)
                    else:
                        existing["page"] = [existing["page"], page_num]
                break

        if not is_duplicate:
            unique_results.append({
                "content": content,
                "document": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "relevance_score": cosine_sim,
                "relevance_tier": get_relevance_tier(cosine_sim),
                "l2_distance": round(float(score), 4),
                "metadata": metadata
            })

        if len(unique_results) >= top_k:
            break

    return unique_results

def load_vectorstore(path: Optional[Any] = None, force_reload: bool = False, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Loads FAISS vectorstore from memory cache if available, or reads from local disk.
    """
    global _GLOBAL_VECTORSTORE, _GLOBAL_VECTORSTORE_MTIME, _GLOBAL_VECTORSTORE_MODEL
    path = path or VECTORSTORE_DIR
    index_file = os.path.join(path, "index.faiss")
    
    if not os.path.exists(index_file):
        _GLOBAL_VECTORSTORE = None
        _GLOBAL_VECTORSTORE_MTIME = None
        _GLOBAL_VECTORSTORE_MODEL = None
        return None

    mtime = os.path.getmtime(index_file)
    if (_GLOBAL_VECTORSTORE is not None and not force_reload 
            and _GLOBAL_VECTORSTORE_MTIME == mtime 
            and _GLOBAL_VECTORSTORE_MODEL == model_name):
        return _GLOBAL_VECTORSTORE

    embeddings = get_embeddings(model_name)
    db = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    _GLOBAL_VECTORSTORE = db
    _GLOBAL_VECTORSTORE_MTIME = mtime
    _GLOBAL_VECTORSTORE_MODEL = model_name
    return db

def retrieve_evidence(
    query: str,
    top_k: int = 3,
    threshold: float = MIN_RELEVANCE_THRESHOLD,
    path: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> Dict[str, Any]:
    """
    Executes similarity search, converts distances to normalized cosine similarities,
    deduplicates overlapping chunks, and evaluates evidence sufficiency.
    """
    db = load_vectorstore(path=path, model_name=model_name)
    if db is None:
        return {
            "chunks": [],
            "sufficient_evidence": False,
            "max_relevance": 0.0,
            "status": "no_vectorstore"
        }

    # Fetch 2x top_k to allow deduplication headroom
    raw_results = db.similarity_search_with_score(query, k=top_k * 2)
    if not raw_results:
        return {
            "chunks": [],
            "sufficient_evidence": False,
            "max_relevance": 0.0,
            "status": "empty"
        }

    cleaned_chunks = deduplicate_chunks(raw_results, top_k=top_k)
    max_relevance = cleaned_chunks[0]["relevance_score"] if cleaned_chunks else 0.0
    sufficient_evidence = max_relevance >= threshold

    return {
        "chunks": cleaned_chunks,
        "sufficient_evidence": sufficient_evidence,
        "max_relevance": max_relevance,
        "status": "ok"
    }
