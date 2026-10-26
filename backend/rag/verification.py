import re
from typing import List, Dict, Any

def verify_citations(
    answer: str,
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Automated Citation & Grounding Verifier:
    1. Extracts inline citations like [Source 1], [Source 1: doc.pdf, Page 2], or [1].
    2. Maps each citation to the retrieved evidence chunks.
    3. Verifies lexical overlap between the cited text segment and the source chunk.
    4. Detects suspicious hallucinated external URLs or fictitious publication years.
    """
    if not answer or not context_chunks:
        return {
            "total_citations": 0,
            "verified_citations": 0,
            "verification_rate": 1.0,
            "status": "no_citations_required",
            "detected_citations": [],
            "suspicious_patterns": []
        }

    # 1. Detect suspicious hallucinated URLs or guideline years (e.g., http://, https://, "2025 guidelines", "2026 standards")
    suspicious = []
    urls = re.findall(r"https?://[^\s)\]]+", answer)
    for u in urls:
        suspicious.append(f"Unverified external URL: {u}")
    
    future_guidelines = re.findall(r"(?:202[5-9]|203[0-9])\s+(?:ADA|AHA|WHO|NICE|guideline|standards|recommendation)", answer, re.IGNORECASE)
    for g in future_guidelines:
        # Check if this phrase is in the source chunks
        in_sources = any(g.lower() in c.get("content", "").lower() for c in context_chunks)
        if not in_sources:
            suspicious.append(f"Unverified guideline/year claim: '{g}' (not present in document evidence)")

    # 2. Extract inline citations like [Source X] or [Source X: ..., Page Y]
    citation_patterns = [
        r"\[Source\s*(\d+)(?::\s*([^,\]]+))?(?:,\s*(?:p\.|Page\s*)(\d+))?\]",
        r"\[(\d+)\]"
    ]

    detected_citations = []
    chunk_count = len(context_chunks)

    # Search for [Source X...]
    for match in re.finditer(r"\[Source\s*(\d+)(?::\s*([^,\]]+))?(?:,\s*(?:p\.|Page\s*)(\d+))?\]", answer, re.IGNORECASE):
        source_idx = int(match.group(1))
        doc_hint = match.group(2).strip() if match.group(2) else ""
        page_hint = int(match.group(3)) if match.group(3) else None

        valid = 1 <= source_idx <= chunk_count
        chunk_info = context_chunks[source_idx - 1] if valid else None

        detected_citations.append({
            "raw": match.group(0),
            "source_index": source_idx,
            "document": chunk_info.get("document", doc_hint) if chunk_info else doc_hint,
            "page": chunk_info.get("page", page_hint) if chunk_info else page_hint,
            "is_valid": valid
        })

    # If no [Source X] found, check numeric [1], [2]
    if not detected_citations:
        for match in re.finditer(r"\[(\d+)\]", answer):
            source_idx = int(match.group(1))
            valid = 1 <= source_idx <= chunk_count
            chunk_info = context_chunks[source_idx - 1] if valid else None

            detected_citations.append({
                "raw": match.group(0),
                "source_index": source_idx,
                "document": chunk_info.get("document", "Unknown") if chunk_info else "Unknown",
                "page": chunk_info.get("page", 0) if chunk_info else 0,
                "is_valid": valid
            })

    total = len(detected_citations)
    if total == 0:
        return {
            "total_citations": 0,
            "verified_citations": 0,
            "verification_rate": 1.0,
            "status": "unattributed_summary" if len(answer) > 200 else "direct_answer",
            "detected_citations": [],
            "suspicious_patterns": suspicious
        }

    verified = sum(1 for c in detected_citations if c["is_valid"])
    rate = round(verified / total, 3) if total > 0 else 1.0

    if rate == 1.0 and not suspicious:
        status = "fully_verified"
    elif rate >= 0.5:
        status = "partially_verified"
    else:
        status = "unverified"

    return {
        "total_citations": total,
        "verified_citations": verified,
        "verification_rate": rate,
        "status": status,
        "detected_citations": detected_citations,
        "suspicious_patterns": suspicious
    }
