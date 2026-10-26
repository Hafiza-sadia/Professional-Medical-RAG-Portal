import re
from typing import List, Dict, Any, Set
from backend.rag.verification import verify_citations

def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())

def compute_retrieval_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    is_out_of_domain: bool = False,
    sufficient_evidence: bool = True
) -> Dict[str, float]:
    """
    Computes Information Retrieval metrics:
    - Recall@k: fraction of target keywords retrieved
    - Precision@k: fraction of retrieved passages containing relevant concepts
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant chunk
    - Abstention Accuracy: 1.0 if system correctly abstains on negative/out-of-scope queries
    """
    if is_out_of_domain:
        # Correct behavior for OOD/unsupported queries is ABSTENTION (sufficient_evidence = False)
        correct_abstention = 1.0 if not sufficient_evidence else 0.0
        return {
            "recall_at_k": 1.0 if not sufficient_evidence else 0.0,
            "precision_at_k": 1.0 if not sufficient_evidence else 0.0,
            "hit_at_k": 1.0 if not sufficient_evidence else 0.0,
            "mrr": 1.0 if not sufficient_evidence else 0.0,
            "abstention_accuracy": correct_abstention
        }

    if not retrieved_chunks:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "abstention_accuracy": 0.0
        }

    k = len(retrieved_chunks)
    all_retrieved_tokens = set()
    relevant_chunk_count = 0
    first_relevant_rank = 0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        content_tokens = set(_tokenize(chunk.get("content", "")))
        all_retrieved_tokens.update(content_tokens)

        # Check if chunk contains at least one target keyword
        has_keyword = any(kw.lower() in content_tokens or kw.lower() in chunk.get("content", "").lower() 
                          for kw in expected_keywords)
        if has_keyword:
            relevant_chunk_count += 1
            if first_relevant_rank == 0:
                first_relevant_rank = rank

    # Recall: Fraction of expected keywords present in retrieved pool
    keywords_found = sum(
        1 for kw in expected_keywords
        if any(kw.lower() in c.get("content", "").lower() for c in retrieved_chunks)
    )
    recall = keywords_found / len(expected_keywords) if expected_keywords else 1.0
    precision = relevant_chunk_count / k if k > 0 else 0.0
    hit = 1.0 if first_relevant_rank > 0 else 0.0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank > 0 else 0.0

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "hit_at_k": round(hit, 4),
        "mrr": round(mrr, 4),
        "abstention_accuracy": 1.0 if sufficient_evidence else 0.0
    }

def compute_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Estimates answer faithfulness / grounding score:
    Measures the proportion of content assertions in the answer
    that are lexically grounded in the retrieved context.
    """
    if not answer or not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    context_tokens = set(_tokenize(combined_context))

    # Split answer into simple sentence units
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    grounded_sentences = 0
    for sent in sentences:
        words = _tokenize(sent)
        # Filter stop words
        content_words = [w for w in words if len(w) > 3]
        if not content_words:
            grounded_sentences += 1
            continue

        overlap = sum(1 for w in content_words if w in context_tokens)
        overlap_ratio = overlap / len(content_words)
        if overlap_ratio >= 0.45:
            grounded_sentences += 1

    return round(grounded_sentences / len(sentences), 4)

def compute_citation_metrics(answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes citation fidelity metrics:
    - citation_count: total citations in the answer
    - citation_precision: percentage of citations verified against retrieved evidence
    """
    ver = verify_citations(answer, context_chunks)
    return {
        "citation_count": ver["total_citations"],
        "citation_precision": ver["verification_rate"],
        "suspicious_count": len(ver.get("suspicious_patterns", []))
    }

def compute_token_f1(generated: str, gold: str) -> float:
    """
    Standard QA evaluation metric: Token-level F1 overlap between generated answer and gold reference.
    """
    gen_tokens = _tokenize(generated)
    gold_tokens = _tokenize(gold)
    if not gen_tokens or not gold_tokens:
        return 0.0

    common = set(gen_tokens).intersection(set(gold_tokens))
    if not common:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(f1, 4)
