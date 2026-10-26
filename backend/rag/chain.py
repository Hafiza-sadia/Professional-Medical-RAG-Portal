import os
import time
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from backend.config import (
    GROQ_API_KEY,
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
    VECTORSTORE_DIR,
    EMERGENCY_KEYWORDS,
    CLINICAL_DISCLAIMER,
    MIN_RELEVANCE_THRESHOLD
)
from backend.rag.retriever import retrieve_evidence
from backend.rag.verification import verify_citations

def check_emergency_symptoms(query: str) -> Optional[str]:
    """Scans the query for red-flag symptoms requiring emergency clinical escalation."""
    q_lower = query.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in q_lower:
            return (
                "### 🚨 URGENT CLINICAL ALERT\n\n"
                "Your inquiry contains symptoms that may indicate a critical, life-threatening emergency "
                f"(*detected: '{kw}'*).\n"
                "**Please contact local emergency medical services immediately (e.g., 911, 112, 999) "
                "or proceed to the nearest hospital emergency department.** Do not delay medical care.\n\n"
                "---\n\n"
            )
    return None

def _get_groq_llm(model_name: str, temperature: float = 0.2):
    """Initializes ChatGroq client with error handling."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "Missing GROQ_API_KEY"

    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=temperature,
        max_retries=1,
        max_tokens=1200
    ), None

def generate_document_grounded_answer(
    query: str,
    context_text: str,
    context_chunks: List[Dict[str, Any]] = None,
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Generates an answer strictly grounded in the retrieved document context.
    Enforces verifiable inline citations and abstains on unsupported details.
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are MediQ, a rigorous, evidence-grounded AI clinical retrieval assistant.
Your answers must be grounded strictly and exclusively in the provided document CONTEXT below.

MANDATORY CLINICAL GROUNDING RULES:
1. EVIDENCE GROUNDING: Ground every factual assertion, clinical recommendation, or definition solely on the provided CONTEXT passages. Do not extrapolate beyond what is documented.
2. MANDATORY IN-TEXT CITATIONS: After EVERY clinical assertion or key statement, cite the corresponding source index in square brackets, for example: [Source 1] or [Source 2]. Every statement must clearly link to its source chunk.
3. BALANCED GROUNDING & ABSTENTION:
   - Provide an answer using the relevant facts present in the CONTEXT PASSAGES (e.g., definitions, clinical features, symptoms, risk factors, or management).
   - If the CONTEXT PASSAGES do NOT contain relevant information to address the clinical question at all, or only cover entirely unrelated topics, state:
     "Based on the provided documents, there is insufficient evidence to answer this question."
   - Do NOT extrapolate, guess, or use outside pre-trained knowledge to supplement unmentioned details.
4. CITATION INTEGRITY:
   - Never cite fictitious guidelines, external URLs, DOIs, or speculative publication years (e.g. 2025/2026 guidelines) unless verbatim in the context.
   - Every citation must correspond to an actual [Source X] in the context.
5. GUIDELINE CURRENCY HONESTY:
   - Do NOT describe any guideline or standard as "the latest", "current", or "most recent" based on general knowledge. You may only describe a document as such if the uploaded context itself states it.
   - If publication status is unclear from the context, state: "Publication status could not be confirmed from the provided document."
6. STRUCTURE:
   - Use clear sections or bullet points where helpful.
   - Keep answers clear, direct, and objectively clinical.

CONTEXT PASSAGES:
{context}

CLINICAL INQUIRY:
{question}

EVIDENCE-GROUNDED ANSWER (with [Source X] citations):
"""
    )

    models_to_try = [model_name] + [m for m in SUPPORTED_GROQ_MODELS if m != model_name]
    last_error = ""

    for m in models_to_try:
        llm, err = _get_groq_llm(m, temperature)
        if not llm:
            return {"answer": f"⚠️ **Groq Configuration Error**: {err}. Please add your key to `.env`.", "model": m, "status": "error"}

        try:
            chain = prompt_template | llm
            res = chain.invoke({"context": context_text, "question": query})
            raw_answer = (res.content or "").strip()

            # Guard: skip to next model if empty output
            if not raw_answer:
                last_error = "Model returned empty output. Trying fallback model."
                continue

            # Verification of citations
            verification = verify_citations(raw_answer, context_chunks or [])
            is_abstention = "insufficient evidence" in raw_answer.lower()

            # Citation Guarantee: If answer was generated but model omitted inline [Source X] tags,
            # append an explicit Referenced Document Evidence block so the user always has verified source details.
            if not is_abstention and context_chunks and verification.get("total_citations", 0) == 0:
                evidence_lines = ["\n\n---\n**📄 Referenced Document Evidence:**"]
                for i, c in enumerate(context_chunks, 1):
                    doc_n = c.get("document", "Document")
                    p_num = c.get("page", 1)
                    p_str = f"Page {', '.join(map(str, p_num))}" if isinstance(p_num, list) else f"Page {p_num}"
                    snippet = c.get("content", "").strip()[:180].replace("\n", " ")
                    evidence_lines.append(f"- **[Source {i}] {doc_n} ({p_str})**: *\"{snippet}...\"*")
                raw_answer += "\n".join(evidence_lines)
                verification = verify_citations(raw_answer, context_chunks)

            return {
                "answer": raw_answer,
                "model": m,
                "status": "success",
                "verification": verification,
                "is_abstention": is_abstention
            }
        except Exception as e:
            last_error = str(e)
            _retryable = ["429", "rate_limit", "empty", "output", "unreachable", "overloaded", "timeout"]
            if any(k in last_error.lower() for k in _retryable):
                continue
            break

    return {
        "answer": f"⚠️ **LLM Generation Warning**: Groq API connection issue. Details: `{last_error}`",
        "model": model_name,
        "status": "warning",
        "verification": {"status": "error", "verification_rate": 0.0},
        "is_abstention": False
    }

def generate_general_mode_answer(
    query: str,
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Generates an answer using general clinical pre-trained knowledge with explicit
    epistemic labeling, avoiding hallucinated publication citations.
    """
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are MediQ, an educational medical assistant operating in GENERAL CLINICAL INFORMATION MODE.
The user is querying general clinical concepts without an uploaded document.

GUIDELINES:
1. Provide a clear, evidence-aligned medical summary based on broad clinical consensus.
2. Structure your response with the following sections where applicable:
   - Definition & Overview
   - Common Causes / Risk Factors
   - Clinical Features / Symptoms
   - General Management Principles
   - Safety Precautions & When to Seek Care

3. EPISTEMIC HONESTY - STRICTLY ENFORCE ALL OF THE FOLLOWING:
   a) NEVER claim that a guideline, recommendation, or standard is "the latest", "current", "updated", or "most recent" unless you have verified it against an official source in this session. If live verification is unavailable, you MUST explicitly state: "Publication status could not be confirmed at this time."
   b) NEVER fabricate specific publication years, edition numbers, version codes, DOIs, URLs, or journal citations. If you cannot verify a specific citation, do NOT include it.
   c) ALWAYS clearly distinguish between:
      - General medical knowledge (AI pre-training): frame as "It is generally understood in clinical practice that..."
      - Established consensus: frame as "Clinical consensus broadly suggests..." or "Widely accepted practice recommends..."
      - Verified evidence: ONLY available in Document Mode with uploaded PDFs.
   d) Do NOT present AI-generated medical summaries as authoritative, current, or policy-level guidance. Clearly label this output as general educational information only.

4. SCOPE REMINDER - End every response with this exact paragraph:
   "Note: This summary reflects general medical knowledge from AI pre-trained consensus and has NOT been verified against live official sources (e.g., WHO, ADA, AHA, NICE). Publication status of any referenced guidelines could not be confirmed. For verified, citation-backed clinical guidance, please switch to Document Mode and upload the relevant official guidelines."

QUESTION:
{question}

GENERAL CLINICAL SUMMARY:
"""
    )

    models_to_try = [model_name] + [m for m in SUPPORTED_GROQ_MODELS if m != model_name]
    last_error = ""

    for m in models_to_try:
        llm, err = _get_groq_llm(m, temperature)
        if not llm:
            return {"answer": f"⚠️ **Groq Configuration Error**: {err}.", "model": m, "status": "error"}

        try:
            chain = prompt_template | llm
            res = chain.invoke({"question": query})
            content = (res.content or "").strip()
            if not content:
                last_error = "Model returned empty output. Trying fallback model."
                continue

            # Epistemic safety guard: hedge unverified guideline currency and future years
            import re
            content = re.sub(
                r"\b(the\s+)?(latest|most\s+recent|current)\s+(guidelines?|recommendations?|standards?)\b",
                r"established clinical \3 (publication currency unverified)",
                content,
                flags=re.IGNORECASE
            )
            content = re.sub(
                r"\b(202[5-9]|203[0-9])\s+(guidelines?|standards?|recommendations?)\b",
                r"clinical \2 (year unconfirmed against official source)",
                content,
                flags=re.IGNORECASE
            )

            disclaimed_answer = (
                "🌐 **General Clinical Information Mode (Parametric AI Knowledge)**\n"
                "*Notice: This overview is generated from general AI pre-trained weights and is **NOT** "
                "verified against any uploaded document in your knowledge base. No external citations are asserted.*\n\n"
                f"{content}\n\n"
                "---\n"
                f"<small>*{CLINICAL_DISCLAIMER}*</small>"
            )
            return {
                "answer": disclaimed_answer,
                "model": m,
                "status": "success",
                "verification": {
                    "total_citations": 0,
                    "verified_citations": 0,
                    "verification_rate": 1.0,
                    "status": "parametric_knowledge_mode"
                },
                "is_abstention": False
            }
        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "rate_limit" in last_error.lower():
                continue
            break

    return {
        "answer": f"⚠️ **LLM Generation Warning**: Groq API connection issue. Details: `{last_error}`",
        "model": model_name,
        "status": "warning",
        "verification": {"status": "error", "verification_rate": 0.0},
        "is_abstention": False
    }

def query_rag_engine(
    query: str,
    mode: str = "document",
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    top_k: int = 3,
    threshold: float = MIN_RELEVANCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Main RAG orchestrator supporting:
    - Evidence retrieval with deduplication
    - Grounding threshold evaluation & abstention
    - Inline citation enforcement and verification
    - Dual answering modes ('document' vs 'general')
    - Emergency red-flag safety checks
    - Latency benchmarking
    """
    start_total = time.time()
    emergency_warning = check_emergency_symptoms(query)

    # ------------------ GENERAL MODE ------------------
    if mode == "general":
        t0 = time.time()
        gen_res = generate_general_mode_answer(query, model_name=model_name, temperature=temperature)
        gen_time = round((time.time() - t0) * 1000, 1)

        final_answer = (emergency_warning or "") + gen_res["answer"]
        return {
            "answer": final_answer,
            "sources": [],
            "mode": "general",
            "status": gen_res["status"],
            "model": gen_res["model"],
            "sufficient_evidence": True,
            "verification": gen_res.get("verification", {}),
            "is_abstention": False,
            "latency": {
                "retrieval_ms": 0.0,
                "generation_ms": gen_time,
                "total_ms": round((time.time() - start_total) * 1000, 1)
            }
        }

    # ------------------ DOCUMENT MODE ------------------
    t_ret_0 = time.time()
    retrieval_result = retrieve_evidence(query=query, top_k=top_k, threshold=threshold)
    retrieval_time = round((time.time() - t_ret_0) * 1000, 1)

    if retrieval_result["status"] == "no_vectorstore":
        return {
            "answer": (
                "📁 **No Documents Uploaded Yet**\n\n"
                "Your document knowledge base is currently empty. Please upload one or more medical PDFs "
                "via the [Upload Page](/upload) to enable evidence-grounded responses, or switch the mode to "
                "**General Clinical Information Mode** above to query general medical knowledge."
            ),
            "sources": [],
            "mode": "document",
            "status": "no_documents",
            "model": model_name,
            "sufficient_evidence": False,
            "verification": {"status": "no_documents", "verification_rate": 0.0},
            "is_abstention": True,
            "latency": {
                "retrieval_ms": retrieval_time,
                "generation_ms": 0.0,
                "total_ms": round((time.time() - start_total) * 1000, 1)
            }
        }

    sources = retrieval_result["chunks"]

    # Check evidence sufficiency / threshold abstention
    if not retrieval_result["sufficient_evidence"]:
        abstain_message = (
            "⚠️ **Insufficient Document Evidence**\n\n"
            "The uploaded medical documents do not contain sufficient evidence to answer this inquiry reliably. "
            f"(Peak document relevance: `{retrieval_result['max_relevance']*100:.1f}%`, below clinical grounding threshold `{threshold*100:.0f}%`).\n\n"
            "To safeguard against medical hallucinations and ungrounded advice, MediQ abstains from generating "
            "an answer without direct textual support.\n\n"
            "💡 *Tip: You can switch to **General Medical Information Mode** to consult general AI medical knowledge.*"
        )
        if emergency_warning:
            abstain_message = emergency_warning + abstain_message

        return {
            "answer": abstain_message,
            "sources": sources,
            "mode": "document",
            "status": "abstain",
            "model": model_name,
            "sufficient_evidence": False,
            "verification": {
                "total_citations": 0,
                "verified_citations": 0,
                "verification_rate": 1.0,
                "status": "correct_abstention"
            },
            "is_abstention": True,
            "latency": {
                "retrieval_ms": retrieval_time,
                "generation_ms": 0.0,
                "total_ms": round((time.time() - start_total) * 1000, 1)
            }
        }

    # Build context with clear indexed sources [Source 1], [Source 2]...
    context_snippets = []
    for idx, s in enumerate(sources, start=1):
        doc_label = f"Source {idx}: {s['document']} (Page {s['page']}) [Relevance: {s['relevance_score']*100:.1f}%]"
        context_snippets.append(f"[{doc_label}]\n{s['content']}")
    context_text = "\n\n---\n\n".join(context_snippets)

    t_gen_0 = time.time()
    gen_res = generate_document_grounded_answer(
        query=query,
        context_text=context_text,
        context_chunks=sources,
        model_name=model_name,
        temperature=temperature
    )
    gen_time = round((time.time() - t_gen_0) * 1000, 1)

    final_answer = (emergency_warning or "") + gen_res["answer"]

    return {
        "answer": final_answer,
        "sources": sources,
        "mode": "document",
        "status": gen_res["status"],
        "model": gen_res["model"],
        "sufficient_evidence": True,
        "verification": gen_res.get("verification", {}),
        "is_abstention": gen_res.get("is_abstention", False),
        "latency": {
            "retrieval_ms": retrieval_time,
            "generation_ms": gen_time,
            "total_ms": round((time.time() - start_total) * 1000, 1)
        }
    }
