"""
test_rag_pipeline.py — Comprehensive Automated Test Suite for MediQ
Verifies:
1. PDF Loading & Page Extraction
2. Text Splitting & Content Hash Deduplication
3. Embedding Normalization & Cosine Similarity
4. Grounded Document Mode & Abstention on Irrelevant Queries
5. General Clinical Knowledge Mode
6. Red-Flag Emergency Symptom Detection
7. FastAPI Health & Query Endpoints
"""

import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config import DATA_DIR, VECTORSTORE_DIR
from backend.rag.pdf_loader import load_pdf_documents
from backend.rag.text_splitter import split_documents
from backend.rag.embeddings import create_vectorstore
from backend.rag.retriever import retrieve_evidence, l2_to_cosine_similarity
from backend.rag.chain import query_rag_engine, check_emergency_symptoms
from backend.main import app

client = TestClient(app)

def test_emergency_symptoms_detection():
    """Verify red-flag clinical emergency keywords trigger safety alert."""
    alert1 = check_emergency_symptoms("I have severe crushing chest pain and shortness of breath")
    assert alert1 is not None
    assert "URGENT CLINICAL ALERT" in alert1

    alert2 = check_emergency_symptoms("What are the diagnostic criteria for diabetes?")
    assert alert2 is None

def test_l2_to_cosine_conversion():
    """Verify mathematical conversion between L2 distance and cosine similarity."""
    # When vectors are identical (L2 = 0), cosine similarity must be 1.0
    assert l2_to_cosine_similarity(0.0) == 1.0
    # When L2 = sqrt(2) ~ 1.414, cosine similarity must be 0.0
    sim_ortho = l2_to_cosine_similarity(1.4142)
    assert abs(sim_ortho - 0.0) < 0.01

def test_pdf_loading_and_deduplication():
    """Verify PDF per-page extraction and chunk deduplication."""
    pdf_path = DATA_DIR / "diabetes_guide.pdf"
    if not pdf_path.exists():
        pytest.skip("Test PDF not found")

    docs = load_pdf_documents(pdf_path, filename="diabetes_guide.pdf")
    assert len(docs) > 0
    assert "page" in docs[0].metadata
    assert docs[0].metadata["page"] == 1

    chunks = split_documents(docs, chunk_size=600, chunk_overlap=120, deduplicate=True)
    assert len(chunks) > 0
    assert "chunk_id" in chunks[0].metadata
    assert "content_hash" in chunks[0].metadata

def test_retrieval_relevance_and_abstention():
    """Verify that relevant query finds evidence and irrelevant query abstains."""
    # Relevant clinical query
    res_rel = retrieve_evidence("What causes diabetes and high blood sugar?", top_k=3, threshold=0.35)
    assert len(res_rel["chunks"]) > 0
    assert res_rel["sufficient_evidence"] is True
    assert res_rel["max_relevance"] >= 0.40

    # Completely irrelevant non-medical query
    res_irr = retrieve_evidence("How do you configure an Apache reverse proxy on Linux?", top_k=3, threshold=0.35)
    assert res_irr["sufficient_evidence"] is False
    assert res_irr["max_relevance"] < 0.35

def test_query_rag_engine_modes():
    """Verify behavior of Document Mode vs General Mode."""
    # 1. Document Mode with irrelevant query -> Abstention
    res_doc = query_rag_engine("What is the capital of Australia?", mode="document")
    assert res_doc["mode"] == "document"
    assert res_doc["sufficient_evidence"] is False
    assert "Insufficient Document Evidence" in res_doc["answer"]

    # 2. General Mode -> Returns general answer with disclaimer (or missing key notice if no key)
    res_gen = query_rag_engine("What is hypertension?", mode="general")
    assert res_gen["mode"] == "general"
    if "Missing GROQ_API_KEY" in res_gen["answer"]:
        assert res_gen["status"] in ["error", "warning"]
    else:
        assert res_gen["sufficient_evidence"] is True
        assert "General Medical Information Mode" in res_gen["answer"]

def test_api_health_endpoint():
    """Verify FastAPI /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "vectorstore_ready" in data

def test_api_query_endpoint():
    """Verify FastAPI /api/query endpoint."""
    payload = {
        "question": "What is the capital of France?",
        "mode": "document",
        "top_k": 3,
        "threshold": 0.35
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "abstain"
    assert data["sufficient_evidence"] is False
    assert "Insufficient Document Evidence" in data["answer"]
