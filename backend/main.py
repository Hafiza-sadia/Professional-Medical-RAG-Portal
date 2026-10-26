import io
import shutil
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import (
    VECTORSTORE_DIR,
    DATA_DIR,
    GROQ_API_KEY,
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
    EVAL_RESULTS_DIR,
    EVAL_DATA_DIR,
    MIN_RELEVANCE_THRESHOLD
)
from backend.rag.pdf_loader import load_pdf_documents
from backend.rag.text_splitter import split_documents
from backend.rag.embeddings import create_vectorstore, append_to_vectorstore
from backend.rag.retriever import load_vectorstore
from backend.rag.chain import query_rag_engine
from backend.evaluation.run_experiments import run_all_experiments

app = FastAPI(
    title="MediQ — Evidence-Grounded Medical RAG Portal API",
    description="Research-oriented AI application for evidence-grounded clinical QA with dual answering modes and empirical benchmark evaluation.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Frontend Mounts
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.on_event("startup")
async def startup_warmup():
    """Warm up vectorstore cache if an index exists."""
    try:
        if (VECTORSTORE_DIR / "index.faiss").exists():
            load_vectorstore(VECTORSTORE_DIR)
    except Exception as e:
        print("Warmup notice:", e)

# ------------------ Frontend Routes ------------------

@app.get("/", include_in_schema=False)
async def serve_home():
    return FileResponse(str(FRONTEND_DIR / "pages" / "index.html"))

@app.get("/chat", include_in_schema=False)
async def serve_chat():
    return FileResponse(str(FRONTEND_DIR / "pages" / "chat.html"))

@app.get("/upload", include_in_schema=False)
async def serve_upload():
    return FileResponse(str(FRONTEND_DIR / "pages" / "upload.html"))

@app.get("/about", include_in_schema=False)
async def serve_about():
    return FileResponse(str(FRONTEND_DIR / "pages" / "about.html"))

@app.get("/research", include_in_schema=False)
async def serve_research():
    return FileResponse(str(FRONTEND_DIR / "pages" / "research.html"))

# ------------------ API Schemas ------------------

class QueryRequest(BaseModel):
    question: str
    mode: str = Field(default="document", description="'document' (grounded) or 'general' (broad)")
    model: str = Field(default=DEFAULT_GROQ_MODEL)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=10)
    threshold: float = Field(default=MIN_RELEVANCE_THRESHOLD, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    sources: list
    mode: str = "document"
    status: str
    model: str = ""
    sufficient_evidence: bool = True
    verification: Optional[Dict[str, Any]] = None
    is_abstention: bool = False
    latency: Dict[str, float] = {}

# ------------------ API Endpoints ------------------

@app.get("/api/health")
async def health_check():
    vs_ready = (VECTORSTORE_DIR / "index.faiss").exists()
    files = list(DATA_DIR.glob("*.pdf")) if vs_ready else []
    return {
        "status": "online",
        "service": "MediQ Medical RAG",
        "vectorstore_ready": vs_ready,
        "document_count": len(files),
        "api_key_configured": bool(GROQ_API_KEY or os.getenv("GROQ_API_KEY")),
        "default_model": DEFAULT_GROQ_MODEL,
        "supported_models": SUPPORTED_GROQ_MODELS
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    content = await file.read()
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        docs = load_pdf_documents(io.BytesIO(content), filename=file.filename)
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract readable text from this PDF.")

        chunks = split_documents(docs, chunk_size=600, chunk_overlap=120, deduplicate=True)

        if (VECTORSTORE_DIR / "index.faiss").exists():
            append_to_vectorstore(chunks, VECTORSTORE_DIR)
        else:
            create_vectorstore(chunks, VECTORSTORE_DIR)

        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(docs),
            "chunks": len(chunks),
            "message": f"Successfully indexed '{file.filename}' ({len(docs)} pages, {len(chunks)} deduplicated evidence chunks)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = query_rag_engine(
        query=body.question,
        mode=body.mode,
        model_name=body.model,
        temperature=body.temperature,
        top_k=body.top_k,
        threshold=body.threshold
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        mode=result.get("mode", body.mode),
        status=result["status"],
        model=result.get("model", body.model),
        sufficient_evidence=result.get("sufficient_evidence", True),
        verification=result.get("verification", {}),
        is_abstention=result.get("is_abstention", False),
        latency=result.get("latency", {})
    )

@app.get("/api/vectorstore/status")
async def vectorstore_status():
    exists = (VECTORSTORE_DIR / "index.faiss").exists()
    files = list(DATA_DIR.glob("*.pdf")) if exists else []
    return {
        "ready": exists,
        "documents": [f.name for f in files],
        "document_count": len(files)
    }

@app.delete("/api/vectorstore/reset")
async def reset_vectorstore():
    try:
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        load_vectorstore(VECTORSTORE_DIR, force_reload=True)
        return {"status": "success", "message": "Knowledge base reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Research & Evaluation API ------------------

@app.get("/api/evaluation/latest")
async def get_latest_evaluation():
    summary_path = EVAL_RESULTS_DIR / "benchmark_summary.json"
    report_path = EVAL_RESULTS_DIR / "BENCHMARK_REPORT.md"

    if not summary_path.exists():
        return {
            "has_results": False,
            "message": "No evaluation run recorded yet. Click 'Run Benchmark Experiments' to start."
        }

    with open(summary_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    report_text = ""
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

    return {
        "has_results": True,
        "summaries": summaries,
        "report_markdown": report_text
    }

@app.post("/api/evaluation/run")
async def run_evaluation_benchmark():
    try:
        results = run_all_experiments()
        return {
            "status": "success",
            "message": "Benchmark evaluation executed successfully across all configurations.",
            "configurations_tested": len(results) if results else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")

@app.get("/api/evaluation/dataset")
async def get_evaluation_dataset():
    data_path = EVAL_DATA_DIR / "medical_eval_benchmark.json"
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark dataset not found.")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"total": len(data), "samples": data}
