import io
import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from backend.config import VECTORSTORE_DIR, DATA_DIR, GOOGLE_API_KEY
from backend.rag.pdf_loader import load_pdf
from backend.rag.text_splitter import split_text
from backend.rag.embeddings import create_vectorstore, append_to_vectorstore
from backend.rag.retriever import load_vectorstore
from backend.rag.chain import query_rag_engine

import os as _os
_os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)

app = FastAPI(
    title="Medical RAG Portal API",
    description="Professional AI-powered medical knowledge retrieval engine.",
    version="1.0.0"
)

# CORS – allow frontend on any port during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve static frontend ───────────────────────────────────────────────────
# __file__ = .../Professional-Medical-RAG-Portal/backend/main.py
# parent   = .../backend/
# parent.parent = .../Professional-Medical-RAG-Portal/  (project root)
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.on_event("startup")
async def startup_warmup():
    """Warm up embeddings model and FAISS vectorstore in memory on server boot."""
    try:
        load_vectorstore(VECTORSTORE_DIR)
    except Exception as e:
        print("Warmup notice:", e)

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


# ─── API Routes ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.2
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list
    status: str
    model: str = ""


@app.get("/api/health")
async def health_check():
    vs_ready = (VECTORSTORE_DIR / "index.faiss").exists()
    return {
        "status": "online",
        "vectorstore_ready": vs_ready,
        "api_key_configured": bool(GOOGLE_API_KEY)
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = load_pdf(io.BytesIO(content))
        chunks = split_text(text)

        if (VECTORSTORE_DIR / "index.faiss").exists():
            append_to_vectorstore(chunks, VECTORSTORE_DIR)
        else:
            create_vectorstore(chunks, VECTORSTORE_DIR)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Successfully processed '{file.filename}' into {len(chunks)} knowledge chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = query_rag_engine(
        query=body.question,
        model_name=body.model,
        temperature=body.temperature,
        top_k=body.top_k
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        status=result["status"],
        model=result.get("model", body.model)
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
        return {"status": "success", "message": "Knowledge base cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
