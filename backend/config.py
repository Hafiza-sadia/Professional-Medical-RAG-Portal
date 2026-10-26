import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Directory Paths
DATA_DIR = BASE_DIR / "backend" / "data"
VECTORSTORE_DIR = BASE_DIR / "backend" / "vectorstore"
EVAL_DIR = BASE_DIR / "backend" / "evaluation"
EVAL_DATA_DIR = EVAL_DIR / "data"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# Ensure all directories exist
for path in [DATA_DIR, VECTORSTORE_DIR, EVAL_DIR, EVAL_DATA_DIR, EVAL_RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Default LLM & Models
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
SUPPORTED_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RAG & Retrieval Thresholds
# For normalized vectors with L2 distance: d^2 = 2 - 2*cos_sim => cos_sim = 1 - (d^2 / 2)
# Distance 1.15 corresponds to ~0.34 cosine similarity
MAX_RETRIEVAL_DISTANCE = 1.25
MIN_RELEVANCE_THRESHOLD = 0.35

# Medical Safety: Emergency Red-Flag Symptoms
EMERGENCY_KEYWORDS = [
    "chest pain", "pressure in chest", "shortness of breath", "difficulty breathing",
    "severe allergic reaction", "anaphylaxis", "throat swelling", "lips swelling",
    "sudden numbness", "facial drooping", "slurred speech", "stroke symptoms",
    "severe bleeding", "unconscious", "loss of consciousness", "coughing blood",
    "suicidal", "suicide", "end my life", "harm myself", "overdose"
]

CLINICAL_DISCLAIMER = (
    "MediQ is an experimental research-oriented AI system designed for informational and "
    "educational exploration of evidence-based medical literature. It does NOT provide clinical "
    "diagnoses, emergency triage, or personalized prescription advice. Always consult a board-certified "
    "physician or licensed healthcare professional for medical conditions."
)
