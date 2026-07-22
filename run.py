#!/usr/bin/env python
"""
run.py — MediQ Professional Medical RAG Portal launcher
Usage: python run.py
"""

import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    print("\n" + "="*60)
    print("  🏥  MediQ AI Portal — Professional Medical RAG System")
    print("="*60)
    print("  🌐  Starting server on http://localhost:8000")
    print("  📄  Pages: / | /chat | /upload | /about")
    print("  📚  API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    # Ensure data & vectorstore dirs exist
    (ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "backend" / "vectorstore").mkdir(parents=True, exist_ok=True)

    # Try to reuse existing Medical-RAG-Chatbot venv first
    venv_candidates = [
        ROOT.parent / "Medical-RAG-Chatbot" / "venv" / "Scripts" / "uvicorn.exe",
        ROOT / "venv" / "Scripts" / "uvicorn.exe",
        ROOT.parent / "Medical-RAG-Chatbot" / "venv" / "bin" / "uvicorn",
        ROOT / "venv" / "bin" / "uvicorn",
    ]

    uvicorn_cmd = "uvicorn"
    for candidate in venv_candidates:
        if candidate.exists():
            uvicorn_cmd = str(candidate)
            break

    cmd = [
        uvicorn_cmd,
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", str(ROOT / "backend"),
        "--log-level", "info"
    ]

    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except KeyboardInterrupt:
        print("\n\n  ⏹  MediQ Portal stopped. Goodbye!\n")
    except FileNotFoundError:
        print(f"\n  ❌  uvicorn not found. Please run:")
        print(f"      pip install fastapi uvicorn python-multipart")
        sys.exit(1)

if __name__ == "__main__":
    main()
