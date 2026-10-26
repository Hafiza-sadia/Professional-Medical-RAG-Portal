#!/usr/bin/env python
"""
run.py — MediQ: Evidence-Grounded Medical RAG System Launcher
Usage: python run.py
"""

import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    print("\n" + "="*65)
    print("  🏥  MediQ — Evidence-Grounded Medical RAG Portal")
    print("  🔬  Research-Oriented Clinical AI Engine & Evaluation Suite")
    print("="*65)
    print("  🌐  Main Web Portal:       http://localhost:8000")
    print("  💬  Evidence-Grounded Chat: http://localhost:8000/chat")
    print("  📤  PDF Document Ingestion: http://localhost:8000/upload")
    print("  📊  Research Benchmark:     http://localhost:8000/research")
    print("  📚  Swagger API Docs:       http://localhost:8000/docs")
    print("="*65 + "\n")

    # Ensure runtime directories exist
    (ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "backend" / "vectorstore").mkdir(parents=True, exist_ok=True)
    (ROOT / "backend" / "evaluation" / "results").mkdir(parents=True, exist_ok=True)

    # Detect python executable (prefer local venv if available)
    venv_python_win = ROOT / "venv" / "Scripts" / "python.exe"
    venv_python_posix = ROOT / "venv" / "bin" / "python"

    if venv_python_win.exists():
        py_cmd = str(venv_python_win)
    elif venv_python_posix.exists():
        py_cmd = str(venv_python_posix)
    else:
        py_cmd = sys.executable

    cmd = [
        py_cmd,
        "-m", "uvicorn",
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
        print("\n\n  ⏹  MediQ Server gracefully stopped. Goodbye!\n")
    except Exception as e:
        print(f"\n  ❌  Failed to launch server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
