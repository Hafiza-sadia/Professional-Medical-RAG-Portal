
# MediQ — GitHub Codespaces Setup & Run Guide

## 1. Project Overview

MediQ is an AI-powered Medical Retrieval-Augmented Generation (RAG) chatbot.

**Technologies used:
- Python and FastAPI
- LangChain
- FAISS vector database
- Sentence Transformers
- Groq API
- Medical PDF document processing

The application allows users to upload medical PDFs and ask questions based on the uploaded documents.

## 2. Open the Project in GitHub Codespaces

1. Open the repository:
   https://github.com/Hafiza-sadia/Professional-Medical-RAG-Portal

2. Click the green **Code** button.
3. Select **Codespaces**.
4. Create a new Codespace, or open the existing Codespace.
5. Wait for the VS Code environment to finish loading.

## 3. Install Dependencies

Open the terminal in Codespaces and run:

```bash
pip install -r requirements.txt
```

Check that Uvicorn is installed:

```bash
python -m uvicorn --version
```

## 4. Configure the Groq API Key

1. Get an API key from:
   https://console.groq.com/keys

2. Create a file named `.env` in the project's root directory.

3. Add your key:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Replace `your_actual_groq_api_key` with your real key.

**Security:** Never upload `.env` or your API key to GitHub. Ensure `.env` is listed in `.gitignore`.

## 5. Update the Groq Model

The old model `llama-3.3-70b-versatile` may no longer be available for your account.

1. Open this file:

   `backend/rag/chain.py`

2. Find the default model name:

```python
model_name: str = "llama-3.3-70b-versatile"
```

3. Replace it with:

```python
model_name: str = "openai/gpt-oss-120b"
```

Save the file. Check Groq's current supported models if this model is unavailable:
https://console.groq.com/docs/models

## 6. Run the Application

From the repository root, run:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Keep this terminal running while using the application.

If port 8000 is detected, open the **Ports** tab in Codespaces and click the link for port 8000 to open the application in your browser.

> Use the command above if `python run.py` reports that Uvicorn is missing, even though Uvicorn is installed.

## 7. Upload a Medical PDF

1. Open the MediQ application in the browser.
2. Navigate to the PDF upload page.
3. Select a medical PDF document.
4. Upload the document and wait for processing to finish.
5. Open the chat page.
6. Ask a question related to the uploaded document.

For example:

- What are the main symptoms discussed in this document?
- Summarize the key points of this medical PDF.
- Which treatment options are mentioned in the document?

The answers depend on the uploaded document and the application's retrieval and generation pipeline.

## 8. Troubleshooting

### Missing GROQ_API_KEY

- Check that `.env` is in the repository root.
- Verify that the key is correct.
- Restart the application after changing environment settings.

### No vector knowledge base found

Upload a medical PDF first. The application needs document content to build or populate its retrieval knowledge base.

### Groq model not found (404)

Check `backend/rag/chain.py` and replace the old model ID with a currently supported Groq model.

### Port 8000 is not opening

- Confirm the server is running in the terminal.
- Open the Codespaces **Ports** tab.
- Make sure port 8000 is forwarded.
- Open the forwarded port in your browser.

### Dependency installation problems

Use the project's Python environment and run:

```bash
python -m pip install -r requirements.txt
```

If a package fails to install, check the Python version and the package's compatibility requirements.

## 9. Upload This Guide to GitHub

1. Create a new file in the repository root named `GITHUB_CODESPACES_GUIDE.md`.
2. Paste this guide into the file.
3. Save the file.
4. Open the Source Control panel in Codespaces.
5. Stage the new file.
6. Enter a commit message, such as:

   `Add GitHub Codespaces setup guide`

7. Commit and push the change to GitHub.

Before pushing, check that `.env`, API keys, private credentials, and unnecessary generated files are not included.

## 10. Final Verification Checklist

- [ ] Dependencies install successfully.
- [ ] `.env` contains a valid Groq API key and is not committed.
- [ ] The configured Groq model is currently supported.
- [ ] The FastAPI application starts.
- [ ] Port 8000 opens in the browser.
- [ ] A medical PDF uploads successfully.
- [ ] A question returns a relevant answer.
- [ ] The response is checked against the source document.

**Important:** A page loading successfully does not prove that the entire AI pipeline works. Test a PDF upload and a real question after fixing the model configuration before describing the project as fully working.
