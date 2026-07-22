import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from backend.config import VECTORSTORE_DIR
from backend.rag.retriever import load_vectorstore

def query_rag_engine(query: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2, top_k: int = 3):
    """
    Retrieves context from FAISS vectorstore and queries Gemini LLM.
    """
    db = load_vectorstore(VECTORSTORE_DIR)
    
    if db is None:
        return {
            "answer": "No vector knowledge base found. Please upload a medical PDF document first.",
            "sources": [],
            "status": "error"
        }
        
    # Similarity search
    docs_and_scores = db.similarity_search_with_score(query, k=top_k)
    
    sources = []
    for doc, score in docs_and_scores:
        # Lower L2 distance in FAISS means higher similarity
        confidence = round(max(0, 100 - (score * 10)), 1)
        sources.append({
            "content": doc.page_content,
            "confidence": f"{confidence}%",
            "metadata": doc.metadata
        })
        
    context_text = "\n\n---\n\n".join([s["content"] for s in sources])
    
    # 2. Build the Prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a highly capable AI clinical assistant named MediQ.
You must answer the clinical question strictly based on the context provided below.
If the context does not contain the answer, politely state that you cannot answer based on the provided documents.
Do not hallucinate or provide external medical advice.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )
    
    # 3. Get Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": "⚠️ **Missing GROQ_API_KEY**\n\nPlease add your free Groq API key to the `.env` file. You can get one at [console.groq.com](https://console.groq.com/keys).",
            "sources": [],
            "status": "error"
        }
        
    # 4. Fallback Model Setup
    groq_models = [
        model_name,
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    last_error = ""
    for current_model in groq_models:
        try:
            llm = ChatGroq(
                model=current_model,
                groq_api_key=api_key,
                temperature=temperature,
                max_retries=0
            )
            chain = prompt_template | llm
            response = chain.invoke({"context": context_text, "question": query})
            
            return {
                "answer": response.content,
                "sources": sources,
                "status": "success",
                "model": current_model
            }
        except Exception as e:
            last_error = str(e)
            print(f"Groq API Error with model {current_model}: {last_error}")
            # If rate limit (429), try the next model
            if "429" in last_error or "rate_limit" in last_error.lower():
                continue
            else:
                break
                
    return {
        "answer": f"⚠️ **Groq API Error**\n\nFailed to connect to the Groq API. Ensure your key is correct and you have an internet connection.\n\nDetails: `{last_error}`",
        "sources": sources,
        "status": "warning"
    }
