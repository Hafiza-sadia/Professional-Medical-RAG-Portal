import os
from typing import List, Union, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

_EMBEDDINGS_CACHE = {}

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a cached HuggingFaceEmbeddings instance for the requested model.
    Encodes vectors with L2 normalization for accurate cosine similarity calculation.
    """
    global _EMBEDDINGS_CACHE
    if model_name not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_CACHE[model_name]

def create_vectorstore(
    items: Union[List[Document], List[str]],
    path: Union[str, os.PathLike],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Generates embeddings and persists FAISS index to disk.
    Accepts either a list of LangChain Document objects (retaining metadata)
    or raw string chunks.
    """
    embeddings = get_embeddings(model_name)

    if items and isinstance(items[0], Document):
        db = FAISS.from_documents(items, embedding=embeddings)
    else:
        # Fallback for plain strings
        docs = [Document(page_content=str(txt), metadata={"source": "document.pdf", "page": 1}) for txt in items]
        db = FAISS.from_documents(docs, embedding=embeddings)

    db.save_local(str(path))
    from backend.rag.retriever import load_vectorstore
    load_vectorstore(path, force_reload=True, model_name=model_name)
    return db

def append_to_vectorstore(
    new_items: Union[List[Document], List[str]],
    path: Union[str, os.PathLike],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Appends new document chunks to an existing FAISS vector database.
    """
    index_file = os.path.join(path, "index.faiss")
    if os.path.exists(index_file):
        embeddings = get_embeddings(model_name)
        db = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        
        if new_items and isinstance(new_items[0], Document):
            db.add_documents(new_items)
        else:
            docs = [Document(page_content=str(txt), metadata={"source": "document.pdf", "page": 1}) for txt in new_items]
            db.add_documents(docs)

        db.save_local(str(path))
        from backend.rag.retriever import load_vectorstore
        load_vectorstore(path, force_reload=True, model_name=model_name)
        return db
    else:
        return create_vectorstore(new_items, path, model_name=model_name)
