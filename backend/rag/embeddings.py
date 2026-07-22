import os
from langchain_community.vectorstores import FAISS
from backend.rag.retriever import get_embeddings, load_vectorstore

def create_vectorstore(chunks, path):
    """
    Generates sentence embeddings and persists FAISS index.
    """
    embeddings = get_embeddings()
    db = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )
    db.save_local(str(path))
    # Force reload memory cache
    load_vectorstore(path, force_reload=True)
    return db

def append_to_vectorstore(new_chunks, path):
    """
    Appends new chunks to an existing FAISS vector database.
    """
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(path, "index.faiss")):
        db = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_texts(new_chunks)
        db.save_local(str(path))
        load_vectorstore(path, force_reload=True)
        return db
    else:
        return create_vectorstore(new_chunks, path)
