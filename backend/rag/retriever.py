import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_GLOBAL_EMBEDDINGS = None
_GLOBAL_VECTORSTORE = None
_GLOBAL_VECTORSTORE_MTIME = None

def get_embeddings():
    """
    Returns cached HuggingFaceEmbeddings instance (singleton).
    """
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _GLOBAL_EMBEDDINGS

def load_vectorstore(path, force_reload=False):
    """
    Loads FAISS vectorstore from memory cache if available, or reads from local disk.
    """
    global _GLOBAL_VECTORSTORE, _GLOBAL_VECTORSTORE_MTIME
    index_file = os.path.join(path, "index.faiss")
    if not os.path.exists(index_file):
        _GLOBAL_VECTORSTORE = None
        _GLOBAL_VECTORSTORE_MTIME = None
        return None

    mtime = os.path.getmtime(index_file)
    if _GLOBAL_VECTORSTORE is not None and not force_reload and _GLOBAL_VECTORSTORE_MTIME == mtime:
        return _GLOBAL_VECTORSTORE

    embeddings = get_embeddings()
    db = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    _GLOBAL_VECTORSTORE = db
    _GLOBAL_VECTORSTORE_MTIME = mtime
    return db
