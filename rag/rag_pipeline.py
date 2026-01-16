"""
RAG Pipeline - функції для побудови RAG системи
"""
from typing import List, Dict, Any, Optional

from rag.embedder import TextEmbedder
from rag.memory_store import MemoryVectorStore
from rag.retriever import RAGRetriever


def build_rag(cfg: Dict[str, Any], documents: Optional[List[str]] = None) -> RAGRetriever:
    """
    Побудувати RAG систему з конфігурації
    
    Args:
        cfg: Конфігурація RAG (з config.yaml)
        documents: Список документів для індексації (опціонально)
    
    Returns:
        RAGRetriever готовий до використання
    """
    # Отримати параметри з конфігурації
    backend = cfg.get('backend', 'memory')  # 'memory' або 'faiss'
    model_name = cfg.get('model_name', 'all-MiniLM-L6-v2')
    
    if backend != 'memory':
        raise ValueError(f"Backend '{backend}' не підтримується. Використовуйте 'memory'")
    
    # Створити embedder
    embedder = TextEmbedder(model_name=model_name)
    
    # Створити vector store
    vector_store = MemoryVectorStore()
    
    # Індексувати документи якщо вони надані
    if documents:
        print(f"📚 Індексація {len(documents)} документів для RAG...")
        embeddings = embedder.encode(documents)
        vector_store.add(embeddings, documents)
        print(f"   ✅ Індексовано {len(vector_store)} документів")
    
    # Створити retriever
    retriever = RAGRetriever(embedder, vector_store)
    
    return retriever

