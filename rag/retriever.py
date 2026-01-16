"""
RAG Retriever для пошуку релевантних контекстів
"""
from typing import List, Optional
import numpy as np

from rag.embedder import TextEmbedder
from rag.memory_store import MemoryVectorStore


class RAGRetriever:
    """
    Retriever для RAG - поєднує embedder та vector store
    """
    
    def __init__(self, embedder: TextEmbedder, vector_store: MemoryVectorStore):
        """
        Ініціалізація retriever
        
        Args:
            embedder: TextEmbedder для створення embeddings
            vector_store: MemoryVectorStore для зберігання та пошуку
        """
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Пошук релевантних контекстів для запиту
        
        Args:
            query: Текст запиту
            k: Кількість контекстів для повернення
        
        Returns:
            Список релевантних контекстів (від найбільш до найменш релевантних)
        """
        if len(self.vector_store) == 0:
            return []
        
        # Створити embedding для запиту
        query_vec = self.embedder.encode(query)
        
        # Пошук у vector store
        contexts = self.vector_store.search(query_vec, k=k)
        
        return contexts

