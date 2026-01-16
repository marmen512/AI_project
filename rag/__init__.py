"""
RAG (Retrieval-Augmented Generation) модуль
"""
from rag.embedder import TextEmbedder
from rag.memory_store import MemoryVectorStore
from rag.retriever import RAGRetriever
from rag.rag_dataset_wrapper import RAGDatasetWrapper
from rag.rag_pipeline import build_rag

__all__ = [
    'TextEmbedder',
    'MemoryVectorStore',
    'RAGRetriever',
    'RAGDatasetWrapper',
    'build_rag'
]

