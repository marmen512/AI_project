"""
Text Embedder для RAG
Використовує sentence-transformers для створення embeddings
"""
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

from typing import List, Union
import numpy as np


class TextEmbedder:
    """
    Клас для створення embeddings тексту
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Ініціалізація embedder
        
        Args:
            model_name: Назва моделі sentence-transformers
                       (default: "all-MiniLM-L6-v2" - швидка та легка)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers не встановлено. "
                "Встановіть: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Створити embeddings для текстів
        
        Args:
            texts: Текст або список текстів
        
        Returns:
            Масив embeddings (нормалізованих)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Створити embeddings та нормалізувати
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

