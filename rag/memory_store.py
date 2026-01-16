"""
In-memory Vector Store для RAG
Просте зберігання embeddings та пошук за cosine similarity
"""
import numpy as np
from typing import List, Optional


class MemoryVectorStore:
    """
    In-memory vector store для зберігання embeddings та текстів
    Використовує cosine similarity для пошуку
    """
    
    def __init__(self):
        """Ініціалізація порожнього vector store"""
        self.vecs: List[np.ndarray] = []  # Список embeddings
        self.texts: List[str] = []  # Список текстів
    
    def add(self, vecs: np.ndarray, texts: List[str]):
        """
        Додати embeddings та тексти до store
        
        Args:
            vecs: Масив embeddings (N x dim)
            texts: Список текстів (N елементів)
        """
        if len(vecs) != len(texts):
            raise ValueError(f"Кількість embeddings ({len(vecs)}) не збігається з кількістю текстів ({len(texts)})")
        
        # Перетворити в список numpy масивів
        if isinstance(vecs, np.ndarray):
            vecs_list = [vecs[i] for i in range(len(vecs))]
        else:
            vecs_list = vecs
        
        self.vecs.extend(vecs_list)
        self.texts.extend(texts)
    
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[str]:
        """
        Пошук k найближчих текстів за cosine similarity
        
        Args:
            query_vec: Query embedding вектор
            k: Кількість результатів для повернення
        
        Returns:
            Список текстів (від найбільш релевантних до найменш)
        """
        if not self.vecs:
            return []
        
        # Обчислити cosine similarity
        # vecs це список numpy масивів, потрібно перетворити в матрицю
        vecs_matrix = np.array(self.vecs)  # (N, dim)
        query_vec = query_vec.flatten()  # Переконатися, що це 1D вектор
        
        # Cosine similarity = dot product (оскільки вектори нормалізовані)
        similarities = np.dot(vecs_matrix, query_vec)
        
        # Знайти індекси k найбільших значень
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Повернути тексти
        return [self.texts[i] for i in top_k_indices]
    
    def __len__(self) -> int:
        """Повернути кількість збережених елементів"""
        return len(self.texts)

