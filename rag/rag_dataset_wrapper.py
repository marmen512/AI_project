"""
RAG Dataset Wrapper
Обгортає базовий датасет та додає RAG контекст до input
"""
from torch.utils.data import Dataset
from typing import Any, Dict, Optional

from rag.retriever import RAGRetriever


class RAGDatasetWrapper(Dataset):
    """
    Wrapper для датасету, який додає RAG контекст до кожного input
    """
    
    def __init__(self, base_dataset: Dataset, rag_retriever: RAGRetriever, k: int = 5):
        """
        Ініціалізація wrapper
        
        Args:
            base_dataset: Базовий датасет (повинен повертати словник з 'input' або tuple)
            rag_retriever: RAGRetriever для пошуку контекстів
            k: Кількість контекстів для додавання
        """
        self.base = base_dataset
        self.rag = rag_retriever
        self.k = k
    
    def __len__(self) -> int:
        """Повернути розмір базового датасету"""
        return len(self.base)
    
    def __getitem__(self, idx: int) -> Any:
        """
        Отримати елемент з доданим RAG контекстом
        
        Args:
            idx: Індекс елемента
        
        Returns:
            Елемент датасету з доданим RAG контекстом у input
        """
        item = self.base[idx]
        
        # Визначити input текст в залежності від формату датасету
        if isinstance(item, dict):
            input_text = item.get('input', item.get('context', ''))
            if input_text:
                # Отримати релевантні контексти
                contexts = self.rag.retrieve(input_text, k=self.k)
                
                # Додати контексти до input
                if contexts:
                    context_text = " ".join(contexts)
                    # Додати контекст перед оригінальним input
                    item = item.copy()  # Зробити копію, щоб не міняти оригінал
                    item['input'] = f"{context_text} {input_text}"
            
            return item
        elif isinstance(item, tuple):
            # Якщо датасет повертає tuple (input, output)
            input_data, output_data = item
            
            # Спробувати витягнути текст з input
            if isinstance(input_data, str):
                input_text = input_data
            else:
                # Якщо input це tensor, не можемо додати RAG
                return item
            
            # Отримати релевантні контексти
            contexts = self.rag.retrieve(input_text, k=self.k)
            
            # Додати контексти до input
            if contexts:
                context_text = " ".join(contexts)
                input_data = f"{context_text} {input_text}"
            
            return (input_data, output_data)
        else:
            # Непідтримуваний формат
            return item

