"""
Базовий клас для evaluators
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch


class BaseEvaluator(ABC):
    """
    Базовий клас для evaluators різних типів задач
    """
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        max_samples: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Оцінити модель на датасеті
        
        Args:
            model: Модель для оцінки
            dataset: Датасет для оцінки
            max_samples: Максимальна кількість прикладів для оцінки (None = всі)
            **kwargs: Додаткові параметри для конкретного evaluator
        
        Returns:
            Словник з метриками оцінки
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Отримати метрики після оцінки
        
        Returns:
            Словник з числовими метриками
        """
        pass
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Відформатувати результати для виводу
        
        Args:
            results: Результати оцінки
        
        Returns:
            Відформатований рядок
        """
        metrics = self.get_metrics()
        lines = ["Evaluation Results:"]
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

