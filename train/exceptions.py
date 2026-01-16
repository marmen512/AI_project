"""
Кастомні винятки для навчання
"""
from typing import Optional


class TrainingError(Exception):
    """Базовий клас для помилок навчання"""
    pass


class DatasetNotFoundError(TrainingError):
    """Датасет не знайдено"""
    def __init__(self, dataset_path: str, message: Optional[str] = None):
        self.dataset_path = dataset_path
        if message is None:
            message = f"Датасет не знайдено: {dataset_path}"
        super().__init__(message)


class CheckpointError(TrainingError):
    """Помилка роботи з checkpoint"""
    def __init__(self, checkpoint_path: str, message: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        if message is None:
            message = f"Помилка checkpoint: {checkpoint_path}"
        super().__init__(message)


class ModelConfigError(TrainingError):
    """Помилка конфігурації моделі"""
    pass


class ModelLoadError(TrainingError):
    """Помилка завантаження моделі"""
    def __init__(self, model_path: str, message: Optional[str] = None):
        self.model_path = model_path
        if message is None:
            message = f"Не вдалося завантажити модель: {model_path}"
        super().__init__(message)




















