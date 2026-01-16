"""
Інтерфейси та абстракції для проекту
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

from core.types import TrainState, EvalState


class Callback(ABC):
    """Базовий клас для callbacks"""
    
    @abstractmethod
    def on_train_start(self, state: TrainState):
        """Викликається на початку навчання"""
        pass
    
    @abstractmethod
    def on_epoch_start(self, state: TrainState):
        """Викликається на початку епохи"""
        pass
    
    @abstractmethod
    def on_batch_end(self, state: TrainState):
        """Викликається після обробки батча"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, state: TrainState):
        """Викликається в кінці епохи"""
        pass
    
    @abstractmethod
    def on_train_end(self, state: TrainState):
        """Викликається в кінці навчання"""
        pass


class CheckpointCallback(Callback):
    """Callback для збереження checkpoint'ів"""
    
    @abstractmethod
    def should_save_checkpoint(self, state: TrainState) -> bool:
        """Чи потрібно зберегти checkpoint"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, state: TrainState, path: Path):
        """Зберегти checkpoint"""
        pass


class EarlyStoppingCallback(Callback):
    """Callback для early stopping"""
    
    @abstractmethod
    def should_stop(self, state: TrainState) -> bool:
        """Чи потрібно зупинити навчання"""
        pass


class DatasetFilter(ABC):
    """Абстракція для фільтрації датасету"""
    
    @abstractmethod
    def should_include(self, item: Dict[str, Any]) -> bool:
        """Чи потрібно включити item в датасет"""
        pass


class Evaluator(ABC):
    """Абстракція для evaluator"""
    
    @abstractmethod
    def evaluate(self, model, dataset) -> EvalState:
        """Оцінити модель"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Отримати метрики"""
        pass

