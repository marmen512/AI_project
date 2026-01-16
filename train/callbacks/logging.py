"""
Callback для логування метрик
"""
from train.callbacks.base import Callback
from core.types import TrainState
from typing import Optional


class LoggingCallback(Callback):
    """Callback для логування метрик навчання"""
    
    def __init__(self, training_logger = None):
        """
        Ініціалізація
        
        Args:
            training_logger: TRMTrainingLogger instance
        """
        self.training_logger = training_logger
    
    def on_train_start(self, state: TrainState):
        """На початку навчання"""
        pass
    
    def on_epoch_start(self, state: TrainState):
        """На початку епохи"""
        pass
    
    def on_batch_end(self, state: TrainState):
        """Після батча - залогувати метрики"""
        if self.training_logger is not None:
            try:
                self.training_logger.log(
                    step=state.step,
                    epoch=state.epoch,
                    loss=state.loss,
                    recursion_depth=float(state.recursion_depth) if state.recursion_depth > 0 else None,
                    tokens_per_sec=None  # Буде додано з trainer якщо доступно
                )
            except Exception:
                pass  # Не зупиняти навчання через помилки логування
    
    def on_epoch_end(self, state: TrainState):
        """В кінці епохи"""
        pass
    
    def on_train_end(self, state: TrainState):
        """В кінці навчання"""
        if self.training_logger is not None:
            try:
                self.training_logger.save()
            except Exception:
                pass

