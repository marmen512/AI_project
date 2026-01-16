"""
Базовий клас для callbacks
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.types import TrainState


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


class CallbackList:
    """Список callbacks для виклику"""
    
    def __init__(self, callbacks: list[Callback] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        """Додати callback"""
        self.callbacks.append(callback)
    
    def on_train_start(self, state: TrainState):
        """Викликати on_train_start для всіх callbacks"""
        for cb in self.callbacks:
            cb.on_train_start(state)
    
    def on_epoch_start(self, state: TrainState):
        """Викликати on_epoch_start для всіх callbacks"""
        for cb in self.callbacks:
            cb.on_epoch_start(state)
    
    def on_batch_end(self, state: TrainState):
        """Викликати on_batch_end для всіх callbacks"""
        for cb in self.callbacks:
            cb.on_batch_end(state)
    
    def on_epoch_end(self, state: TrainState):
        """Викликати on_epoch_end для всіх callbacks"""
        for cb in self.callbacks:
            cb.on_epoch_end(state)
    
    def on_train_end(self, state: TrainState):
        """Викликати on_train_end для всіх callbacks"""
        for cb in self.callbacks:
            cb.on_train_end(state)

