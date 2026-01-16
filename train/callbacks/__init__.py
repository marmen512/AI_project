"""
Callbacks для trainer
Callback-based архітектура для розділення orchestration та logic
"""
from train.callbacks.base import Callback
from train.callbacks.curriculum import CurriculumCallback
from train.callbacks.checkpoint import CheckpointCallback
from train.callbacks.logging import LoggingCallback
from train.callbacks.early_stopping import EarlyStoppingCallback

__all__ = [
    'Callback',
    'CurriculumCallback',
    'CheckpointCallback',
    'LoggingCallback',
    'EarlyStoppingCallback'
]

