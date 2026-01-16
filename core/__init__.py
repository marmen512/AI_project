"""
Core модулі проекту
Централізовані типи, інтерфейси та константи
"""
from core.types import TrainState, EvalState, Batch, Output
from core.constants import TrainingMode, InferenceMode

__all__ = [
    'TrainState',
    'EvalState', 
    'Batch',
    'Output',
    'TrainingMode',
    'InferenceMode'
]

