"""
Runtime пакет для керування процесом навчання
Єдина точка входу для всіх режимів навчання
"""
from runtime.bootstrap import bootstrap, main
from runtime.modes import TrainingMode
from runtime.resume import find_latest_checkpoint, get_checkpoint_info, should_auto_resume
from runtime.checkpointing import CheckpointManager
from runtime.recursion_state import RecursionState

__all__ = [
    'bootstrap',
    'main',
    'TrainingMode',
    'find_latest_checkpoint',
    'get_checkpoint_info',
    'should_auto_resume',
    'CheckpointManager',
    'RecursionState',
]

