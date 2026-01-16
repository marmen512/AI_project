"""
Callback для curriculum learning
"""
from train.callbacks.base import Callback
from core.types import TrainState
from train.curriculum import CurriculumScheduler


class CurriculumCallback(Callback):
    """Callback для curriculum learning"""
    
    def __init__(self, scheduler: CurriculumScheduler):
        """
        Ініціалізація
        
        Args:
            scheduler: CurriculumScheduler instance
        """
        self.scheduler = scheduler
    
    def on_train_start(self, state: TrainState):
        """На початку навчання"""
        pass
    
    def on_epoch_start(self, state: TrainState):
        """На початку епохи"""
        pass
    
    def on_batch_end(self, state: TrainState):
        """Після батча"""
        pass
    
    def on_epoch_end(self, state: TrainState):
        """В кінці епохи - перейти до наступного stage"""
        self.scheduler.on_epoch_end()
    
    def on_train_end(self, state: TrainState):
        """В кінці навчання"""
        pass

