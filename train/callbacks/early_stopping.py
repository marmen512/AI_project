"""
Callback для early stopping (entropy-driven, в парі з loss)
"""
from train.callbacks.base import Callback
from core.types import TrainState
from typing import Optional
from collections import deque


class EarlyStoppingCallback(Callback):
    """
    Callback для early stopping на основі entropy деградації (в парі з loss)
    Entropy early stop не автономний, він працює в парі з loss
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        entropy_patience: int = 3,
        entropy_min_delta: float = 0.01,
        monitor_loss: bool = True,
        monitor_entropy: bool = True
    ):
        """
        Ініціалізація
        
        Args:
            patience: Patience для loss-based early stopping
            min_delta: Мінімальна зміна loss для вважання покращенням
            entropy_patience: Patience для entropy-based early stopping
            entropy_min_delta: Мінімальна зміна entropy для виявлення деградації
            monitor_loss: Відстежувати loss
            monitor_entropy: Відстежувати entropy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.entropy_patience = entropy_patience
        self.entropy_min_delta = entropy_min_delta
        self.monitor_loss = monitor_loss
        self.monitor_entropy = monitor_entropy
        
        # Відстеження loss
        self.best_loss: Optional[float] = None
        self.loss_no_improve_count = 0
        self.loss_history = deque(maxlen=10)
        
        # Відстеження entropy
        self.best_entropy: Optional[float] = None
        self.entropy_no_improve_count = 0
        self.entropy_history = deque(maxlen=10)
        
        self.should_stop_flag = False
    
    def on_train_start(self, state: TrainState):
        """На початку навчання"""
        self.should_stop_flag = False
        self.loss_no_improve_count = 0
        self.entropy_no_improve_count = 0
    
    def on_epoch_start(self, state: TrainState):
        """На початку епохи"""
        pass
    
    def on_batch_end(self, state: TrainState):
        """Після батча"""
        # Перевірка раннього зупинки відбувається в кінці епохи
        pass
    
    def on_epoch_end(self, state: TrainState):
        """В кінці епохи - перевірити чи потрібно зупинити навчання"""
        # Перевірити loss-based early stopping
        if self.monitor_loss and state.loss > 0:
            if self.best_loss is None or state.loss < (self.best_loss - self.min_delta):
                self.best_loss = state.loss
                self.loss_no_improve_count = 0
            else:
                self.loss_no_improve_count += 1
            
            self.loss_history.append(state.loss)
            
            if self.loss_no_improve_count >= self.patience:
                self.should_stop_flag = True
                return
        
        # Перевірити entropy-based early stopping (в парі з loss)
        if self.monitor_entropy and 'entropy' in state.metadata and state.metadata['entropy']:
            # Обчислити середнє entropy за епоху
            avg_entropy = sum(state.metadata['entropy']) / len(state.metadata['entropy'])
            
            # Entropy деградація: якщо entropy зменшується занадто швидко (порівняно з loss)
            # Це означає що модель стає надто впевненою без покращення loss
            if self.best_entropy is None:
                self.best_entropy = avg_entropy
            else:
                # Деградація: entropy зменшується на більше ніж min_delta
                entropy_delta = self.best_entropy - avg_entropy
                if entropy_delta > self.entropy_min_delta:
                    # Entropy деградує швидше ніж покращується loss
                    self.entropy_no_improve_count += 1
                else:
                    self.entropy_no_improve_count = 0
                    if avg_entropy > self.best_entropy:
                        self.best_entropy = avg_entropy
            
            self.entropy_history.append(avg_entropy)
            
            # Якщо entropy деградує разом з loss (обидва не покращуються)
            if (self.entropy_no_improve_count >= self.entropy_patience and 
                self.loss_no_improve_count > 0):
                self.should_stop_flag = True
                return
    
    def on_train_end(self, state: TrainState):
        """В кінці навчання"""
        pass
    
    def should_stop(self) -> bool:
        """
        Чи потрібно зупинити навчання
        
        Returns:
            True якщо потрібно зупинити
        """
        return self.should_stop_flag
    
    def get_status(self) -> dict:
        """Отримати статус early stopping"""
        return {
            'should_stop': self.should_stop_flag,
            'best_loss': self.best_loss,
            'loss_no_improve_count': self.loss_no_improve_count,
            'best_entropy': self.best_entropy,
            'entropy_no_improve_count': self.entropy_no_improve_count
        }

