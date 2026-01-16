"""
Callback для checkpointing
"""
from train.callbacks.base import Callback
from core.types import TrainState
from pathlib import Path
from typing import Optional


class CheckpointCallback(Callback):
    """Callback для збереження checkpoint'ів"""
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_interval: int = 100,
        checkpoint_manager = None  # runtime.checkpointing.CheckpointManager
    ):
        """
        Ініціалізація
        
        Args:
            checkpoint_dir: Директорія для checkpoint'ів
            checkpoint_interval: Інтервал збереження (кожні N батчів)
            checkpoint_manager: CheckpointManager instance (для best checkpointing)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_step = 0
    
    def on_train_start(self, state: TrainState):
        """На початку навчання"""
        pass
    
    def on_epoch_start(self, state: TrainState):
        """На початку епохи"""
        pass
    
    def on_batch_end(self, state: TrainState):
        """Після батча - перевірити чи потрібно зберегти checkpoint"""
        # Перевірка відбувається в trainer, тут тільки відстеження
        pass
    
    def should_save_periodic(self, state: TrainState) -> bool:
        """
        Чи потрібно зберегти періодичний checkpoint
        
        Args:
            state: TrainState
        
        Returns:
            True якщо потрібно зберегти
        """
        if state.step % self.checkpoint_interval == 0 and state.step > self.last_checkpoint_step:
            self.last_checkpoint_step = state.step
            return True
        return False
    
    def should_save_checkpoint(self, state: TrainState) -> bool:
        """
        Чи потрібно зберегти checkpoint
        
        Args:
            state: TrainState
        
        Returns:
            True якщо потрібно зберегти
        """
        # Повертається через метод, щоб можна було перевизначити логіку
        return True
    
    def save_checkpoint(self, state: TrainState, model, optimizer, scheduler):
        """
        Зберегти checkpoint
        
        Args:
            state: TrainState
            model: Модель
            optimizer: Optimizer
            scheduler: Scheduler
        """
        # Цей метод буде викликатися з trainer
        # Реалізація в trainer.save_checkpoint()
        pass
    
    def save_best_checkpoints(self, state: TrainState, accelerator, model, optimizer):
        """
        Зберегти best checkpoints (best_loss, best_entropy)
        Викликається з trainer.save_checkpoint()
        
        Args:
            state: TrainState
            accelerator: Accelerator instance для get_state_dict
            model: Модель
            optimizer: Optimizer
        """
        if self.checkpoint_manager is None:
            return
        
        # Отримати state_dict через accelerator
        try:
            model_state = accelerator.get_state_dict(model) if accelerator else model.state_dict()
            optim_state = accelerator.get_state_dict(optimizer) if accelerator else optimizer.state_dict()
        except Exception as e:
            # Fallback до звичайного state_dict
            model_state = model.state_dict() if hasattr(model, 'state_dict') else None
            optim_state = optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None
        
        if not (model_state and optim_state):
            return
        
        # Best loss (PRIMARY метрика)
        # Використовуємо main_loss якщо доступний, інакше loss
        loss_value = state.main_loss if state.main_loss > 0 else state.loss
        if loss_value > 0:
            try:
                is_new_best = self.checkpoint_manager.save_best_loss(
                    loss=loss_value,
                    model_state_dict=model_state,
                    optimizer_state_dict=optim_state,
                    train_state=state.to_dict()
                )
                if is_new_best:
                    # Логування через accelerator якщо доступно
                    if accelerator and accelerator.is_main_process:
                        accelerator.print(f"💾 Best loss checkpoint saved: loss={loss_value:.4f} (step={state.step})")
            except Exception as e:
                # Не зупиняти навчання через помилки best checkpointing
                if accelerator and accelerator.is_main_process:
                    accelerator.print(f"⚠️ Помилка збереження best loss checkpoint: {e}")
        
        # Best entropy (SECONDARY метрика)
        if 'entropy' in state.metadata and state.metadata['entropy']:
            try:
                avg_entropy = sum(state.metadata['entropy']) / len(state.metadata['entropy'])
                is_new_best = self.checkpoint_manager.save_best_entropy(
                    entropy=avg_entropy,
                    model_state_dict=model_state,
                    optimizer_state_dict=optim_state,
                    train_state=state.to_dict()
                )
                if is_new_best and accelerator and accelerator.is_main_process:
                    accelerator.print(f"💾 Best entropy checkpoint saved: entropy={avg_entropy:.4f}")
            except Exception as e:
                if accelerator and accelerator.is_main_process:
                    accelerator.print(f"⚠️ Помилка збереження best entropy checkpoint: {e}")
    
    def on_epoch_end(self, state: TrainState):
        """В кінці епохи"""
        pass
    
    def on_train_end(self, state: TrainState):
        """В кінці навчання - зберегти фінальний checkpoint"""
        pass

