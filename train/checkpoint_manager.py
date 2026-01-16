"""
Checkpoint Manager для збереження та завантаження checkpoint'ів
Інтегрований з runtime/checkpointing.py
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from runtime.checkpointing import CheckpointManager as RuntimeCheckpointManager


class CheckpointManager:
    """
    Менеджер для роботи з checkpoint'ами
    Обгортка навколо runtime.checkpointing.CheckpointManager
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last: int = 5):
        """
        Ініціалізація менеджера
        
        Args:
            checkpoint_dir: Директорія для збереження checkpoint'ів
            keep_last: Скільки останніх checkpoint'ів зберігати
        """
        self.runtime_manager = RuntimeCheckpointManager(
            path=checkpoint_dir,
            keep_last=keep_last
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        curriculum_stage: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        filename: Optional[str] = None,
        max_checkpoints: int = 5
    ) -> Path:
        """
        Зберегти checkpoint
        
        Args:
            model: Модель для збереження
            optimizer: Optimizer для збереження
            epoch: Номер епохи
            step: Номер кроку
            loss: Поточний loss
            curriculum_stage: Інформація про curriculum stage (опціонально)
            metrics: Додаткові метрики (опціонально)
            filename: Ім'я файлу (якщо None - автоматичне)
            max_checkpoints: Максимальна кількість checkpoint'ів для збереження
        
        Returns:
            Шлях до збереженого checkpoint'у
        """
        extra = {
            'epoch': epoch,
            'loss': loss,
            'curriculum_stage': curriculum_stage,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Використати runtime manager для збереження
        return self.runtime_manager.save(step, model, optimizer, extra)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Завантажити checkpoint
        
        Args:
            model: Модель для завантаження
            optimizer: Optimizer для завантаження
            checkpoint_path: Шлях до checkpoint'у (None = останній)
        
        Returns:
            Інформація про checkpoint
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            return checkpoint.get('extra', {})
        else:
            # Завантажити останній
            step = self.runtime_manager.load_latest(model, optimizer)
            return {'step': step}
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Знайти останній checkpoint
        
        Returns:
            Шлях до останнього checkpoint'у або None
        """
        return self.runtime_manager.latest()

