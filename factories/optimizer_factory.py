"""
Factory для створення optimizers та LR schedulers
"""
from typing import Dict, Any, Optional, Tuple
import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from adam_atan2_pytorch import MuonAdamAtan2
except ImportError:
    MuonAdamAtan2 = None


def build_optimizer(
    cfg: Dict[str, Any],
    model: torch.nn.Module
) -> Tuple[Optimizer, Optional[LambdaLR]]:
    """
    Створити optimizer та LR scheduler з конфігурації
    
    Args:
        cfg: Конфігурація з ключами:
            - training.optimizer: Тип optimizer ("adamw", "muon") (default: "adamw")
            - training.learning_rate: Learning rate (default: 1e-4)
            - training.weight_decay: Weight decay (default: 1.0)
            - training.muon_learning_rate: Muon learning rate (для MuonAdamAtan2)
            - training.warmup_steps: Кількість warmup кроків (default: 2000)
        model: Модель для оптимізації
    
    Returns:
        Tuple (optimizer, scheduler) або (optimizer, None) якщо scheduler не потрібен
    """
    training_cfg = cfg.get('training', {})
    
    # Параметри optimizer
    optimizer_type = training_cfg.get('optimizer', 'adamw').lower()
    learning_rate = training_cfg.get('learning_rate', 1e-4)
    # Конвертувати в float якщо рядок (наприклад "1e-4" з YAML)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    weight_decay = training_cfg.get('weight_decay', 1.0)
    # Конвертувати в float якщо рядок
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    warmup_steps = training_cfg.get('warmup_steps', 2000)
    # Конвертувати в int якщо рядок
    if isinstance(warmup_steps, str):
        warmup_steps = int(warmup_steps)
    
    # Створити optimizer
    if optimizer_type == 'muon' and MuonAdamAtan2 is not None:
        muon_lr = training_cfg.get('muon_learning_rate', 1e-3)
        optimizer = MuonAdamAtan2(
            model.parameters(),
            lr=learning_rate,
            muon_lr=muon_lr,
            weight_decay=weight_decay
        )
    else:
        # AdamW за замовчуванням
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Створити LR scheduler з warmup
    scheduler = None
    if warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler

