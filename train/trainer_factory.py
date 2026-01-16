"""
Фабрика для створення Trainer
"""
from typing import Optional, TYPE_CHECKING
import torch
import multiprocessing

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from tiny_recursive_model import Trainer, TinyRecursiveModel

from torch.utils.data import Dataset
from train.curriculum import CurriculumScheduler
from train.constants import (
    MAX_NUM_WORKERS, DEFAULT_HALT_PROB_THRES,
    DEFAULT_WARMUP_STEPS, DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS
)


def create_trainer(
    model: 'TinyRecursiveModel',
    dataset: Dataset,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    max_recurrent_steps: int,
    use_gpu: bool = True,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    warmup_steps: Optional[int] = None,
    halt_prob_thres: float = DEFAULT_HALT_PROB_THRES,
    thinking_cost_weight: float = 0.01,  # Вага thinking cost в loss
    num_workers: Optional[int] = None,
    mixed_precision: Optional[str] = None,
    validation_dataset: Optional[Dataset] = None,
    validation_interval: int = 500,
    early_stopping_patience: Optional[int] = None,
    save_best_model: bool = True,
    resource_monitor = None,  # ResourceMonitor для моніторингу ресурсів
    training_logger = None,  # TRMTrainingLogger для логування метрик
    curriculum_scheduler: Optional[CurriculumScheduler] = None,  # CurriculumScheduler для керування етапами
    enable_early_stopping: bool = False,  # Увімкнути entropy-driven early stopping
    early_stopping_entropy_patience: int = 3,  # Patience для entropy
    **kwargs
) -> 'Trainer':
    """
    Створити Trainer з оптимальними параметрами
    
    Args:
        model: TRM модель
        dataset: Датасет для навчання
        learning_rate: Learning rate
        batch_size: Розмір батчу
        epochs: Кількість епох
        max_recurrent_steps: Максимальна кількість рекурсивних кроків
        use_gpu: Використовувати GPU
        checkpoint_dir: Директорія для checkpoint'ів
        checkpoint_interval: Інтервал збереження checkpoint'ів
        gradient_accumulation_steps: Кроки накопичення градієнтів
        warmup_steps: Кількість warmup кроків (None = автоматично)
        halt_prob_thres: Поріг для раннього виходу
        num_workers: Кількість worker'ів для DataLoader (None = автоматично)
        mixed_precision: Mixed precision mode ('fp16', 'bf16', None)
        validation_dataset: Validation датасет (None = без валідації)
        validation_interval: Кожні скільки батчів робити валідацію
        early_stopping_patience: Patience для early stopping (None = вимкнено)
        save_best_model: Зберігати найкращу модель за validation loss
        **kwargs: Додаткові параметри
    
    Returns:
        Надініціалізований Trainer
    """
    # Визначити num_workers (оптимізовано для швидкості)
    if num_workers is None:
        if use_gpu and torch.cuda.is_available():
            # Використовувати оптимальну кількість workers для балансу швидкості та пам'яті
            # 2-4 workers зазвичай оптимально для більшості випадків
            num_workers = min(MAX_NUM_WORKERS, max(2, multiprocessing.cpu_count() // 2))
        else:
            num_workers = 0
    
    # Визначити mixed_precision
    if mixed_precision is None and use_gpu and torch.cuda.is_available():
        mixed_precision = 'fp16'
    
    # Warmup steps
    if warmup_steps is None:
        warmup_steps = DEFAULT_WARMUP_STEPS
    
    # Створити Trainer з підтримуваними параметрами
    # Примітка: Trainer.__init__() не підтримує gradient_accumulation_steps, 
    # dataloader_num_workers, mixed_precision, validation_dataset та інші параметри
    # Вони можуть бути додані в майбутньому, але зараз використовуємо тільки підтримувані
    
    trainer_kwargs = {
        'model': model,
        'dataset': dataset,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'max_recurrent_steps': max_recurrent_steps,
        'halt_prob_thres': halt_prob_thres,
        'thinking_cost_weight': thinking_cost_weight,
        'warmup_steps': warmup_steps,
        'cpu': not use_gpu,
        'checkpoint_dir': checkpoint_dir,
        'checkpoint_interval': checkpoint_interval,
    }
    
    # Додати log_file якщо передано
    if 'log_file' in kwargs:
        trainer_kwargs['log_file'] = kwargs.pop('log_file')
    
    # Додати resource_monitor, training_logger та curriculum_scheduler якщо передано
    if resource_monitor is not None:
        trainer_kwargs['resource_monitor'] = resource_monitor
    if training_logger is not None:
        trainer_kwargs['training_logger'] = training_logger
    if curriculum_scheduler is not None:
        trainer_kwargs['curriculum_scheduler'] = curriculum_scheduler
    
    # Додати early stopping callback якщо увімкнено
    callbacks = []
    if enable_early_stopping:
        from train.callbacks import EarlyStoppingCallback
        early_stop_patience = early_stopping_patience if early_stopping_patience is not None else 5
        early_stop_callback = EarlyStoppingCallback(
            patience=early_stop_patience,
            entropy_patience=early_stopping_entropy_patience
        )
        callbacks.append(early_stop_callback)
        trainer_kwargs['callbacks'] = callbacks
    
    # Додати тільки ті kwargs, які можуть бути корисні для Accelerator
    # (через accelerate_kwargs)
    accelerate_kwargs = {}
    if gradient_accumulation_steps > 1:
        accelerate_kwargs['gradient_accumulation_steps'] = gradient_accumulation_steps
    if mixed_precision:
        accelerate_kwargs['mixed_precision'] = mixed_precision
    
    if accelerate_kwargs:
        trainer_kwargs['accelerate_kwargs'] = accelerate_kwargs
    
    # ВИДАЛИТИ непідтримувані параметри, які передаються як явні аргументи
    # (не через kwargs) - вони не підтримуються Trainer.__init__()
    unsupported_params = [
        'gradient_accumulation_steps', 'dataloader_num_workers', 
        'mixed_precision', 'validation_dataset', 'validation_interval',
        'early_stopping_patience', 'save_best_model', 'num_workers'
    ]
    
    # Додати інші kwargs якщо вони не конфліктують
    for key, value in kwargs.items():
        if key not in unsupported_params:
            trainer_kwargs[key] = value
    
    # Lazy import to avoid circular dependency
    from tiny_recursive_model import Trainer
    return Trainer(**trainer_kwargs)
