"""
Конфігурація для навчання моделі
Окрема від InferConfig для уникнення shared state та багів
Перейменовано з trm_config.py для ясності
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TrainConfig:
    """Конфігурація для навчання TRM моделі"""
    
    # Модель (TRM-оптимальні розміри)
    dim: int = 256
    depth: int = 4
    seq_len: int = 256
    vocab_size: Optional[int] = None  # Автоматично з tokenizer
    
    # Навчання
    batch_size: int = 4
    effective_batch_size: int = 16  # batch_size * gradient_accumulation
    epochs: int = 10
    learning_rate: float = 1e-4
    
    # Рекурсія
    max_recurrent_steps: int = 12
    halt_prob_thres: float = 0.5
    max_recursion_depth: int = 20
    adaptive_recursion: bool = False  # Увімкнути adaptive recursion gate
    
    # Thinking cost
    thinking_cost_weight: float = 0.01
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_start_len: int = 64
    curriculum_max_len: int = 256
    curriculum_stages: int = 4
    curriculum_epochs_per_stage: int = 3
    
    # Оптимізація
    warmup_steps: int = 1000
    weight_decay: float = 1.0
    
    # Dataset
    cache_size: int = 1000
    validate_format: bool = True
    
    @property
    def gradient_accumulation_steps(self) -> int:
        """Автоматично обчислити з effective_batch_size"""
        return max(1, self.effective_batch_size // self.batch_size)
    
    def validate(self):
        """Перевірити коректність конфігурації"""
        assert self.dim > 0, "dim повинен бути > 0"
        assert 0 < self.halt_prob_thres <= 1, "halt_prob_thres повинен бути в (0, 1]"
        assert self.max_recurrent_steps > 0, "max_recurrent_steps повинен бути > 0"
        assert self.seq_len >= self.curriculum_start_len, "seq_len повинен бути >= curriculum_start_len"
        assert self.curriculum_max_len >= self.curriculum_start_len, "curriculum_max_len повинен бути >= curriculum_start_len"
        assert self.batch_size > 0, "batch_size повинен бути > 0"
        assert self.effective_batch_size >= self.batch_size, "effective_batch_size повинен бути >= batch_size"
    
    def to_dict(self) -> dict:
        """Конвертувати в словник"""
        return {
            'dim': self.dim,
            'depth': self.depth,
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'effective_batch_size': self.effective_batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'max_recurrent_steps': self.max_recurrent_steps,
            'halt_prob_thres': self.halt_prob_thres,
            'max_recursion_depth': self.max_recursion_depth,
            'adaptive_recursion': self.adaptive_recursion,
            'thinking_cost_weight': self.thinking_cost_weight,
            'curriculum_enabled': self.curriculum_enabled,
            'curriculum_start_len': self.curriculum_start_len,
            'curriculum_max_len': self.curriculum_max_len,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainConfig':
        """Створити з словника"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Backwards compatibility: TRMConfig -> TrainConfig
# Старий код може використовувати TRMConfig, він буде аліасом до TrainConfig
TRMConfig = TrainConfig

