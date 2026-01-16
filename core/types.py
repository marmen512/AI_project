"""
Централізовані типи для проекту
Dataclasses для Batch, Output, TrainState, EvalState
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from pathlib import Path


@dataclass
class Batch:
    """Batch даних для навчання"""
    input_ids: torch.Tensor
    labels: torch.Tensor
    doc_id: Optional[torch.Tensor] = None  # ID документа (для document-aware dataset)
    segment_id: Optional[torch.Tensor] = None  # ID сегменту в документі
    
    def to(self, device):
        """Перемістити batch на device"""
        return Batch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            doc_id=self.doc_id.to(device) if self.doc_id is not None else None,
            segment_id=self.segment_id.to(device) if self.segment_id is not None else None
        )


@dataclass
class Output:
    """Вихід моделі"""
    logits: torch.Tensor
    predictions: torch.Tensor
    halt_prob: Optional[torch.Tensor] = None
    recursion_depth: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainState:
    """
    Стан навчання - централізований об'єкт для відстеження стану
    """
    epoch: int = 0
    step: int = 0  # Загальний крок (batch_count)
    batch_idx: int = 0  # Індекс батча в поточній епосі
    loss: float = 0.0
    main_loss: float = 0.0
    halt_loss: float = 0.0
    thinking_cost: float = 0.0
    recursion_depth: int = 0
    actual_recurrent_steps: int = 0  # Фактична кількість використаних recurrent steps
    memory_usage: Optional[Dict[str, float]] = None  # Використання пам'яті
    checkpoint_path: Optional[Path] = None
    best_loss: Optional[float] = None
    best_eval_score: Optional[float] = None
    best_entropy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs):
        """Оновити стан"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертувати в словник"""
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = value
            elif value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainState':
        """Створити з словника"""
        if 'checkpoint_path' in d and d['checkpoint_path']:
            d['checkpoint_path'] = Path(d['checkpoint_path'])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalState:
    """
    Стан оцінки моделі
    """
    eval_step: int = 0
    eval_score: float = 0.0
    eval_loss: float = 0.0
    entropy: float = 0.0
    samples: list = field(default_factory=list)  # Зразки для qualitative eval
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs):
        """Оновити стан"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертувати в словник"""
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, (list, dict)):
                result[key] = value
            elif value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvalState':
        """Створити з словника"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

