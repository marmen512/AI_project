"""
Конфігурація для інференсу моделі
Окрема від TrainConfig для уникнення shared state та багів
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from core.constants import InferenceMode


@dataclass
class InferConfig:
    """Конфігурація для інференсу TRM моделі"""
    
    # Модель (мінімальна конфігурація для інференсу)
    model_path: str | Path  # Шлях до завантаженої моделі
    
    # Інференс параметри
    max_deep_refinement_steps: int = 12
    halt_prob_thres: float = 0.5
    temperature: float = 0.7
    top_k: int = 50
    max_seq_len: int = 2048
    
    # Режим інференсу
    mode: InferenceMode = InferenceMode.STANDARD
    deterministic: bool = False  # Детерміністичний режим (тільки для debug)
    
    # Timeout для безпеки
    timeout_seconds: Optional[float] = None  # Timeout для recursion (None = без timeout)
    
    # Device
    device: str = 'cpu'  # 'cpu' або 'cuda'
    
    def validate(self):
        """Перевірити коректність конфігурації"""
        assert 0 < self.halt_prob_thres <= 1, "halt_prob_thres повинен бути в (0, 1]"
        assert self.max_deep_refinement_steps > 0, "max_deep_refinement_steps повинен бути > 0"
        assert self.temperature > 0, "temperature повинен бути > 0"
        assert self.top_k > 0, "top_k повинен бути > 0"
        assert self.max_seq_len > 0, "max_seq_len повинен бути > 0"
        assert self.device in ['cpu', 'cuda'], "device повинен бути 'cpu' або 'cuda'"
        
        if isinstance(self.model_path, str):
            model_path = Path(self.model_path)
        else:
            model_path = self.model_path
        assert model_path.exists(), f"Модель не знайдена: {model_path}"
    
    def to_dict(self) -> dict:
        """Конвертувати в словник"""
        return {
            'model_path': str(self.model_path),
            'max_deep_refinement_steps': self.max_deep_refinement_steps,
            'halt_prob_thres': self.halt_prob_thres,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'max_seq_len': self.max_seq_len,
            'mode': self.mode.value if isinstance(self.mode, InferenceMode) else self.mode,
            'deterministic': self.deterministic,
            'timeout_seconds': self.timeout_seconds,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'InferConfig':
        """Створити з словника"""
        d = d.copy()
        if 'mode' in d and isinstance(d['mode'], str):
            d['mode'] = InferenceMode(d['mode'])
        return cls(**d)

