"""
Модуль для автоматичного продовження навчання та перевірки дублювання
DEPRECATED: Використовуйте runtime.resume замість цього модуля
"""
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path

# Deprecation warning
warnings.warn(
    "config.training_resume застарів. "
    "Використовуйте runtime.resume замість цього.",
    DeprecationWarning,
    stacklevel=2
)

# Імпортувати з runtime для backwards compatibility
from runtime.resume import (
    find_latest_checkpoint,
    get_checkpoint_info,
    should_auto_resume,
    maybe_resume
)


@dataclass
class TrainingResumeConfig:
    """
    Dataclass для конфігурації resume
    Тільки структура даних, без runtime-логіки
    """
    checkpoint_dir: str = "checkpoints"
    auto_resume: bool = False
    max_checkpoints: int = 5


# Backwards compatibility: створити wrapper клас
class TrainingResume:
    """
    DEPRECATED: Використовуйте runtime.resume замість цього
    Wrapper для backwards compatibility
    """
    
    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        warnings.warn(
            "TrainingResume застарів. Використовуйте runtime.resume.find_latest_checkpoint()",
            DeprecationWarning,
            stacklevel=2
        )
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """DEPRECATED: Використовуйте runtime.resume.find_latest_checkpoint()"""
        return find_latest_checkpoint(self.checkpoint_dir)
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict:
        """DEPRECATED: Використовуйте runtime.resume.get_checkpoint_info()"""
        return get_checkpoint_info(checkpoint_path)
