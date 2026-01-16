"""
Визначення режимів навчання
"""
from enum import Enum
from typing import Optional


class TrainingMode(Enum):
    """Режими навчання"""
    NEW = "new"  # Новий запуск навчання
    RESUME = "resume"  # Продовження з checkpoint
    SERVICE = "service"  # Service режим (автоматичний restart)


def parse_mode(mode_str: Optional[str]) -> TrainingMode:
    """
    Парсити режим з рядка
    
    Args:
        mode_str: Рядок режиму ("new", "resume", "service")
    
    Returns:
        TrainingMode enum
    """
    if mode_str is None:
        return TrainingMode.NEW
    
    mode_str = mode_str.lower().strip()
    try:
        return TrainingMode(mode_str)
    except ValueError:
        # За замовчуванням - NEW
        return TrainingMode.NEW

