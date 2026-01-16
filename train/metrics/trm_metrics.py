"""
TRM Training Logger для логування метрик навчання в JSONL формат
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


class TRMTrainingLogger:
    """
    Логер для запису метрик навчання в JSONL формат
    
    Кожен запис містить:
    - step: номер кроку (батча)
    - epoch: номер епохи
    - loss: значення loss
    - recursion_depth: глибина рекурсії
    - tokens_per_sec: швидкість генерації токенів
    - ts: timestamp
    """
    
    def __init__(self, path: str | Path):
        """
        Ініціалізація логера
        
        Args:
            path: Шлях до JSONL файлу для логування
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Відкрити файл в режимі append з буферизацією
        self.f = open(self.path, "a", buffering=1, encoding='utf-8')
    
    def log(self, step: int, epoch: int, loss: float, 
            recursion_depth: Optional[float] = None,
            tokens_per_sec: Optional[float] = None,
            **kwargs):
        """
        Записати метрики в JSONL файл
        
        Args:
            step: Номер кроку (батча)
            epoch: Номер епохи
            loss: Значення loss
            recursion_depth: Глибина рекурсії (опціонально)
            tokens_per_sec: Швидкість генерації токенів (опціонально)
            **kwargs: Додаткові метрики для логування
        """
        log_entry = {
            "step": step,
            "epoch": epoch,
            "loss": float(loss),
            "ts": time.time()
        }
        
        if recursion_depth is not None:
            log_entry["recursion_depth"] = float(recursion_depth)
        
        if tokens_per_sec is not None:
            log_entry["tokens_per_sec"] = float(tokens_per_sec)
        
        # Додати додаткові метрики
        log_entry.update({k: float(v) if isinstance(v, (int, float)) else v 
                         for k, v in kwargs.items()})
        
        # Записати рядок JSON
        self.f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self.f.flush()  # Переконатися, що дані записані
    
    def close(self):
        """Закрити файл логера"""
        if hasattr(self, 'f') and self.f:
            self.f.close()
    
    def __del__(self):
        """Деструктор - закрити файл при видаленні об'єкта"""
        self.close()

