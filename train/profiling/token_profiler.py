"""
Token-level Profiler
Профілювання продуктивності на рівні окремих токенів
"""
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path
import json


class TokenProfiler:
    """
    Профілер для токенів
    Відстежує час обробки окремих токенів
    """
    
    def __init__(self, output_dir: str = "logs/profiling"):
        """
        Ініціалізація профілера
        
        Args:
            output_dir: Директорія для збереження звітів
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = defaultdict(list)
        self.profiling_active = False
    
    def start(self):
        """Почати профілювання"""
        self.profiling_active = True
        self.stats = defaultdict(list)
    
    def stop(self):
        """Зупинити профілювання"""
        self.profiling_active = False
    
    def record(self, token: int, duration: float):
        """
        Записати час обробки токена
        
        Args:
            token: ID токена
            duration: Час обробки в секундах
        """
        if self.profiling_active:
            self.stats[token].append(duration)
    
    def summary(self, top_k: int = 10) -> List[Tuple[int, float, int]]:
        """
        Отримати статистику по токенах
        
        Args:
            top_k: Кількість топ токенів для повернення
        
        Returns:
            Список (token_id, avg_duration, count) відсортований за avg_duration
        """
        avg = {}
        for token, durations in self.stats.items():
            if durations:
                avg[token] = sum(durations) / len(durations)
        
        # Сортувати за середнім часом (найповільніші першими)
        sorted_tokens = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        
        # Повернути top_k з кількістю викликів
        result = []
        for token, avg_duration in sorted_tokens[:top_k]:
            count = len(self.stats[token])
            result.append((token, avg_duration, count))
        
        return result
    
    def save_report(self, filename: str = "token_profiling_report.json") -> Path:
        """
        Зберегти звіт профілювання
        
        Args:
            filename: Ім'я файлу
        
        Returns:
            Шлях до збереженого файлу
        """
        report = {
            'total_tokens': len(self.stats),
            'top_tokens': [
                {
                    'token_id': token,
                    'avg_duration': avg_duration,
                    'count': count
                }
                for token, avg_duration, count in self.summary(top_k=20)
            ]
        }
        
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path

