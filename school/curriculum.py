"""
Програма навчання для школи - адаптивна складність
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class LessonTopic:
    """Тема уроку"""
    topic: str
    category: str  # "математика", "наука", "мова", тощо
    base_difficulty: int = 1


class SchoolCurriculum:
    """Програма навчання як в садочку"""
    
    def __init__(self, topics_file: Optional[str] = None):
        """
        Ініціалізація програми
        
        Args:
            topics_file: JSON файл з темами (опціонально)
        """
        self.current_level = 1
        self.lessons_completed = 0
        self.performance_history = []
        
        # Завантажити теми
        if topics_file and Path(topics_file).exists():
            self.topics = self._load_topics(topics_file)
        else:
            self.topics = self._default_topics()
        
        print(f"[CURRICULUM] Програма створена: {len(self.topics)} тем, рівень {self.current_level}")
    
    def _default_topics(self) -> List[LessonTopic]:
        """Теми за замовчуванням"""
        return [
            LessonTopic("що таке число", "математика", 1),
            LessonTopic("як рахувати до 10", "математика", 1),
            LessonTopic("що таке буква", "мова", 1),
            LessonTopic("як працює комп'ютер", "наука", 2),
            LessonTopic("що таке AI", "наука", 3),
            LessonTopic("як працює інтернет", "наука", 3),
            LessonTopic("що таке програмування", "наука", 4),
            LessonTopic("як працює нейронна мережа", "наука", 5),
        ]
    
    def _load_topics(self, filepath: str) -> List[LessonTopic]:
        """Завантажити теми з файлу"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [LessonTopic(**item) for item in data]
        except Exception as e:
            print(f"[CURRICULUM] Помилка завантаження тем: {e}, використовуються дефолтні")
            return self._default_topics()
    
    def get_next_lesson(self) -> Optional[LessonTopic]:
        """Отримати наступний урок"""
        # Фільтрувати теми за поточним рівнем
        available = [
            t for t in self.topics 
            if t.base_difficulty <= self.current_level
        ]
        
        if not available:
            return None
        
        # Вибрати тему (можна додати логіку вибору)
        return available[self.lessons_completed % len(available)]
    
    def update_performance(self, score: float):
        """
        Оновити прогрес дитини
        
        Args:
            score: Оцінка (0-1)
        """
        self.performance_history.append(score)
        self.lessons_completed += 1
        
        # Адаптувати рівень
        if len(self.performance_history) >= 3:
            avg_score = sum(self.performance_history[-3:]) / 3
            
            if avg_score > 0.8 and self.current_level < 5:
                old_level = self.current_level
                self.current_level += 1
                print(f"[CURRICULUM] [OK] Перехід на рівень {self.current_level}!")
            elif avg_score < 0.3 and self.current_level > 1:
                old_level = self.current_level
                self.current_level -= 1
                print(f"[CURRICULUM] [INFO] Повертаємося до рівня {self.current_level}")
    
    def get_difficulty(self) -> int:
        """Отримати поточну складність"""
        return self.current_level
    
    def get_progress(self) -> Dict:
        """Отримати прогрес"""
        avg_score = 0.0
        if self.performance_history:
            recent = self.performance_history[-10:]
            avg_score = sum(recent) / len(recent) if recent else 0.0
        
        return {
            'level': self.current_level,
            'lessons_completed': self.lessons_completed,
            'avg_score': avg_score,
            'total_lessons': len(self.performance_history)
        }

