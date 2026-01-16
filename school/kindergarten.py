"""
Головна система навчання "Садочок" - координація вчительки та дитини
"""
import torch
from torch import nn
from typing import Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime

from .gpt_teacher import GPTTeacher
from .trm_student import TRMStudent
from .curriculum import SchoolCurriculum, LessonTopic


class KindergartenLearning:
    """Повна система навчання як в садочку"""
    
    def __init__(
        self,
        student_model,  # TRM модель
        tokenizer,
        teacher_model_name: str = "gpt2",
        curriculum_topics_file: Optional[str] = None,
        device: str = "cpu",
        save_dir: str = "school_progress"
    ):
        """
        Ініціалізація системи навчання
        
        Args:
            student_model: TRM модель (дитина)
            tokenizer: Токенізатор
            teacher_model_name: Назва GPT моделі для вчительки
            curriculum_topics_file: Файл з темами (опціонально)
            device: Пристрій
            save_dir: Де зберігати прогрес
        """
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Створити вчительку
        self.teacher = GPTTeacher(
            model_name=teacher_model_name,
            device=device
        )
        
        # Створити дитину
        self.student = TRMStudent(
            model=student_model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Створити програму
        self.curriculum = SchoolCurriculum(curriculum_topics_file)
        
        # Історія навчання
        self.history = []
        
        print(f"[SCHOOL] Садочок готовий до навчання!")
    
    def daily_lesson(self, day: int = 1) -> Optional[Dict]:
        """
        Щоденний урок як в садочку
        
        Args:
            day: Номер дня
        
        Returns:
            Результати уроку або None якщо уроки закінчилися
        """
        print(f"\n{'='*60}")
        print(f"[DAY {day}] Початок уроку")
        print(f"{'='*60}")
        
        # 1. Вчителька дає урок
        topic = self.curriculum.get_next_lesson()
        if not topic:
            print("[SCHOOL] Всі уроки завершено!")
            return None
        
        difficulty = self.curriculum.get_difficulty()
        lesson = self.teacher.teach_lesson(
            topic=topic.topic,
            difficulty_level=difficulty
        )
        
        print(f"[TEACHER] Тема: {topic.topic}")
        print(f"[TEACHER] Рівень: {difficulty}")
        print(f"[TEACHER] Питання: {lesson['question']}")
        
        # 2. Дитина намагається відповісти
        print(f"\n[STUDENT] Дитина думає...")
        student_answer = self.student.try_to_answer(lesson['question'])
        if len(student_answer) > 100:
            print(f"[STUDENT] Відповідь: {student_answer[:100]}...")
        else:
            print(f"[STUDENT] Відповідь: {student_answer}")
        
        # 3. Вчителька перевіряє
        reward, feedback, metadata = self.teacher.check_homework(
            student_answer,
            lesson['correct_answer']
        )
        
        print(f"\n[TEACHER] {feedback}")
        print(f"[TEACHER] Оцінка: {reward:.2f}/1.0")
        if len(lesson['correct_answer']) > 100:
            print(f"[TEACHER] Правильна відповідь: {lesson['correct_answer'][:100]}...")
        else:
            print(f"[TEACHER] Правильна відповідь: {lesson['correct_answer']}")
        
        # 4. Оновити прогрес
        self.curriculum.update_performance(reward)
        
        # 5. Зберегти результат
        result = {
            'day': day,
            'topic': topic.topic,
            'difficulty': difficulty,
            'question': lesson['question'],
            'student_answer': student_answer,
            'correct_answer': lesson['correct_answer'],
            'reward': reward,
            'feedback': feedback,
            'metadata': metadata,
            'progress': self.curriculum.get_progress()
        }
        
        self.history.append(result)
        self._save_progress(day)
        
        return result
    
    def full_education(self, days: int = 100, save_every: int = 10):
        """
        Повний курс навчання
        
        Args:
            days: Кількість днів навчання
            save_every: Зберігати прогрес кожні N днів
        """
        print(f"\n{'='*60}")
        print(f"[SCHOOL] Початок навчання на {days} днів")
        print(f"{'='*60}\n")
        
        for day in range(1, days + 1):
            result = self.daily_lesson(day)
            
            if result is None:
                break
            
            # Зберегти прогрес
            if day % save_every == 0:
                self._save_progress(day)
                progress = self.curriculum.get_progress()
                print(f"\n[PROGRESS] День {day}: Рівень {progress['level']}, "
                      f"Уроків: {progress['lessons_completed']}, "
                      f"Середня оцінка: {progress['avg_score']:.2f}")
        
        # Фінальний звіт
        self._print_final_report()
    
    def _save_progress(self, day: int):
        """Зберегти прогрес"""
        try:
            progress_file = self.save_dir / f"progress_day_{day}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'day': day,
                    'history': self.history[-10:],  # Останні 10 уроків
                    'curriculum': self.curriculum.get_progress()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SCHOOL] [WARN] Помилка збереження прогресу: {e}")
    
    def _print_final_report(self):
        """Вивести фінальний звіт"""
        print(f"\n{'='*60}")
        print(f"[SCHOOL] ФІНАЛЬНИЙ ЗВІТ")
        print(f"{'='*60}")
        
        progress = self.curriculum.get_progress()
        print(f"Рівень: {progress['level']}/5")
        print(f"Уроків завершено: {progress['lessons_completed']}")
        print(f"Середня оцінка: {progress['avg_score']:.2f}/1.0")
        
        if self.history:
            recent_scores = [h['reward'] for h in self.history[-10:]]
            print(f"Останні 10 оцінок: {[f'{s:.2f}' for s in recent_scores]}")

