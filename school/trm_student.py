"""
TRM як дитина - навчається у GPT вчительки
"""
import torch
from torch import nn
from typing import Dict, Optional
from pathlib import Path

from tiny_recursive_model import TinyRecursiveModel
from inference.model_inference import TRMInference


class TRMStudent:
    """TRM як дитина в садочку"""
    
    def __init__(
        self,
        model: TinyRecursiveModel,
        tokenizer,
        device: str = "cpu"
    ):
        """
        Ініціалізація дитини (TRM)
        
        Args:
            model: TRM модель
            tokenizer: Токенізатор
            device: Пристрій
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Створити TRMInference для правильної генерації
        self.inference = TRMInference(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_seq_len=1024,
            timeout_seconds=30.0
        )
        
        print(f"[STUDENT] Дитина готова до навчання!")
    
    def try_to_answer(self, question: str, max_length: int = 512) -> str:
        """
        Дитина намагається відповісти на питання
        
        Args:
            question: Питання від вчительки
            max_length: Максимальна довжина відповіді
        
        Returns:
            Відповідь дитини
        """
        try:
            # Використати TRMInference для правильної генерації
            result = self.inference.predict(
                context="",  # Порожній контекст для простих питань
                query=question,
                max_deep_refinement_steps=12,
                halt_prob_thres=0.5
            )
            
            # Отримати completion з результату
            answer = result.get('completion', '').strip()
            
            # Обмежити довжину якщо потрібно
            if max_length and len(answer) > max_length:
                answer = answer[:max_length] + "..."
            
            return answer if answer else "Не вдалося згенерувати відповідь"
        except Exception as e:
            print(f"[STUDENT] Помилка генерації: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def learn_from_feedback(
        self,
        student_answer: str,
        correct_answer: str,
        reward: float
    ) -> Dict:
        """
        Дитина вчиться на feedback від вчительки
        
        Args:
            student_answer: Відповідь дитини
            correct_answer: Правильна відповідь
            reward: Нагорода (0-1)
        
        Returns:
            Метрики навчання
        """
        # Токенізувати обидві відповіді
        try:
            student_ids = self.tokenizer.encode(
                student_answer, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            correct_ids = self.tokenizer.encode(
                correct_answer, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
        except Exception as e:
            print(f"[STUDENT] Помилка токенізації: {e}")
            student_ids = torch.tensor([[]]).to(self.device)
            correct_ids = torch.tensor([[]]).to(self.device)
        
        # Обчислити loss (використовується в trainer)
        # Це буде викликано в trainer з правильним loss
        
        return {
            'reward': reward,
            'student_length': len(student_answer),
            'correct_length': len(correct_answer)
        }

