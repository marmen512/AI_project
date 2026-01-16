"""
GPT як вчителька - генерує уроки та оцінює відповіді TRM
"""
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class GPTTeacher:
    """GPT як вчителька для TRM (як в садочку)"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.7,
        max_length: int = 512
    ):
        """
        Ініціалізація вчительки
        
        Args:
            model_name: Назва GPT моделі
            cache_dir: Де зберігати моделі
            device: Пристрій (cpu/cuda)
            temperature: Температура для генерації
            max_length: Максимальна довжина генерації
        """
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = str(project_root / "models" / "pretrained")
        
        print(f"[TEACHER] Завантаження GPT вчительки: {model_name}...")
        
        # Завантажити GPT-2 для генерації
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)
        self.model.eval()  # Заморожений teacher
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[TEACHER] Вчителька готова! (vocab_size={len(self.tokenizer)})")
    
    def teach_lesson(
        self, 
        topic: str, 
        difficulty_level: int = 1,
        num_examples: int = 1
    ) -> Dict:
        """
        Вчителька дає урок (генерує приклади)
        
        Args:
            topic: Тема уроку
            difficulty_level: Рівень складності (1-5)
            num_examples: Кількість прикладів
        
        Returns:
            Словник з уроком: {'question', 'correct_answer', 'difficulty', 'topic'}
        """
        # Адаптувати складність як в садочку
        if difficulty_level == 1:
            prompt = f"Простими словами, як для дитини 5 років: {topic}"
            max_len = 50
        elif difficulty_level == 2:
            prompt = f"Поясни просто: {topic}"
            max_len = 100
        elif difficulty_level == 3:
            prompt = f"Поясни: {topic}"
            max_len = 200
        elif difficulty_level == 4:
            prompt = f"Детально поясни: {topic}"
            max_len = 300
        else:  # level 5
            prompt = f"Дуже детально поясни з прикладами: {topic}"
            max_len = 512
        
        lessons = []
        
        with torch.no_grad():
            for _ in range(num_examples):
                # Токенізувати prompt
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Обмежити довжину
                max_gen_len = min(inputs.size(1) + max_len, self.max_length)
                
                # Генерувати відповідь
                outputs = self.model.generate(
                    inputs,
                    max_length=max_gen_len,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                # Декодувати
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_text[len(prompt):].strip()
                
                lessons.append({
                    'question': prompt,
                    'correct_answer': answer,
                    'difficulty': difficulty_level,
                    'topic': topic
                })
        
        return lessons[0] if num_examples == 1 else lessons
    
    def check_homework(
        self, 
        student_answer: str, 
        correct_answer: str
    ) -> Tuple[float, str, Dict]:
        """
        Вчителька перевіряє домашнє завдання
        
        Args:
            student_answer: Відповідь студента (TRM)
            correct_answer: Правильна відповідь (GPT)
        
        Returns:
            (score, feedback, metadata)
        """
        # Обчислити схожість через embeddings
        try:
            student_tokens = self.tokenizer.encode(
                student_answer, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            correct_tokens = self.tokenizer.encode(
                correct_answer, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                # Отримати embeddings
                student_emb = self.model.transformer.wte(student_tokens).mean(dim=1)
                correct_emb = self.model.transformer.wte(correct_tokens).mean(dim=1)
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    student_emb, correct_emb, dim=1
                ).item()
        except Exception as e:
            # Fallback: проста схожість по словам
            student_words = set(student_answer.lower().split())
            correct_words = set(correct_answer.lower().split())
            if len(correct_words) == 0:
                similarity = 0.0
            else:
                similarity = len(student_words & correct_words) / len(correct_words)
        
        # Генерувати feedback
        if similarity > 0.8:
            feedback = "Чудово! [OK]"
            reward = 1.0
        elif similarity > 0.6:
            feedback = "Добре, але можна краще"
            reward = 0.7
        elif similarity > 0.4:
            feedback = "Майже правильно, спробуй ще раз"
            reward = 0.4
        else:
            feedback = "Потрібно вчитися більше. Ось правильна відповідь..."
            reward = 0.1
        
        metadata = {
            'similarity': similarity,
            'student_length': len(student_answer),
            'correct_length': len(correct_answer)
        }
        
        return reward, feedback, metadata
    
    def generate_teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Генерувати logits від teacher для knowledge distillation
        
        Args:
            input_ids: Input tokens [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits
    
    def get_teacher_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Отримати embeddings від teacher
        
        Args:
            input_ids: Input tokens [batch_size, seq_len]
        
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            return self.model.transformer.wte(input_ids)

