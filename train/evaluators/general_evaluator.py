"""
General Evaluator
Оцінка моделі на загальних NLP задачах (текст, код, тощо)
Використовується як базовий evaluator для більшості задач
"""
from typing import Dict, Any, List, Optional
import torch

from train.evaluators.base_evaluator import BaseEvaluator


class GeneralEvaluator(BaseEvaluator):
    """
    Evaluator для загальних NLP задач
    Використовує метрики: accuracy, exact_match, partial_match, keyword_match
    """
    
    def __init__(self):
        """Ініціалізація General evaluator"""
        self.results: List[Dict[str, Any]] = []
        self.metrics_cache: Optional[Dict[str, float]] = None
    
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        max_samples: Optional[int] = None,
        max_deep_refinement_steps: int = 12,
        halt_prob_thres: float = 0.5,
        tokenizer: Optional[Any] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Оцінити модель на загальному датасеті
        
        Args:
            model: TRM модель
            dataset: Датасет (очікує формат з 'context', 'query', 'completion')
            max_samples: Максимальна кількість прикладів
            max_deep_refinement_steps: Максимальна кількість кроків уточнення
            halt_prob_thres: Поріг для раннього виходу
            tokenizer: Tokenizer для декодування (опціонально)
            **kwargs: Додаткові параметри
        
        Returns:
            Словник з результатами оцінки
        """
        model.eval()
        self.results = []
        
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        correct = 0
        exact_matches = 0
        partial_matches = 0
        keyword_matches = 0
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    sample = dataset[i]
                    
                    # Отримати context, query, expected output
                    context = sample.get('context', sample.get('input', ''))
                    query = sample.get('query', '')
                    expected_output = sample.get('completion', sample.get('output', ''))
                    
                    if not expected_output:
                        continue
                    
                    # Підготувати вхідний текст
                    if context and query:
                        input_text = f"{context}\n\n[QUERY]\n{query}"
                    elif query:
                        input_text = query
                    else:
                        input_text = context
                    
                    # Токенізувати (спрощено - очікуємо що dataset вже токенізований)
                    # Або використати tokenizer якщо передано
                    if tokenizer and hasattr(tokenizer, 'encode'):
                        input_ids = tokenizer.encode(
                            input_text,
                            max_length=512,
                            truncation=True,
                            return_tensors='pt'
                        )
                    else:
                        # Спрощений підхід - конвертувати текст в числа
                        input_ids = torch.tensor([[ord(c) % 256 for c in input_text[:512]]], dtype=torch.long)
                    
                    # Передбачення
                    if hasattr(model, 'predict'):
                        pred_output, exit_steps = model.predict(
                            input_ids,
                            max_deep_refinement_steps=max_deep_refinement_steps,
                            halt_prob_thres=halt_prob_thres
                        )
                        
                        # Декодувати
                        if tokenizer and hasattr(tokenizer, 'decode'):
                            predicted_text = tokenizer.decode(
                                pred_output[0].cpu().numpy(),
                                skip_special_tokens=True
                            )
                        else:
                            # Спрощене декодування
                            predicted_text = ''.join([chr(int(t) % 256) if 0 <= int(t) < 256 else '?' for t in pred_output[0].cpu().numpy()])
                    else:
                        predicted_text = ""
                    
                    # Оцінити передбачення
                    exact_match = (
                        predicted_text.strip().lower() == expected_output.strip().lower()
                    )
                    
                    partial_match = (
                        expected_output.lower().strip() in predicted_text.lower().strip() or
                        predicted_text.lower().strip() in expected_output.lower().strip()
                    )
                    
                    # Keyword match
                    expected_words = set(expected_output.lower().split())
                    predicted_words = set(predicted_text.lower().split())
                    keyword_match = len(expected_words & predicted_words) / max(len(expected_words), 1) > 0.5
                    
                    is_correct = exact_match or partial_match or keyword_match
                    
                    if is_correct:
                        correct += 1
                    if exact_match:
                        exact_matches += 1
                    if partial_match:
                        partial_matches += 1
                    if keyword_match:
                        keyword_matches += 1
                    
                    # Обчислити score
                    score = 0.0
                    if exact_match:
                        score = 1.0
                    elif partial_match:
                        score = 0.7
                    elif keyword_match:
                        score = 0.5
                    
                    self.results.append({
                        'sample_id': i,
                        'input': input_text[:80],
                        'expected': expected_output[:80],
                        'predicted': predicted_text[:120],
                        'correct': is_correct,
                        'score': score,
                        'exact_match': exact_match,
                        'partial_match': partial_match,
                        'keyword_match': keyword_match,
                        'exit_steps': exit_steps[0].item() if hasattr(exit_steps, '__getitem__') else 0
                    })
                
                except Exception as e:
                    self.results.append({
                        'sample_id': i,
                        'error': str(e),
                        'correct': False
                    })
        
        # Обчислити метрики
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0.0
        exact_match_rate = exact_matches / total if total > 0 else 0.0
        partial_match_rate = partial_matches / total if total > 0 else 0.0
        keyword_match_rate = keyword_matches / total if total > 0 else 0.0
        avg_score = sum(r.get('score', 0.0) for r in self.results) / total if total > 0 else 0.0
        
        self.metrics_cache = {
            'accuracy': accuracy,
            'exact_match_rate': exact_match_rate,
            'partial_match_rate': partial_match_rate,
            'keyword_match_rate': keyword_match_rate,
            'avg_score': avg_score,
            'total_samples': float(total),
            'correct_samples': float(correct)
        }
        
        return {
            'metrics': self.metrics_cache,
            'results': self.results
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Отримати метрики"""
        if self.metrics_cache is None:
            return {}
        return self.metrics_cache

