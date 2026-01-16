"""
ARC-AGI Evaluator
Оцінка моделі на ARC (Abstraction and Reasoning Corpus) задачах
"""
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from train.evaluators.base_evaluator import BaseEvaluator


class ARCEvaluator(BaseEvaluator):
    """
    Evaluator для ARC-AGI задач
    ARC задачі вимагають трансформації 2D grid'ів
    """
    
    def __init__(self, exact_match_threshold: float = 1.0):
        """
        Ініціалізація ARC evaluator
        
        Args:
            exact_match_threshold: Поріг для exact match (1.0 = повне співпадіння)
        """
        self.exact_match_threshold = exact_match_threshold
        self.results: List[Dict[str, Any]] = []
        self.metrics_cache: Optional[Dict[str, float]] = None
    
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        max_samples: Optional[int] = None,
        max_deep_refinement_steps: int = 12,
        halt_prob_thres: float = 0.5,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Оцінити модель на ARC датасеті
        
        Args:
            model: TRM модель
            dataset: ARC датасет
            max_samples: Максимальна кількість прикладів
            max_deep_refinement_steps: Максимальна кількість кроків уточнення
            halt_prob_thres: Поріг для раннього виходу
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
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    sample = dataset[i]
                    
                    # ARC формат: {'train': [...], 'test': [...], 'task_id': ...}
                    # train містить приклади input/output пар
                    # test містить тестові input, для яких потрібно передбачити output
                    
                    if 'test' not in sample or not sample['test']:
                        continue
                    
                    # Для ARC задач, модель повинна передбачити output grid для кожного test input
                    test_inputs = sample['test']
                    
                    predictions = []
                    for test_input in test_inputs:
                        # Конвертувати grid в tensor або послідовність
                        input_grid = test_input['input']  # 2D grid
                        
                        # Трансформувати grid в послідовність для моделі
                        # Це спрощена версія - в реальності потрібні puzzle embeddings
                        input_seq = self._grid_to_sequence(input_grid)
                        input_tensor = torch.tensor([input_seq], dtype=torch.long)
                        
                        # Передбачення
                        if hasattr(model, 'predict'):
                            pred_output, exit_steps = model.predict(
                                input_tensor,
                                max_deep_refinement_steps=max_deep_refinement_steps,
                                halt_prob_thres=halt_prob_thres
                            )
                            pred_grid = self._sequence_to_grid(pred_output[0].cpu().numpy(), input_grid)
                        else:
                            # Fallback - просте передбачення
                            pred_grid = [[0]]
                        
                        predictions.append(pred_grid)
                    
                    # Оцінити передбачення
                    test_outputs = [t.get('output', None) for t in test_inputs]
                    
                    sample_correct = 0
                    sample_exact = 0
                    sample_partial = 0
                    
                    for pred, expected in zip(predictions, test_outputs):
                        if expected is None:
                            continue
                        
                        exact_match = self._compare_grids(pred, expected) >= self.exact_match_threshold
                        partial_match = self._compare_grids(pred, expected) > 0.5
                        
                        if exact_match:
                            sample_exact += 1
                            sample_correct += 1
                        elif partial_match:
                            sample_partial += 1
                            sample_correct += 0.5
                    
                    if sample_exact > 0:
                        exact_matches += 1
                    if sample_partial > 0 or sample_exact > 0:
                        partial_matches += 1
                    if sample_correct >= len(test_outputs):
                        correct += 1
                    
                    self.results.append({
                        'sample_id': sample.get('task_id', i),
                        'correct': sample_correct >= len(test_outputs),
                        'exact_match': sample_exact > 0,
                        'partial_match': sample_partial > 0,
                        'num_tests': len(test_outputs),
                        'num_correct': sample_correct
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
        
        self.metrics_cache = {
            'accuracy': accuracy,
            'exact_match_rate': exact_match_rate,
            'partial_match_rate': partial_match_rate,
            'total_samples': total,
            'correct_samples': correct
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
    
    def _grid_to_sequence(self, grid: List[List[int]]) -> List[int]:
        """Конвертувати 2D grid в послідовність"""
        seq = []
        for row in grid:
            seq.extend(row)
            seq.append(-1)  # Маркер нового рядка
        return seq
    
    def _sequence_to_grid(self, seq: np.ndarray, original_grid: List[List[int]]) -> List[List[int]]:
        """Конвертувати послідовність назад в 2D grid"""
        # Спрощена версія - повертаємо grid того ж розміру що і original
        height = len(original_grid)
        width = len(original_grid[0]) if original_grid else 1
        
        grid = []
        idx = 0
        for _ in range(height):
            row = []
            for _ in range(width):
                if idx < len(seq):
                    val = int(seq[idx])
                    row.append(max(0, min(9, val)))  # Обмежити до 0-9
                    idx += 1
                else:
                    row.append(0)
            grid.append(row)
        
        return grid
    
    def _compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Порівняти два grids та повернути схожість (0.0 - 1.0)"""
        if not grid1 or not grid2:
            return 0.0
        
        # Exact match
        if grid1 == grid2:
            return 1.0
        
        # Розрахувати частку співпадаючих елементів
        max_h = max(len(grid1), len(grid2))
        max_w = max(max(len(r) for r in grid1) if grid1 else 0,
                   max(len(r) for r in grid2) if grid2 else 0)
        
        matches = 0
        total = 0
        
        for i in range(max_h):
            row1 = grid1[i] if i < len(grid1) else []
            row2 = grid2[i] if i < len(grid2) else []
            
            for j in range(max_w):
                val1 = row1[j] if j < len(row1) else 0
                val2 = row2[j] if j < len(row2) else 0
                
                if val1 == val2:
                    matches += 1
                total += 1
        
        return matches / total if total > 0 else 0.0

