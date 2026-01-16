"""
Sudoku Evaluator
Оцінка моделі на Sudoku задачах
"""
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from train.evaluators.base_evaluator import BaseEvaluator


class SudokuEvaluator(BaseEvaluator):
    """
    Evaluator для Sudoku задач
    Перевіряє валідність рішення Sudoku (9x9 grid з правилами)
    """
    
    def __init__(self):
        """Ініціалізація Sudoku evaluator"""
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
        Оцінити модель на Sudoku датасеті
        
        Args:
            model: TRM модель
            dataset: Sudoku датасет
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
        valid = 0  # Валідні рішення (навіть якщо не повністю правильні)
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    sample = dataset[i]
                    
                    # Sudoku формат: {'puzzle': 9x9 grid, 'solution': 9x9 grid}
                    puzzle = sample.get('puzzle', sample.get('input', []))
                    expected_solution = sample.get('solution', sample.get('output', []))
                    
                    if not puzzle or not expected_solution:
                        continue
                    
                    # Конвертувати puzzle в послідовність
                    puzzle_seq = self._grid_to_sequence(puzzle)
                    input_tensor = torch.tensor([puzzle_seq], dtype=torch.long)
                    
                    # Передбачення
                    if hasattr(model, 'predict'):
                        pred_output, exit_steps = model.predict(
                            input_tensor,
                            max_deep_refinement_steps=max_deep_refinement_steps,
                            halt_prob_thres=halt_prob_thres
                        )
                        pred_grid = self._sequence_to_grid(pred_output[0].cpu().numpy(), puzzle)
                    else:
                        pred_grid = puzzle.copy()
                    
                    # Перевірити валідність та правильність
                    is_valid = self._is_valid_sudoku(pred_grid)
                    is_correct = self._compare_grids(pred_grid, expected_solution) >= 1.0
                    
                    if is_valid:
                        valid += 1
                    if is_correct:
                        correct += 1
                    
                    self.results.append({
                        'sample_id': i,
                        'correct': is_correct,
                        'valid': is_valid,
                        'exit_steps': exit_steps[0].item() if hasattr(exit_steps, '__getitem__') else 0
                    })
                
                except Exception as e:
                    self.results.append({
                        'sample_id': i,
                        'error': str(e),
                        'correct': False,
                        'valid': False
                    })
        
        # Обчислити метрики
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0.0
        validity_rate = valid / total if total > 0 else 0.0
        
        self.metrics_cache = {
            'accuracy': accuracy,
            'validity_rate': validity_rate,
            'total_samples': float(total),
            'correct_samples': float(correct),
            'valid_samples': float(valid)
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
        """Конвертувати 9x9 grid в послідовність"""
        seq = []
        for row in grid:
            seq.extend(row)
        return seq
    
    def _sequence_to_grid(self, seq: np.ndarray, original_grid: List[List[int]]) -> List[List[int]]:
        """Конвертувати послідовність назад в 9x9 grid"""
        grid = []
        idx = 0
        for i in range(9):
            row = []
            for j in range(9):
                if idx < len(seq):
                    val = int(seq[idx])
                    row.append(max(1, min(9, val)))  # Sudoku: 1-9
                    idx += 1
                else:
                    row.append(original_grid[i][j] if i < len(original_grid) and j < len(original_grid[i]) else 0)
            grid.append(row)
        return grid
    
    def _is_valid_sudoku(self, grid: List[List[int]]) -> bool:
        """Перевірити чи є grid валідним Sudoku рішенням"""
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            return False
        
        # Перевірити рядки
        for row in grid:
            if not self._is_valid_unit(row):
                return False
        
        # Перевірити колонки
        for col in range(9):
            column = [grid[row][col] for row in range(9)]
            if not self._is_valid_unit(column):
                return False
        
        # Перевірити 3x3 блоки
        for box_row in range(3):
            for box_col in range(3):
                box = []
                for i in range(3):
                    for j in range(3):
                        box.append(grid[box_row * 3 + i][box_col * 3 + j])
                if not self._is_valid_unit(box):
                    return False
        
        return True
    
    def _is_valid_unit(self, unit: List[int]) -> bool:
        """Перевірити чи є одиниця (рядок/колонка/блок) валідною"""
        # Видалити 0 (пусті клітинки) та перевірити чи немає дублікатів
        values = [v for v in unit if v != 0]
        return len(values) == len(set(values)) and all(1 <= v <= 9 for v in values)
    
    def _compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Порівняти два grids та повернути частку співпадінь"""
        if not grid1 or not grid2:
            return 0.0
        
        if grid1 == grid2:
            return 1.0
        
        matches = 0
        total = 0
        
        for i in range(9):
            for j in range(9):
                val1 = grid1[i][j] if i < len(grid1) and j < len(grid1[i]) else 0
                val2 = grid2[i][j] if i < len(grid2) and j < len(grid2[i]) else 0
                
                if val1 == val2:
                    matches += 1
                total += 1
        
        return matches / total if total > 0 else 0.0

