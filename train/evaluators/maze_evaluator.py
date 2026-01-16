"""
Maze Evaluator
Оцінка моделі на Maze задачах (пошук шляху в лабіринті)
"""
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np

from train.evaluators.base_evaluator import BaseEvaluator


class MazeEvaluator(BaseEvaluator):
    """
    Evaluator для Maze задач
    Перевіряє правильність знайденого шляху в лабіринті
    """
    
    def __init__(self):
        """Ініціалізація Maze evaluator"""
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
        Оцінити модель на Maze датасеті
        
        Args:
            model: TRM модель
            dataset: Maze датасет
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
        valid_paths = 0  # Валідні шляхи (з start до end, але можливо не оптимальні)
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    sample = dataset[i]
                    
                    # Maze формат: {'maze': 2D grid, 'start': (x, y), 'end': (x, y), 'solution': [(x, y), ...]}
                    maze = sample.get('maze', sample.get('input', []))
                    start = sample.get('start', (0, 0))
                    end = sample.get('end', (len(maze)-1, len(maze[0])-1))
                    expected_path = sample.get('solution', sample.get('output', []))
                    
                    if not maze:
                        continue
                    
                    # Конвертувати maze в послідовність (послідовність кроків)
                    maze_seq = self._maze_to_sequence(maze, start)
                    input_tensor = torch.tensor([maze_seq], dtype=torch.long)
                    
                    # Передбачення
                    if hasattr(model, 'predict'):
                        pred_output, exit_steps = model.predict(
                            input_tensor,
                            max_deep_refinement_steps=max_deep_refinement_steps,
                            halt_prob_thres=halt_prob_thres
                        )
                        pred_path = self._sequence_to_path(pred_output[0].cpu().numpy(), start, len(maze), len(maze[0]))
                    else:
                        pred_path = [start]
                    
                    # Перевірити правильність шляху
                    is_valid = self._is_valid_path(pred_path, maze, start, end)
                    is_correct = self._compare_paths(pred_path, expected_path) >= 1.0 if expected_path else False
                    
                    if is_valid:
                        valid_paths += 1
                    if is_correct:
                        correct += 1
                    
                    self.results.append({
                        'sample_id': i,
                        'correct': is_correct,
                        'valid': is_valid,
                        'path_length': len(pred_path),
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
        validity_rate = valid_paths / total if total > 0 else 0.0
        
        self.metrics_cache = {
            'accuracy': accuracy,
            'validity_rate': validity_rate,
            'total_samples': float(total),
            'correct_samples': float(correct),
            'valid_samples': float(valid_paths)
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
    
    def _maze_to_sequence(self, maze: List[List[int]], start: Tuple[int, int]) -> List[int]:
        """Конвертувати maze в послідовність"""
        seq = []
        for row in maze:
            seq.extend(row)
        # Додати стартову позицію
        seq.append(start[0] * 1000 + start[1])  # Кодування позиції
        return seq
    
    def _sequence_to_path(self, seq: np.ndarray, start: Tuple[int, int], height: int, width: int) -> List[Tuple[int, int]]:
        """Конвертувати послідовність в шлях (список координат)"""
        path = [start]
        x, y = start
        
        # Інтерпретувати послідовність як послідовність напрямків
        # 0=up, 1=down, 2=left, 3=right
        for val in seq[:100]:  # Обмежити довжину шляху
            direction = int(val) % 4
            dx, dy = 0, 0
            
            if direction == 0:  # up
                dx = -1
            elif direction == 1:  # down
                dx = 1
            elif direction == 2:  # left
                dy = -1
            elif direction == 3:  # right
                dy = 1
            
            new_x = x + dx
            new_y = y + dy
            
            # Перевірити межі
            if 0 <= new_x < height and 0 <= new_y < width:
                x, y = new_x, new_y
                path.append((x, y))
        
        return path
    
    def _is_valid_path(self, path: List[Tuple[int, int]], maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Перевірити чи є шлях валідним (з start до end, тільки через прохідні клітинки)"""
        if not path or path[0] != start:
            return False
        
        # Перевірити чи всі кроки валідні
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            
            # Перевірити чи крок сусідній
            dx = abs(curr[0] - next_pos[0])
            dy = abs(curr[1] - next_pos[1])
            if dx + dy != 1:
                return False
            
            # Перевірити чи клітинка прохідна (0 = стіна, 1 = прохідна)
            if next_pos[0] < 0 or next_pos[0] >= len(maze):
                return False
            if next_pos[1] < 0 or next_pos[1] >= len(maze[next_pos[0]]):
                return False
            
            if maze[next_pos[0]][next_pos[1]] == 0:  # Стіна
                return False
        
        # Перевірити чи шлях закінчується на end
        return path[-1] == end
    
    def _compare_paths(self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> float:
        """Порівняти два шляхи та повернути схожість"""
        if not path1 or not path2:
            return 0.0
        
        if path1 == path2:
            return 1.0
        
        # Порівняти кінцеві точки
        if path1[-1] != path2[-1]:
            return 0.0
        
        # Порівняти спільні точки
        set1 = set(path1)
        set2 = set(path2)
        intersection = set1 & set2
        union = set1 | set2
        
        return len(intersection) / len(union) if union else 0.0

