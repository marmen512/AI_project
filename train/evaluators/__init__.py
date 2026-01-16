"""
Evaluators для різних типів задач
"""
from train.evaluators.base_evaluator import BaseEvaluator
from train.evaluators.arc_evaluator import ARCEvaluator
from train.evaluators.sudoku_evaluator import SudokuEvaluator
from train.evaluators.maze_evaluator import MazeEvaluator
from train.evaluators.general_evaluator import GeneralEvaluator

__all__ = [
    "BaseEvaluator",
    "ARCEvaluator",
    "SudokuEvaluator",
    "MazeEvaluator",
    "GeneralEvaluator"
]

