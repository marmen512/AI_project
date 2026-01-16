"""
CLI скрипт для оцінки моделі з різними evaluators
"""
import sys
import argparse
import torch
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tiny_recursive_model import TinyRecursiveModel
from tiny_recursive_model.utils import load_tokenizer
from inference.model_inference import load_trained_model
from train.evaluators import (
    BaseEvaluator,
    ARCEvaluator,
    SudokuEvaluator,
    MazeEvaluator,
    GeneralEvaluator
)
from train.datasets.trm_dataset import TRMDataset
from train.constants import DEFAULT_TOKENIZER_NAME


def get_evaluator(evaluator_type: str) -> BaseEvaluator:
    """
    Отримати evaluator за типом
    
    Args:
        evaluator_type: Тип evaluator ('arc', 'sudoku', 'maze', 'general')
    
    Returns:
        Evaluator instance
    """
    evaluator_type = evaluator_type.lower()
    
    if evaluator_type == 'arc':
        return ARCEvaluator()
    elif evaluator_type == 'sudoku':
        return SudokuEvaluator()
    elif evaluator_type == 'maze':
        return MazeEvaluator()
    elif evaluator_type == 'general':
        return GeneralEvaluator()
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(
        description="Оцінити навчену TRM модель з різними evaluators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Оцінити модель на загальних задачах:
  python scripts/evaluate_model.py --model models/trained/model.pt --evaluator general --dataset datasets/eval/data.json
  
  # Оцінити на ARC задачах:
  python scripts/evaluate_model.py --model models/trained/model.pt --evaluator arc --dataset datasets/puzzles/arc.json
  
  # Оцінити на Sudoku:
  python scripts/evaluate_model.py --model models/trained/model.pt --evaluator sudoku --dataset datasets/puzzles/sudoku.json
  
  # Оцінити на Maze:
  python scripts/evaluate_model.py --model models/trained/model.pt --evaluator maze --dataset datasets/puzzles/maze.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Шлях до навченої моделі (.pt файл)"
    )
    
    parser.add_argument(
        "--evaluator",
        type=str,
        required=True,
        choices=["arc", "sudoku", "maze", "general"],
        help="Тип evaluator для використання"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Шлях до датасету для оцінки"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Максимальна кількість прикладів для оцінки (за замовчуванням: всі)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Пристрій для оцінки (за замовчуванням: auto)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Шлях для збереження результатів (JSON файл, за замовчуванням: temp/evaluation_results.json)"
    )
    
    args = parser.parse_args()
    
    # Перевірити наявність файлів
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ Модель не знайдено: {model_path}")
        return
    
    dataset_path = project_root / args.dataset
    if not dataset_path.exists():
        print(f"❌ Датасет не знайдено: {dataset_path}")
        return
    
    # Визначити пристрій
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 80)
    print("📊 ОЦІНКА МОДЕЛІ")
    print("=" * 80)
    print(f"🤖 Модель: {model_path.name}")
    print(f"📚 Датасет: {dataset_path.name}")
    print(f"🔍 Evaluator: {args.evaluator}")
    print(f"💻 Пристрій: {device}")
    
    # Завантажити модель
    print(f"\n📥 Завантаження моделі...")
    try:
        tokenizer, _, _ = load_tokenizer(DEFAULT_TOKENIZER_NAME)
        
        inference = load_trained_model(
            model_path=str(model_path),
            device=device,
            tokenizer_name=DEFAULT_TOKENIZER_NAME
        )
        
        model = inference.model
        print(f"✅ Модель завантажено")
        
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Завантажити датасет
    print(f"\n📥 Завантаження датасету...")
    try:
        dataset = TRMDataset(
            data_path=dataset_path,
            tokenizer=tokenizer,
            max_seq_len=512
        )
        print(f"✅ Датасет завантажено: {len(dataset)} прикладів")
        
    except Exception as e:
        print(f"❌ Помилка завантаження датасету: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Створити evaluator
    print(f"\n🔍 Створення evaluator...")
    try:
        evaluator = get_evaluator(args.evaluator)
        print(f"✅ Evaluator створено: {args.evaluator}")
        
    except Exception as e:
        print(f"❌ Помилка створення evaluator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Запустити оцінку
    print(f"\n🚀 Початок оцінки...")
    try:
        results = evaluator.evaluate(
            model=model,
            dataset=dataset,
            max_samples=args.max_samples,
            tokenizer=tokenizer
        )
        
        # Вивести результати
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТИ ОЦІНКИ")
        print("=" * 80)
        print(evaluator.format_results(results))
        print("=" * 80)
        
        # Зберегти результати
        output_path = args.output
        if output_path is None:
            temp_dir = project_root / "temp"
            temp_dir.mkdir(exist_ok=True, parents=True)
            output_path = temp_dir / f"evaluation_results_{args.evaluator}.json"
        else:
            output_path = project_root / output_path
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результати збережено: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Помилка під час оцінки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

