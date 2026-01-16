"""
Модуль для швидкого тестування моделі з прогрес-баром
Використовується для інтерактивного тестування під час навчання
"""
import torch
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import time

from tiny_recursive_model.utils import load_tokenizer, tokenize_and_pad, prepare_code_input
from inference.model_inference import load_trained_model, TRMInference
from train.constants import DEFAULT_TOKENIZER_NAME


def quick_test_model(
    model_path: str,
    dataset_path: Optional[str] = None,
    num_samples: int = 10,
    device: str = "cpu",
    show_progress: bool = True
) -> Dict:
    """
    Швидкий тест моделі на обмеженій кількості прикладів з прогрес-баром
    
    Args:
        model_path: Шлях до моделі
        dataset_path: Шлях до eval датасету (опціонально, автоматично знайде)
        num_samples: Кількість прикладів для тестування
        device: Пристрій
        show_progress: Показувати прогрес-бар
    
    Returns:
        Словник з результатами: accuracy, avg_similarity, test_time, details
    """
    start_time = time.time()
    
    # Знайти eval датасет
    if dataset_path is None:
        # Визначити project_root правильно
        model_path_obj = Path(model_path).resolve()
        # Якщо model_path в checkpoints/, то project_root на один рівень вище
        if 'checkpoints' in model_path_obj.parts:
            # Знайти індекс 'checkpoints' і взяти все до нього
            parts = list(model_path_obj.parts)
            checkpoints_idx = parts.index('checkpoints')
            project_root = Path(*parts[:checkpoints_idx])
        else:
            # Якщо не в checkpoints, спробувати знайти project root іншим способом
            # Шукати папку з config/config.yaml
            current = model_path_obj.parent
            while current != current.parent:
                if (current / "config" / "config.yaml").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                # Fallback: припустити що project_root на 2 рівні вище
                project_root = model_path_obj.parent.parent
        
        eval_dir = project_root / "datasets" / "eval"
        eval_datasets = list(eval_dir.glob("*.json"))
        if eval_datasets:
            dataset_path = str(eval_datasets[0])
        else:
            # Спробувати train датасет як fallback
            train_dir = project_root / "datasets" / "train"
            train_datasets = list(train_dir.glob("*.json"))
            if train_datasets:
                dataset_path = str(train_datasets[0])
            else:
                return {
                    'error': 'Eval датасет не знайдено',
                    'accuracy': 0.0,
                    'avg_similarity': 0.0,
                    'test_time': 0.0
                }
    
    # Завантажити датасет
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            test_cases = data['data']
        elif isinstance(data, list):
            test_cases = data
        else:
            test_cases = []
        
        # Обмежити кількість
        test_cases = test_cases[:num_samples]
        
        if len(test_cases) == 0:
            return {
                'error': 'Немає тестових прикладів',
                'accuracy': 0.0,
                'avg_similarity': 0.0,
                'test_time': 0.0
            }
    except Exception as e:
        return {
            'error': f'Помилка завантаження датасету: {e}',
            'accuracy': 0.0,
            'avg_similarity': 0.0,
            'test_time': 0.0
        }
    
    # Завантажити модель
    try:
        tokenizer, _, _ = load_tokenizer(DEFAULT_TOKENIZER_NAME)
        inference = load_trained_model(
            model_path=model_path,
            device=device,
            tokenizer_name=DEFAULT_TOKENIZER_NAME
        )
    except Exception as e:
        return {
            'error': f'Помилка завантаження моделі: {e}',
            'accuracy': 0.0,
            'avg_similarity': 0.0,
            'test_time': 0.0
        }
    
    # Тестування з прогрес-баром
    correct = 0
    similarities = []
    details = []
    
    if show_progress:
        pbar = tqdm(total=len(test_cases), desc="Тестування", unit="тест", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, test_case in enumerate(test_cases):
        try:
            # Отримати context та completion
            context = test_case.get('context', test_case.get('input', ''))
            expected = test_case.get('completion', test_case.get('output', ''))
            query = test_case.get('query', 'Продовжи')
            
            # Генерувати відповідь
            result = inference.predict(
                context=context,
                query=query,
                max_deep_refinement_steps=12,
                halt_prob_thres=0.5
            )
            
            predicted = result.get('completion', '')
            
            # Обчислити схожість (простий підхід)
            similarity = _compute_similarity(predicted, expected)
            similarities.append(similarity)
            
            # Вважати правильним якщо схожість > 0.5
            if similarity > 0.5:
                correct += 1
            
            details.append({
                'test': i + 1,
                'similarity': similarity,
                'correct': similarity > 0.5
            })
            
        except Exception as e:
            similarities.append(0.0)
            details.append({
                'test': i + 1,
                'error': str(e),
                'similarity': 0.0,
                'correct': False
            })
        
        if show_progress:
            # Оновити опис з поточними метриками
            current_acc = (correct / (i + 1)) * 100
            current_avg = sum(similarities) / len(similarities) if similarities else 0.0
            pbar.set_postfix({
                'Accuracy': f'{current_acc:.1f}%',
                'Avg Sim': f'{current_avg:.2f}'
            })
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    # Обчислити фінальні метрики
    accuracy = (correct / len(test_cases)) * 100 if test_cases else 0.0
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    test_time = time.time() - start_time
    
    return {
        'accuracy': accuracy,
        'avg_similarity': avg_similarity,
        'test_time': test_time,
        'num_tests': len(test_cases),
        'correct': correct,
        'details': details
    }


def _compute_similarity(text1: str, text2: str) -> float:
    """
    Обчислити схожість між двома текстами (простий підхід)
    
    Args:
        text1: Перший текст
        text2: Другий текст
    
    Returns:
        Схожість (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Нормалізувати
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return 1.0
    
    # Схожість по словам
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    jaccard = intersection / union if union > 0 else 0.0
    
    # Схожість по символам (для коротких текстів)
    if len(text1) < 50 or len(text2) < 50:
        from difflib import SequenceMatcher
        char_similarity = SequenceMatcher(None, text1, text2).ratio()
        return (jaccard + char_similarity) / 2
    
    return jaccard


def full_test_model(
    model_path: str,
    dataset_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict:
    """
    Повний тест моделі (використовує scripts/test_model.py)
    
    Args:
        model_path: Шлях до моделі
        dataset_path: Шлях до eval датасету
        device: Пристрій
    
    Returns:
        Результати тестування
    """
    import subprocess
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    test_script = project_root / "scripts" / "test_model.py"
    
    cmd = [sys.executable, str(test_script), "--model", model_path, "--device", device]
    if dataset_path:
        cmd.extend(["--dataset", dataset_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Тест перевищив час очікування (5 хвилин)'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def print_test_results(results: Dict, test_type: str = "quick"):
    """
    Вивести результати тестування у консолі
    
    Args:
        results: Результати тестування
        test_type: Тип тесту ("quick" або "full")
    """
    print("\n" + "="*60)
    print(f"[TEST RESULTS] Результати {test_type} тестування")
    print("="*60)
    
    if 'error' in results:
        print(f"[ERROR] {results['error']}")
        return
    
    if test_type == "quick":
        print(f"Точність: {results['accuracy']:.1f}%")
        print(f"Середня схожість: {results['avg_similarity']:.2f}")
        print(f"Правильних: {results['correct']}/{results['num_tests']}")
        print(f"Час тестування: {results['test_time']:.2f} сек")
        
        # Деталі
        if 'details' in results:
            print("\nДеталі:")
            for detail in results['details'][:5]:  # Перші 5
                status = "[OK]" if detail.get('correct', False) else "[FAIL]"
                print(f"  {status} Тест {detail['test']}: схожість {detail.get('similarity', 0):.2f}")
    else:
        # Повний тест
        if results.get('success'):
            print("[OK] Тест завершено успішно")
            if 'stdout' in results:
                print("\nВихід:")
                print(results['stdout'][-500:])  # Останні 500 символів
        else:
            print("[ERROR] Тест не пройдено")
            if 'error' in results:
                print(f"Помилка: {results['error']}")
            if 'stderr' in results:
                print(f"Stderr: {results['stderr']}")
    
    print("="*60 + "\n")

