"""
Тестування навченої моделі з RAG (Retrieval-Augmented Generation)
Використовує релевантні приклади з датасету для покращення генерації
"""

import torch
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D
from tiny_recursive_model.utils import load_tokenizer, tokenize_and_pad, prepare_code_input
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class RAGSystem:
    """Простий RAG (Retrieval-Augmented Generation) для зберігання та пошуку прикладів"""
    
    def __init__(self, max_examples: int = 50):
        self.examples = []
        self.max_examples = max_examples
    
    def add_example(self, context: str, query: str, completion: str):
        """Додати приклад до RAG"""
        self.examples.append({
            'context': context,
            'query': query,
            'completion': completion
        })
        # Зберігати тільки останні N прикладів
        if len(self.examples) > self.max_examples:
            self.examples.pop(0)
    
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Знайти схожі приклади (простий пошук по ключових словах)"""
        if not self.examples:
            return []
        
        query_lower = query.lower()
        scored = []
        
        for ex in self.examples:
            score = 0
            # Простий підрахунок спільних слів
            ex_text = (ex['context'] + ' ' + ex['query']).lower()
            for word in query_lower.split():
                if word in ex_text:
                    score += 1
            scored.append((score, ex))
        
        # Сортувати за релевантністю
        scored.sort(reverse=True, key=lambda x: x[0])
        return [ex for _, ex in scored[:top_k]]


def load_rag_examples(dataset_path: Path, max_examples: int = 50) -> RAGSystem:
    """
    Завантажити приклади з датасету для RAG
    
    Args:
        dataset_path: Шлях до датасету (JSON файл)
        max_examples: Максимальна кількість прикладів для завантаження
        
    Returns:
        RAGSystem з завантаженими прикладами
    """
    rag = RAGSystem(max_examples=max_examples)
    
    if not dataset_path.exists():
        print(f"⚠️  Датасет не знайдено: {dataset_path}")
        return rag
    
    print(f"📚 Завантаження прикладів для RAG з {dataset_path.name}...")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Підтримка різних форматів датасету
        if isinstance(data, dict) and 'data' in data:
            examples = data['data']
        elif isinstance(data, list):
            examples = data
        else:
            examples = []
        
        # Обмежити кількість прикладів
        examples = examples[:max_examples]
        
        # Додати приклади до RAG
        for ex in examples:
            context = ex.get('context', '')
            query = ex.get('query', '')
            completion = ex.get('completion', '')
            
            if context or query or completion:
                rag.add_example(context, query, completion)
        
        print(f"✅ Завантажено {len(rag.examples)} прикладів для RAG")
        
    except Exception as e:
        print(f"⚠️  Помилка завантаження датасету: {e}")
    
    return rag


def format_rag_context(similar_examples: List[Dict]) -> str:
    """
    Форматувати RAG контекст зі схожих прикладів
    
    Args:
        similar_examples: Список схожих прикладів
        
    Returns:
        Відформатований RAG контекст
    """
    if not similar_examples:
        return ""
    
    rag_context = "\n\nПриклади схожих завдань:\n"
    for idx, ex in enumerate(similar_examples, 1):
        context_preview = ex['context'][:150] + "..." if len(ex['context']) > 150 else ex['context']
        completion_preview = ex['completion'][:150] + "..." if len(ex['completion']) > 150 else ex['completion']
        
        rag_context += f"{idx}. Контекст: {context_preview}\n"
        rag_context += f"   Запит: {ex['query']}\n"
        rag_context += f"   Відповідь: {completion_preview}\n\n"
    
    return rag_context


def load_model(model_path="trm_optimized.pt", dim=512, depth=4, seq_len=2048, vocab_size=50257):
    """Завантажити навчену модель"""
    print("📦 Завантаження моделі...")
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"✅ Модель завантажена: {model_path}")
    return model

def test_comprehensive(model, tokenizer, test_cases, rag_system: Optional[RAGSystem] = None, 
                      rag_top_k: int = 3, seq_len=2048):
    """
    Комплексне тестування моделі з RAG підтримкою
    
    Args:
        model: Навчена модель
        tokenizer: Токенізатор
        test_cases: Список тестових випадків
        rag_system: RAG система для пошуку схожих прикладів (опціонально)
        rag_top_k: Кількість схожих прикладів для використання
        seq_len: Довжина послідовності
    """
    results = []
    
    print("\n" + "="*70)
    print("🧪 КОМПЛЕКСНЕ ТЕСТУВАННЯ МОДЕЛІ" + (" З RAG" if rag_system else ""))
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"ТЕСТ {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'─'*70}")
        print(f"📝 Контекст:\n{test_case['context']}")
        print(f"\n❓ Запит: {test_case['query']}")
        
        try:
            # RAG: Знайти схожі приклади
            rag_context = ""
            used_rag = False
            similar_examples = []
            if rag_system:
                similar_examples = rag_system.retrieve_similar(test_case['query'], top_k=rag_top_k)
                if similar_examples:
                    rag_context = format_rag_context(similar_examples)
                    used_rag = True
                    print(f"\n🔍 RAG: Знайдено {len(similar_examples)} схожих прикладів")
            
            # Підготувати вхід (з RAG контекстом якщо є)
            base_input = prepare_code_input(test_case['context'], test_case['query'])
            input_text = rag_context + base_input if rag_context else base_input
            
            # Отримати pad_token_id
            pad_token_id = 0
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                pad_token_id = tokenizer.pad_token_id
            
            # Токенізувати та додати padding
            input_ids = tokenize_and_pad(
                tokenizer,
                input_text,
                seq_len,
                pad_token_id=pad_token_id,
                truncation=True
            ).unsqueeze(0)
            
            # Передбачення
            print("\n🤖 Генерація відповіді...")
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            else:
                import time
                start_cpu = time.time()
            
            with torch.no_grad():
                pred_tokens, exit_steps = model.predict(
                    input_ids,
                    max_deep_refinement_steps=12,
                    halt_prob_thres=0.5
                )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                gen_time = start_time.elapsed_time(end_time) / 1000
            else:
                gen_time = time.time() - start_cpu
            
            # Декодувати
            output = tokenizer.decode(pred_tokens[0][:300], skip_special_tokens=True)
            steps = exit_steps[0].item() if len(exit_steps) > 0 else 0
            
            print(f"\n📤 Відповідь моделі:")
            print(f"{output[:500]}{'...' if len(output) > 500 else ''}")
            print(f"\n📊 Статистика:")
            print(f"   - Кроки уточнення: {steps}")
            print(f"   - Час генерації: {gen_time:.2f}с")
            print(f"   - Довжина відповіді: {len(output)} символів")
            
            # Аналіз якості
            has_code = any(c in output for c in ['def ', 'class ', 'import ', 'return ', '='])
            has_structure = any(c in output for c in ['\n', '    ', '(', ')'])
            
            print(f"   - Містить код: {'✅' if has_code else '❌'}")
            print(f"   - Структурована: {'✅' if has_structure else '❌'}")
            
            results.append({
                'test': test_case['name'],
                'input': test_case['query'],
                'output': output,
                'steps': steps,
                'time': gen_time,
                'has_code': has_code,
                'has_structure': has_structure,
                'used_rag': used_rag,
                'rag_examples_count': len(similar_examples),
                'success': True
            })
            
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
            results.append({
                'test': test_case['name'],
                'input': test_case['query'],
                'error': str(e),
                'success': False
            })
    
    return results

def generate_report(results, model_path, rag_enabled: bool = False):
    """Генерувати звіт"""
    print("\n" + "="*70)
    print("📊 ЗВІТ ПРО ТЕСТУВАННЯ")
    print("="*70)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"\n📈 Загальна статистика:")
    print(f"   Успішних тестів: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"   Помилок: {total - successful}")
    
    if successful > 0:
        avg_steps = sum(r.get('steps', 0) for r in results if r.get('success')) / successful
        avg_time = sum(r.get('time', 0) for r in results if r.get('success')) / successful
        code_count = sum(1 for r in results if r.get('has_code', False))
        
        print(f"\n📊 Середні показники:")
        print(f"   Середня кількість кроків: {avg_steps:.1f}")
        print(f"   Середній час генерації: {avg_time:.2f}с")
        print(f"   Тестів з кодом: {code_count}/{successful} ({code_count/successful*100:.1f}%)")
        
        # RAG статистика
        if rag_enabled:
            rag_used_count = sum(1 for r in results if r.get('used_rag', False))
            avg_rag_examples = sum(r.get('rag_examples_count', 0) for r in results if r.get('success')) / successful
            print(f"\n🔍 RAG статистика:")
            print(f"   Тестів з RAG: {rag_used_count}/{successful} ({rag_used_count/successful*100:.1f}%)")
            print(f"   Середня кількість RAG прикладів: {avg_rag_examples:.1f}")
    
    # Зберегти звіт
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': model_path,
        'total_tests': total,
        'successful': successful,
        'rag_enabled': rag_enabled,
        'results': results
    }
    
    # Створити папку temp якщо не існує
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    report_path = temp_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Звіт збережено: {report_path}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Тестування навченої моделі з RAG (Retrieval-Augmented Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Тестування з RAG (автоматично знайде eval датасет):
  python scripts/test_with_rag.py --model models/trained/openassistant_train.pt
  
  # Тестування з конкретним RAG датасетом:
  python scripts/test_with_rag.py \\
      --model models/trained/openassistant_train.pt \\
      --rag-dataset datasets/eval/openassistant_eval.json
  
  # Тестування без RAG:
  python scripts/test_with_rag.py \\
      --model models/trained/openassistant_train.pt \\
      --disable-rag
        """
    )
    parser.add_argument("--model", type=str, default="trm_optimized.pt",
                       help="Шлях до моделі")
    parser.add_argument("--dim", type=int, default=512,
                       help="Розмірність моделі")
    parser.add_argument("--depth", type=int, default=4,
                       help="Глибина")
    parser.add_argument("--seq-len", type=int, default=2048,
                       help="Довжина послідовності")
    parser.add_argument("--rag-dataset", type=str, default=None,
                       help="Шлях до датасету для RAG (за замовчуванням: автоматично знайти в datasets/eval/)")
    parser.add_argument("--rag-top-k", type=int, default=3,
                       help="Кількість схожих прикладів для використання в RAG (за замовчуванням: 3)")
    parser.add_argument("--rag-max-examples", type=int, default=50,
                       help="Максимальна кількість прикладів для завантаження в RAG (за замовчуванням: 50)")
    parser.add_argument("--disable-rag", action="store_true",
                       help="Вимкнути RAG (для порівняння результатів)")
    
    args = parser.parse_args()
    
    # Визначити project_root
    project_root = Path(__file__).parent.parent
    
    # Тестові випадки
    test_cases = [
        {
            'name': 'Аналіз функції',
            'context': 'def add(a, b):\n    return a + b',
            'query': 'Проаналізуй цю функцію та додай docstring'
        },
        {
            'name': 'Оптимізація коду',
            'context': 'def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total',
            'query': 'Оптимізуй цю функцію'
        },
        {
            'name': 'Додавання функціональності',
            'context': 'class User:\n    def __init__(self, name):\n        self.name = name',
            'query': 'Додай метод get_email'
        },
        {
            'name': 'Обробка помилок',
            'context': 'def divide(a, b):\n    return a / b',
            'query': 'Додай обробку помилок для ділення на нуль'
        },
        {
            'name': 'Рефакторинг',
            'context': 'data = [1, 2, 3, 4, 5]\nresult = []\nfor x in data:\n    result.append(x ** 2)',
            'query': 'Перепиши через list comprehension'
        }
    ]
    
    print("="*70)
    print("🧪 ТЕСТУВАННЯ НАВЧЕНОЇ МОДЕЛІ")
    print("="*70)
    
    # Завантажити токенізатор
    print("\n1. Завантаження токенізатора...")
    try:
        tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
        print(f"✅ GPT-2 tokenizer завантажено (vocab_size={vocab_size})")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        exit(1)
    
    # Завантажити модель
    print("\n2. Завантаження моделі...")
    try:
        model_path = project_root / args.model if not Path(args.model).is_absolute() else Path(args.model)
        model = load_model(str(model_path), args.dim, args.depth, args.seq_len)
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        exit(1)
    
    # Завантажити RAG систему (якщо не вимкнено)
    rag_system = None
    if not args.disable_rag:
        print("\n3. Завантаження RAG системи...")
        rag_dataset_path = None
        
        if args.rag_dataset:
            rag_dataset_path = project_root / args.rag_dataset if not Path(args.rag_dataset).is_absolute() else Path(args.rag_dataset)
        else:
            # Автоматично знайти eval датасет
            eval_dir = project_root / "datasets" / "eval"
            eval_datasets = list(eval_dir.glob("*.json"))
            if eval_datasets:
                rag_dataset_path = eval_datasets[0]
                print(f"📚 Автоматично вибрано eval датасет: {rag_dataset_path.name}")
            else:
                # Спробувати train датасет
                train_dir = project_root / "datasets" / "train"
                train_datasets = list(train_dir.glob("*.json"))
                if train_datasets:
                    rag_dataset_path = train_datasets[0]
                    print(f"📚 Використовується train датасет: {rag_dataset_path.name}")
        
        if rag_dataset_path and rag_dataset_path.exists():
            rag_system = load_rag_examples(rag_dataset_path, max_examples=args.rag_max_examples)
            if len(rag_system.examples) == 0:
                print("⚠️  Не вдалося завантажити приклади для RAG, RAG вимкнено")
                rag_system = None
        else:
            print("⚠️  Датасет для RAG не знайдено, RAG вимкнено")
            rag_system = None
    else:
        print("\n3. RAG вимкнено (--disable-rag)")
    
    # Тестування
    print("\n4. Запуск тестів...")
    results = test_comprehensive(
        model, 
        tokenizer, 
        test_cases, 
        rag_system=rag_system,
        rag_top_k=args.rag_top_k,
        seq_len=args.seq_len
    )
    
    # Звіт
    report = generate_report(results, args.model, rag_enabled=rag_system is not None)
    
    print("\n" + "="*70)
    print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО")
    print("="*70)

