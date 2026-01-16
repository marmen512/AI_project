"""
Скрипт для тестування можливостей навченої моделі
"""

import torch
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D
from tiny_recursive_model.utils import load_tokenizer, tokenize_and_pad, prepare_code_input
import json
from pathlib import Path

def load_model(model_path="ai_assistant_model.pt", dim=256, depth=2, seq_len=512, vocab_size=50257):
    """Завантажити навчену модель"""
    print("Завантаження моделі...")
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✅ Модель завантажена")
    return model

def prepare_input(tokenizer, context, query, seq_len=512):
    """Підготувати вхід для моделі"""
    # Використовуємо утиліти з tiny_recursive_model
    input_text = prepare_code_input(context, query)
    
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
    
    return input_ids

def test_model(model, tokenizer, test_cases, seq_len=512):
    """Тестувати модель на різних завданнях"""
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"Контекст:\n{test_case['context']}")
        print(f"\nЗапит: {test_case['query']}")
        
        try:
            # Підготувати вхід
            input_ids = prepare_input(tokenizer, test_case['context'], test_case['query'], seq_len)
            
            # Передбачення
            with torch.no_grad():
                pred_tokens, exit_steps = model.predict(
                    input_ids,
                    max_deep_refinement_steps=12,
                    halt_prob_thres=0.5
                )
            
            # Декодувати результат
            output = tokenizer.decode(pred_tokens[0][:200], skip_special_tokens=True)
            steps = exit_steps[0].item() if len(exit_steps) > 0 else 0
            
            print(f"\n📤 Відповідь моделі:")
            print(f"{output[:500]}...")  # Перші 500 символів
            print(f"\n🔢 Кроки уточнення: {steps}")
            
            results.append({
                'test': test_case['name'],
                'input': test_case['query'],
                'output': output,
                'steps': steps,
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

def analyze_capabilities(results):
    """Аналіз можливостей моделі"""
    print("\n" + "="*60)
    print("АНАЛІЗ МОЖЛИВОСТЕЙ МОДЕЛІ")
    print("="*60)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"\n📊 Статистика:")
    print(f"   Успішних тестів: {successful}/{total}")
    print(f"   Помилок: {total - successful}")
    
    print(f"\n🔍 Аналіз відповідей:")
    for result in results:
        if result.get('success'):
            output = result['output']
            # Перевірити чи є змістовна відповідь
            has_code = any(c in output for c in ['def ', 'class ', 'import ', 'return ', '='])
            has_text = len(output.strip()) > 20
            
            print(f"\n   {result['test']}:")
            print(f"      - Кроки: {result['steps']}")
            print(f"      - Містить код: {'✅' if has_code else '❌'}")
            print(f"      - Довжина: {len(output)} символів")
        else:
            print(f"\n   {result['test']}: ❌ Помилка - {result.get('error', 'Unknown')}")

def main():
    # Тестові випадки
    test_cases = [
        {
            'name': 'Аналіз простої функції',
            'context': 'def add(a, b):\n    return a + b',
            'query': 'Проаналізуй цю функцію'
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
            'query': 'Додай обробку помилок'
        },
        {
            'name': 'Рефакторинг',
            'context': 'data = [1, 2, 3, 4, 5]\nresult = []\nfor x in data:\n    result.append(x * 2)',
            'query': 'Перепиши через list comprehension'
        },
        {
            'name': 'Складний запит',
            'context': 'import requests\n\ndef fetch_data(url):\n    response = requests.get(url)\n    return response.json()',
            'query': 'Додай timeout та обробку помилок'
        },
        {
            'name': 'Документація',
            'context': 'def process(items):\n    return [x for x in items if x > 0]',
            'query': 'Додай docstring та type hints'
        }
    ]
    
    print("="*60)
    print("ТЕСТУВАННЯ МОДЕЛІ AI-АСИСТЕНТА")
    print("="*60)
    
    # Завантажити токенізатор
    print("\n1. Завантаження токенізатора...")
    try:
        tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
        print(f"✅ GPT-2 tokenizer завантажено (vocab_size={vocab_size})")
    except Exception as e:
        print(f"❌ Помилка завантаження tokenizer: {e}")
        return
    
    # Завантажити модель
    print("\n2. Завантаження моделі...")
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        return
    
    # Тестування
    print("\n3. Запуск тестів...")
    results = test_model(model, tokenizer, test_cases, seq_len=512)
    
    # Аналіз
    analyze_capabilities(results)
    
    # Створити папку temp якщо не існує
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Зберегти результати
    results_path = temp_dir / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результати збережено в {results_path}")
    
    # Плюси та мінуси
    print("\n" + "="*60)
    print("ПЛЮСИ ТА МІНУСИ МОДЕЛІ")
    print("="*60)
    
    print("\n✅ ПЛЮСИ:")
    print("   1. Рекурсивне уточнення - модель покроково покращує відповіді")
    print("   2. Early stopping - автоматично визначає коли зупинитися")
    print("   3. Компактна архітектура - відносно мала модель")
    print("   4. Гнучкість - може працювати з різними типами завдань")
    print("   5. Latent states - зберігає проміжну інформацію")
    
    print("\n❌ МІНУСИ:")
    print("   1. Мало даних - навчена тільки на 21 прикладі")
    print("   2. Обмежена якість - потребує більше навчальних даних")
    print("   3. Мала модель - dim=256, depth=2 обмежує можливості")
    print("   4. Короткі послідовності - seq_len=512 обмежує контекст")
    print("   5. Низька точність - генерація може бути некоректною")
    print("   6. Відсутність контексту - не пам'ятає попередніх взаємодій")
    print("   7. Обмежена семантика - може не розуміти складні запити")
    
    print("\n💡 РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ:")
    print("   1. Збільшити датасет до 1000+ прикладів")
    print("   2. Використати більшу модель (dim=512-1024, depth=4-6)")
    print("   3. Збільшити seq_len до 2048-4096")
    print("   4. Навчити на більшій кількості епох (20+)")
    print("   5. Додати fine-tuning на специфічних завданнях")
    print("   6. Використати більш спеціалізований tokenizer для коду")
    print("   7. Додати контекстне навчання з історією діалогів")

if __name__ == "__main__":
    main()

