"""
Скрипт для завантаження та конвертації OpenAssistant датасету
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Потрібно встановити datasets: pip install datasets")
    sys.exit(1)


def convert_to_format(dataset, max_examples=None, split_name="train"):
    """
    Конвертувати OpenAssistant датасет у формат проекту
    
    Args:
        dataset: Hugging Face dataset
        max_examples: Максимальна кількість прикладів (None = всі)
        split_name: Назва split'у для логування
    
    Returns:
        Список конвертованих прикладів
    """
    data = []
    count = 0
    
    print(f"🔄 Конвертація {split_name} датасету...")
    
    for item in dataset:
        # OpenAssistant має структуру з message_id, parent_id, text, role
        text = item.get('text', '')
        role = item.get('role', 'assistant')
        
        # Фільтр за мінімальною довжиною
        if len(text) > 50:
            # Створити context/query/completion структуру
            if len(text) > 200:
                # Розділити довгий текст на context та completion
                mid = len(text) // 2
                data.append({
                    'context': text[:mid],
                    'query': f'Continue this {role} message',
                    'completion': text[mid:]
                })
            else:
                # Для коротких текстів - створити пару з query
                data.append({
                    'context': '',
                    'query': f'Generate a {role} response',
                    'completion': text
                })
            
            count += 1
            if max_examples and count >= max_examples:
                break
        
        # Прогрес кожні 100 прикладів
        if count % 100 == 0 and count > 0:
            print(f"   ✅ Оброблено {count} прикладів...")
    
    return data


def main():
    """Головна функція"""
    print("=" * 80)
    print("📥 ЗАВАНТАЖЕННЯ ТА КОНВЕРТАЦІЯ OPENASSISTANT ДАТАСЕТУ")
    print("=" * 80)
    
    # Створити папки якщо не існують
    train_dir = project_root / "datasets" / "train"
    eval_dir = project_root / "datasets" / "eval"
    raw_dir = project_root / "datasets" / "raw"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Структура папок:")
    print(f"   Training: {train_dir}")
    print(f"   Eval: {eval_dir}")
    print(f"   Raw: {raw_dir}")
    
    # Завантажити датасети
    print("\n📥 Завантаження з Hugging Face...")
    try:
        train_dataset = load_dataset(
            "OpenAssistant/oasst_top1_2023-08-25", 
            split="train"
        )
        eval_dataset = load_dataset(
            "OpenAssistant/oasst_top1_2023-08-25", 
            split="test"  # Виправлено: датасет має split "test" замість "eval"
        )
        
        print(f"✅ Training: {len(train_dataset)} прикладів")
        print(f"✅ Eval: {len(eval_dataset)} прикладів")
    except Exception as e:
        print(f"❌ Помилка завантаження: {e}")
        return
    
    # Конвертувати training датасет
    print("\n" + "=" * 80)
    print("🔄 КОНВЕРТАЦІЯ TRAINING ДАТАСЕТУ")
    print("=" * 80)
    
    train_data = convert_to_format(train_dataset, max_examples=2000, split_name="train")
    
    # Зберегти training датасет
    train_output = train_dir / "openassistant_train.json"
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': 'OpenAssistant/oasst_top1_2023-08-25',
                'split': 'train',
                'num_examples': len(train_data),
                'original_size': len(train_dataset)
            },
            'data': train_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Training датасет збережено:")
    print(f"   📁 {train_output}")
    print(f"   📊 {len(train_data)} прикладів")
    
    # Конвертувати eval датасет
    print("\n" + "=" * 80)
    print("🔄 КОНВЕРТАЦІЯ EVAL ДАТАСЕТУ")
    print("=" * 80)
    
    eval_data = convert_to_format(eval_dataset, max_examples=500, split_name="eval")
    
    # Зберегти eval датасет
    eval_output = eval_dir / "openassistant_eval.json"
    with open(eval_output, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': 'OpenAssistant/oasst_top1_2023-08-25',
                'split': 'eval',
                'num_examples': len(eval_data),
                'original_size': len(eval_dataset)
            },
            'data': eval_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Eval датасет збережено:")
    print(f"   📁 {eval_output}")
    print(f"   📊 {len(eval_data)} прикладів")
    
    # Підсумок
    print("\n" + "=" * 80)
    print("✅ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"\n📊 Підсумок:")
    print(f"   Training: {len(train_data)} прикладів → {train_output}")
    print(f"   Eval: {len(eval_data)} прикладів → {eval_output}")
    print(f"\n🚀 Тепер можна навчати модель:")
    print(f"   python scripts/train_model.py --dataset datasets/train/openassistant_train.json")
    print(f"\n🧪 Або тестувати:")
    print(f"   python scripts/test_model.py --dataset datasets/eval/openassistant_eval.json")


if __name__ == "__main__":
    main()

