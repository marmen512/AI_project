#!/usr/bin/env python3
"""
Універсальний конвертер HuggingFace датасетів у внутрішній JSON формат
Конвертує: Alpaca, DailyDialog, Simple Wikipedia, SQuAD v1, SQuAD v2
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys

def install_datasets():
    """Встановити datasets якщо не встановлено"""
    try:
        import datasets
        print("✅ datasets вже встановлено")
        return True
    except ImportError:
        print("📦 Встановлюємо datasets...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            print("✅ datasets встановлено")
            return True
        except Exception as e:
            print(f"❌ Помилка встановлення datasets: {e}")
            return False

def save_dataset(name: str, dtype: str, data: List[Dict], out_dir: Path):
    """Зберегти датасет у внутрішньому форматі"""
    out = {
        "metadata": {
            "name": name,
            "type": dtype,
            "source": "huggingface",
            "size": len(data),
            "description": f"Converted from HuggingFace dataset"
        },
        "data": data
    }
    
    output_file = out_dir / f"{name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {name}.json створено ({len(data):,} записів)")
    return output_file

def convert_alpaca(out_dir: Path):
    """Конвертувати Alpaca dataset"""
    print("\n📚 Завантаження Alpaca...")
    from datasets import load_dataset
    
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        print(f"   Завантажено {len(alpaca):,} записів")
        
        alpaca_data = []
        for x in alpaca:
            # Обробити input - може бути порожнім
            input_text = x.get("input", "").strip()
            
            alpaca_data.append({
                "instruction": x["instruction"].strip(),
                "input": input_text,
                "output": x["output"].strip()
            })
        
        return save_dataset("alpaca", "instruction", alpaca_data, out_dir)
        
    except Exception as e:
        print(f"❌ Помилка завантаження Alpaca: {e}")
        return None

def convert_dailydialog(out_dir: Path):
    """Конвертувати DailyDialog dataset"""
    print("\n💬 Завантаження DailyDialog...")
    from datasets import load_dataset
    
    try:
        # Спробувати альтернативний датасет
        dd = load_dataset("daily_dialog", split="train")
        print(f"   Завантажено {len(dd):,} діалогів")
        
        dd_data = []
        for dialog in dd:
            utterances = dialog["dialog"]
            # Створити пари запит-відповідь з діалогу
            for i in range(len(utterances) - 1):
                if utterances[i].strip() and utterances[i + 1].strip():
                    dd_data.append({
                        "instruction": utterances[i].strip(),
                        "input": "",
                        "output": utterances[i + 1].strip()
                    })
        
        return save_dataset("dailydialog", "dialog", dd_data, out_dir)
        
    except Exception as e:
        print(f"❌ Помилка завантаження DailyDialog: {e}")
        # Створити мінімальний діалог датасет
        print("   Створюємо мінімальний діалог датасет...")
        minimal_data = [
            {"instruction": "Hello", "input": "", "output": "Hi there! How can I help you?"},
            {"instruction": "How are you?", "input": "", "output": "I'm doing well, thank you for asking!"},
            {"instruction": "What's your name?", "input": "", "output": "I'm an AI assistant here to help you."},
            {"instruction": "Goodbye", "input": "", "output": "Goodbye! Have a great day!"}
        ]
        return save_dataset("dailydialog_minimal", "dialog", minimal_data, out_dir)

def convert_simple_wikipedia(out_dir: Path):
    """Конвертувати Simple Wikipedia dataset"""
    print("\n📖 Завантаження Simple Wikipedia...")
    from datasets import load_dataset
    
    try:
        wiki = load_dataset("rahular/simple-wikipedia", split="train")
        print(f"   Завантажено {len(wiki):,} статей")
        
        wiki_data = []
        for i, x in enumerate(wiki):
            if i >= 10000:  # Обмежити кількість для швидкості
                break
                
            text = x["text"].strip()
            
            if text and len(text) > 100:  # Фільтр коротких текстів
                # Використати перші слова як "title"
                words = text.split()
                if len(words) > 5:
                    title = " ".join(words[:3])  # Перші 3 слова як заголовок
                    content = text[:1500]  # Обмежити довжину
                    
                    wiki_data.append({
                        "instruction": f"Explain: {title}",
                        "input": "",
                        "output": content
                    })
        
        return save_dataset("simple_wiki", "knowledge", wiki_data, out_dir)
        
    except Exception as e:
        print(f"❌ Помилка завантаження Simple Wikipedia: {e}")
        print(f"   Доступні поля: {list(wiki[0].keys()) if len(wiki) > 0 else 'немає даних'}")
        
        # Створити мінімальний knowledge датасет
        print("   Створюємо мінімальний knowledge датасет...")
        minimal_data = [
            {"instruction": "Explain: Artificial Intelligence", "input": "", "output": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."},
            {"instruction": "Explain: Machine Learning", "input": "", "output": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."},
            {"instruction": "Explain: Neural Networks", "input": "", "output": "Neural Networks are computing systems inspired by biological neural networks that constitute animal brains."},
            {"instruction": "Explain: Deep Learning", "input": "", "output": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns."}
        ]
        return save_dataset("simple_wiki_minimal", "knowledge", minimal_data, out_dir)

def convert_squad_v1(out_dir: Path):
    """Конвертувати SQuAD v1.1 dataset"""
    print("\n❓ Завантаження SQuAD v1.1...")
    from datasets import load_dataset
    
    try:
        squad = load_dataset("rajpurkar/squad", split="train")
        print(f"   Завантажено {len(squad):,} питань")
        
        squad_data = []
        for x in squad:
            question = x["question"].strip()
            context = x["context"].strip()
            answers = x["answers"]["text"]
            
            if question and context and answers:
                squad_data.append({
                    "instruction": question,
                    "input": context,
                    "output": answers[0].strip()  # Перша відповідь
                })
        
        return save_dataset("squad", "qa", squad_data, out_dir)
        
    except Exception as e:
        print(f"❌ Помилка завантаження SQuAD v1: {e}")
        return None

def convert_squad_v2(out_dir: Path):
    """Конвертувати SQuAD v2.0 dataset"""
    print("\n❓ Завантаження SQuAD v2.0...")
    from datasets import load_dataset
    
    try:
        squad2 = load_dataset("rajpurkar/squad_v2", split="train")
        print(f"   Завантажено {len(squad2):,} питань")
        
        squad2_data = []
        for x in squad2:
            question = x["question"].strip()
            context = x["context"].strip()
            answers = x["answers"]["text"]
            
            # Тільки питання з відповідями (не impossible)
            if question and context and answers:
                squad2_data.append({
                    "instruction": question,
                    "input": context,
                    "output": answers[0].strip()
                })
        
        return save_dataset("squad_v2", "qa", squad2_data, out_dir)
        
    except Exception as e:
        print(f"❌ Помилка завантаження SQuAD v2: {e}")
        return None

def main():
    """Головна функція конвертації"""
    print("=" * 80)
    print("🔄 УНІВЕРСАЛЬНИЙ КОНВЕРТЕР HF → JSON")
    print("=" * 80)
    
    # Перевірити/встановити datasets
    if not install_datasets():
        return
    
    # Створити вихідну папку
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Вихідна папка: {out_dir.absolute()}")
    
    # Список конверторів
    converters = [
        ("Alpaca", convert_alpaca),
        ("DailyDialog", convert_dailydialog),
        ("Simple Wikipedia", convert_simple_wikipedia),
        ("SQuAD v1.1", convert_squad_v1),
        ("SQuAD v2.0", convert_squad_v2)
    ]
    
    # Конвертувати всі датасети
    converted = []
    failed = []
    
    for name, converter in converters:
        try:
            result = converter(out_dir)
            if result:
                converted.append((name, result))
            else:
                failed.append(name)
        except Exception as e:
            print(f"❌ Критична помилка в {name}: {e}")
            failed.append(name)
    
    # Підсумок
    print("\n" + "=" * 80)
    print("📊 ПІДСУМОК КОНВЕРТАЦІЇ")
    print("=" * 80)
    
    if converted:
        print(f"\n✅ Успішно конвертовано ({len(converted)}):")
        total_records = 0
        for name, filepath in converted:
            # Прочитати розмір
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                size = len(data['data'])
                total_records += size
                print(f"   📄 {filepath.name} - {size:,} записів")
        
        print(f"\n📈 Загалом: {total_records:,} записів")
    
    if failed:
        print(f"\n❌ Не вдалося конвертувати ({len(failed)}):")
        for name in failed:
            print(f"   ⚠️ {name}")
    
    if converted:
        print(f"\n🎯 НАСТУПНІ КРОКИ:")
        print(f"   1. Перевірити файли в папці datasets/")
        print(f"   2. Запустити навчання: ./start_training.sh")
        print(f"   3. Моніторити прогрес: python scripts/check_training_status.py")
    
    print("\n✨ Конвертація завершена!")

if __name__ == "__main__":
    main()
