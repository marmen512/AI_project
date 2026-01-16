#!/usr/bin/env python3
"""
Скрипт для підготовки ФАЗИ 1 - Language Pretraining Dataset
Завантажує Simple Wikipedia та створює plain text corpus для causal language modeling

КРИТИЧНІ ВИМОГИ:
- Тільки plain text (БЕЗ instruction format)
- Заміна newlines на spaces
- Обмеження до ~15-20M токенів (CPU-safe)
- Збереження в datasets/pretrain_text.txt
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer
import re

# Додати project root до sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def clean_text(text: str) -> str:
    """
    Очищення тексту для language pretraining
    - Заміна newlines на spaces
    - Видалення зайвих пробілів
    - Видалення спеціальних символів
    """
    # Заміна newlines на spaces (як вимагається)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Видалення зайвих пробілів
    text = re.sub(r'\s+', ' ', text)
    
    # Видалення спеціальних символів та залишення тільки основного тексту
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"]+', ' ', text)
    
    # Видалення зайвих пробілів знову
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("🚀 ФАЗА 1 - Підготовка Language Pretraining Dataset")
    print("=" * 60)
    
    # Налаштування
    output_file = project_root / "datasets" / "pretrain_text.txt"
    target_tokens = 20_000_000  # 20M токенів (CPU-safe)
    min_text_length = 50  # Мінімальна довжина тексту
    
    # Створити папку datasets якщо не існує
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"📁 Вихідний файл: {output_file}")
    print(f"🎯 Цільова кількість токенів: {target_tokens:,}")
    
    # Завантажити GPT-2 tokenizer для підрахунку токенів
    print("\n🔤 Завантаження GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Завантажити Simple Wikipedia
    print("\n📖 Завантаження Simple Wikipedia dataset...")
    try:
        dataset = load_dataset("rahular/simple-wikipedia", split="train")
        print(f"   ✅ Завантажено {len(dataset):,} статей")
    except Exception as e:
        print(f"   ❌ Помилка завантаження: {e}")
        print("   🔄 Спробуйте: pip install datasets")
        return False
    
    # Обробка та збереження
    print(f"\n📝 Обробка статей (ціль: {target_tokens:,} токенів)...")
    
    total_tokens = 0
    processed_articles = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, article in enumerate(dataset):
            if total_tokens >= target_tokens:
                break
                
            # Витягти текст статті
            text = article.get('text', '').strip()
            
            if len(text) < min_text_length:
                continue
                
            # Очистити текст
            cleaned_text = clean_text(text)
            
            if len(cleaned_text) < min_text_length:
                continue
            
            # Підрахувати токени
            tokens = tokenizer.encode(cleaned_text)
            token_count = len(tokens)
            
            # Перевірити чи не перевищуємо ліміт
            if total_tokens + token_count > target_tokens:
                # Обрізати текст щоб влізти в ліміт
                remaining_tokens = target_tokens - total_tokens
                if remaining_tokens > 100:  # Тільки якщо залишилось достатньо місця
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = tokenizer.decode(truncated_tokens)
                    f.write(truncated_text + ' ')
                    total_tokens += len(truncated_tokens)
                break
            
            # Записати текст
            f.write(cleaned_text + ' ')
            total_tokens += token_count
            processed_articles += 1
            
            # Прогрес
            if processed_articles % 1000 == 0:
                progress = (total_tokens / target_tokens) * 100
                print(f"   📊 Оброблено: {processed_articles:,} статей, "
                      f"токенів: {total_tokens:,} ({progress:.1f}%)")
    
    print(f"\n✅ Завершено!")
    print(f"   📄 Файл: {output_file}")
    print(f"   📊 Статей оброблено: {processed_articles:,}")
    print(f"   🔤 Загальна кількість токенів: {total_tokens:,}")
    print(f"   📏 Розмір файлу: {output_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Перевірка результату
    print(f"\n🔍 Перевірка результату...")
    with open(output_file, 'r', encoding='utf-8') as f:
        sample = f.read(500)
        print(f"   📝 Перші 500 символів:")
        print(f"   {sample}...")
    
    print(f"\n🎯 ФАЗА 1 готова до навчання!")
    print(f"   Використовуйте файл: {output_file}")
    print(f"   Для causal language modeling на CPU")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


