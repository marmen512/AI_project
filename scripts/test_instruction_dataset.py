#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки InstructionDataset
Перевіряє правильність маскування labels для instruction tuning
"""

import sys
from pathlib import Path
import json

# Додати project root до sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_instruction_dataset():
    """Тест InstructionDataset з правильним маскуванням"""
    
    try:
        import torch
        from transformers import GPT2Tokenizer
    except ImportError as e:
        print(f"❌ Помилка імпорту: {e}")
        print("   Активуйте віртуальне середовище: source venv/bin/activate")
        return False
    
    # Імпортувати InstructionDataset
    sys.path.insert(0, str(project_root / "scripts"))
    from train_phase2_instruction_tuning import InstructionDataset
    
    print("🧪 Тестування InstructionDataset...")
    
    # Створити тестовий dataset файл
    test_data = [
        {
            "instruction": "Explain what is AI",
            "input": "",
            "output": "AI is artificial intelligence that simulates human thinking."
        },
        {
            "instruction": "Translate to Ukrainian",
            "input": "Hello world",
            "output": "Привіт світ"
        }
    ]
    
    test_file = project_root / "test_instruction_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"   📄 Створено тестовий файл: {test_file}")
    
    # Завантажити tokenizer
    print("   🔤 Завантаження GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Створити dataset
    print("   📚 Створення InstructionDataset...")
    dataset = InstructionDataset(
        data_files=[str(test_file)],
        tokenizer=tokenizer,
        max_seq_len=64  # Коротка довжина для тесту
    )
    
    print(f"   📊 Dataset розмір: {len(dataset)}")
    
    # Тестувати перший sample
    print("\n🔍 Тестування першого sample...")
    sample = dataset[0]
    
    input_ids = sample['input_ids']
    labels = sample['labels']
    
    print(f"   📏 Input IDs shape: {input_ids.shape}")
    print(f"   📏 Labels shape: {labels.shape}")
    
    # Декодувати для перевірки
    print("\n📝 Декодування для перевірки:")
    
    # Знайти де починається output (перші не-masked labels)
    first_label_idx = None
    for i, label in enumerate(labels):
        if label != -100:
            first_label_idx = i
            break
    
    if first_label_idx is not None:
        # Контекст (має бути замаскований)
        context_ids = input_ids[:first_label_idx]
        context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
        print(f"   🔒 Контекст (masked): {context_text}")
        
        # Output (має навчатися)
        output_labels = labels[first_label_idx:]
        output_labels_clean = [l for l in output_labels if l != -100]
        if output_labels_clean:
            output_text = tokenizer.decode(output_labels_clean, skip_special_tokens=True)
            print(f"   🎯 Output (supervised): {output_text}")
        
        # Перевірити маскування
        masked_count = sum(1 for l in labels if l == -100)
        supervised_count = sum(1 for l in labels if l != -100 and l != tokenizer.pad_token_id)
        
        print(f"\n📊 Статистика маскування:")
        print(f"   🔒 Masked tokens (контекст): {masked_count}")
        print(f"   🎯 Supervised tokens (output): {supervised_count}")
        print(f"   📏 Загальна довжина: {len(labels)}")
        
        if masked_count > 0 and supervised_count > 0:
            print("   ✅ Маскування правильне!")
        else:
            print("   ❌ Проблема з маскуванням!")
            return False
    else:
        print("   ❌ Не знайдено supervised labels!")
        return False
    
    # Очистити тестовий файл
    test_file.unlink()
    print(f"\n🧹 Видалено тестовий файл: {test_file}")
    
    print("\n✅ Тест InstructionDataset пройшов успішно!")
    return True

if __name__ == "__main__":
    success = test_instruction_dataset()
    sys.exit(0 if success else 1)


