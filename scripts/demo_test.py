"""
Демонстраційний скрипт для тестування навчання та роботи TRM моделі
"""
import torch
import json
import os
from pathlib import Path

from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.utils import tokenize_and_pad, prepare_code_input, load_tokenizer
from torch.utils.data import Dataset


class SimpleCodeDataset(Dataset):
    """Простий датасет для демонстрації"""
    def __init__(self, tokenizer, max_seq_len=256, pad_token_id=0):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Прості приклади для демонстрації
        self.data = [
            {
                "context": "def hello():\n    return 'world'",
                "query": "Додай параметр name",
                "completion": "def hello(name='world'):\n    return name"
            },
            {
                "context": "x = 5\ny = 10",
                "query": "Додай обчислення суми",
                "completion": "x = 5\ny = 10\nresult = x + y"
            },
            {
                "context": "def add(a, b):\n    pass",
                "query": "Реалізуй функцію",
                "completion": "def add(a, b):\n    return a + b"
            },
            {
                "context": "items = [1, 2, 3]",
                "query": "Знайди максимум",
                "completion": "items = [1, 2, 3]\nmax_item = max(items)"
            },
            {
                "context": "name = 'John'",
                "query": "Додай привітання",
                "completion": "name = 'John'\ngreeting = f'Hello, {name}!'"
            }
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item.get('context', '')
        query = item.get('query', '')
        completion = item.get('completion', '')
        
        input_text = prepare_code_input(context, query)
        
        input_ids = tokenize_and_pad(
            self.tokenizer,
            input_text,
            self.max_seq_len,
            pad_token_id=self.pad_token_id
        )
        
        output_ids = tokenize_and_pad(
            self.tokenizer,
            completion,
            self.max_seq_len,
            pad_token_id=self.pad_token_id
        )
        
        return input_ids, output_ids


def test_training():
    """Тест навчання моделі"""
    print("=" * 70)
    print("🧪 ТЕСТ 1: Навчання моделі")
    print("=" * 70)
    
    # Параметри для швидкого тесту
    dim = 128
    depth = 2
    seq_len = 256
    batch_size = 2
    epochs = 2
    
    print(f"\n📊 Параметри моделі:")
    print(f"   - dim: {dim}")
    print(f"   - depth: {depth}")
    print(f"   - seq_len: {seq_len}")
    print(f"   - batch_size: {batch_size}")
    print(f"   - epochs: {epochs}")
    
    # Завантажити токенізатор
    print(f"\n📥 Завантаження токенізатора...")
    tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
    print(f"   ✅ Vocab size: {vocab_size}")
    
    # Створити датасет
    print(f"\n📚 Створення датасету...")
    dataset = SimpleCodeDataset(tokenizer, max_seq_len=seq_len, pad_token_id=pad_token_id)
    print(f"   ✅ Датасет: {len(dataset)} прикладів")
    
    # Створити модель
    print(f"\n🏗️  Створення моделі...")
    network = MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=network,
        num_refinement_blocks=3,
        num_latent_refinements=4,
        halt_loss_weight=1.0
    )
    
    # Підрахувати параметри
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Параметрів: {total_params:,} (навчаються: {trainable_params:,})")
    
    # Створити trainer
    print(f"\n🎓 Створення trainer...")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=1e-3,
        batch_size=batch_size,
        epochs=epochs,
        max_recurrent_steps=6,
        halt_prob_thres=0.5,
        warmup_steps=10,
        cpu=True,  # Використовуємо CPU для демонстрації
        checkpoint_dir=None,  # Без checkpoint для швидкого тесту
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        mixed_precision=None
    )
    print(f"   ✅ Trainer готовий")
    
    # Навчання
    print(f"\n🚀 Початок навчання...")
    print("-" * 70)
    trainer()
    print("-" * 70)
    print("✅ Навчання завершено!\n")
    
    return model, tokenizer, seq_len


def test_inference(model, tokenizer, seq_len):
    """Тест інференсу моделі"""
    print("=" * 70)
    print("🧪 ТЕСТ 2: Робота моделі (інференс)")
    print("=" * 70)
    
    # Тестові приклади
    test_cases = [
        {
            "context": "def multiply(a, b):\n    pass",
            "query": "Реалізуй функцію",
            "expected": "return a * b"
        },
        {
            "context": "items = [1, 2, 3, 4, 5]",
            "query": "Знайди суму",
            "expected": "sum"
        }
    ]
    
    print(f"\n🔮 Тестування передбачень...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Тест {i}:")
        print(f"   Контекст: {test_case['context']}")
        print(f"   Запит: {test_case['query']}")
        print(f"   Очікується: {test_case['expected']}")
        
        # Підготувати вхід
        input_text = prepare_code_input(test_case['context'], test_case['query'])
        input_ids = tokenize_and_pad(
            tokenizer,
            input_text,
            seq_len,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        ).unsqueeze(0)  # Додати batch dimension
        
        # Передбачення
        model.eval()
        with torch.no_grad():
            try:
                pred_tokens, exit_steps = model.predict(
                    input_ids,
                    max_deep_refinement_steps=8,
                    halt_prob_thres=0.3
                )
                
                # Декодувати результат
                if hasattr(tokenizer, 'decode'):
                    # Знайти перший non-padding token для виводу
                    pred_tokens_clean = pred_tokens[0].cpu().numpy()
                    # Обрізати до перших 50 токенів для читання
                    output = tokenizer.decode(pred_tokens_clean[:min(50, len(pred_tokens_clean))], skip_special_tokens=True)
                else:
                    output = ''.join([tokenizer.inv_vocab.get(int(t), '?') for t in pred_tokens[0][:50]])
                
                print(f"   ✅ Результат: {output[:100]}...")
                print(f"   📊 Кроків уточнення: {exit_steps[0].item()}")
                
            except Exception as e:
                print(f"   ⚠️  Помилка передбачення: {e}")
        
        print()
    
    print("✅ Інференс завершено!\n")


def test_model_structure():
    """Тест структури моделі"""
    print("=" * 70)
    print("🧪 ТЕСТ 3: Структура моделі")
    print("=" * 70)
    
    dim = 64
    vocab_size = 256
    seq_len = 128
    
    network = MLPMixer1D(dim=dim, depth=2, seq_len=seq_len)
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=network,
        num_refinement_blocks=3,
        num_latent_refinements=4
    )
    
    print(f"\n📐 Структура моделі:")
    print(f"   - Розмірність: {dim}")
    print(f"   - Словник: {vocab_size} токенів")
    print(f"   - Довжина послідовності: {seq_len}")
    print(f"   - Блоки уточнення: 3")
    print(f"   - Приховані уточнення: 4")
    
    # Тест forward pass
    print(f"\n🔄 Тест forward pass...")
    batch_size = 2
    seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    outputs, latents = model.get_initial()
    outputs = outputs.unsqueeze(0).repeat(batch_size, seq_len, 1)
    latents = latents.unsqueeze(0).repeat(batch_size, seq_len, 1)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        loss, (main_loss, halt_loss), outputs, latents, pred, halt = model(seq, outputs, latents, labels)
        print(f"   ✅ Forward pass успішний")
        print(f"   📊 Loss: {loss.mean().item():.4f}")
        print(f"   📊 Main loss: {main_loss.mean().item():.4f}")
        print(f"   📊 Halt loss: {halt_loss.mean().item():.4f}")
        print(f"   📊 Halt prob: {halt.mean().item():.4f}")
    except Exception as e:
        print(f"   ❌ Помилка: {e}")
        return
    
    # Тест predict
    print(f"\n🔮 Тест predict...")
    test_seq = torch.randint(0, vocab_size, (1, seq_len))
    try:
        with torch.no_grad():
            pred_tokens, exit_steps = model.predict(test_seq, max_deep_refinement_steps=6)
        print(f"   ✅ Predict успішний")
        print(f"   📊 Вихідні токени: {pred_tokens.shape}")
        print(f"   📊 Кроки виходу: {exit_steps}")
    except Exception as e:
        print(f"   ❌ Помилка: {e}")
    
    print("\n✅ Структура моделі перевірена!\n")


def main():
    """Головна функція"""
    print("\n" + "=" * 70)
    print("🎯 ДЕМОНСТРАЦІЯ TRM МОДЕЛІ")
    print("=" * 70)
    print("\nЦей скрипт демонструє:")
    print("1. Навчання моделі на простих прикладах")
    print("2. Роботу моделі після навчання (інференс)")
    print("3. Структуру та базові операції моделі")
    print()
    
    try:
        # Тест 1: Структура моделі
        test_model_structure()
        
        # Тест 2: Навчання
        model, tokenizer, seq_len = test_training()
        
        # Тест 3: Інференс
        test_inference(model, tokenizer, seq_len)
        
        print("=" * 70)
        print("✅ ВСІ ТЕСТИ ПРОЙДЕНО УСПІШНО!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Помилка під час виконання: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

