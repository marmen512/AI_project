#!/usr/bin/env python3
"""
Простий тест моделі з чекпоінту
"""
import sys
import torch
from pathlib import Path

# Додати шлях до проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tiny_recursive_model import TinyRecursiveModel, TransformerBackbone
from tiny_recursive_model.utils import load_tokenizer
from train.constants import DEFAULT_TOKENIZER_NAME

def test_model_from_checkpoint():
    """Тест моделі з чекпоінту"""
    print("=" * 60)
    print("🧪 ТЕСТ МОДЕЛІ З ЧЕКПОІНТУ")
    print("=" * 60)
    
    # Завантажити токенізатор
    print("📥 Завантаження токенізатора...")
    tokenizer, _, _ = load_tokenizer(DEFAULT_TOKENIZER_NAME)
    vocab_size = len(tokenizer)
    print(f"✅ Токенізатор завантажено (vocab_size: {vocab_size})")
    
    # Параметри моделі (з конфігурації)
    dim = 768
    depth = 12
    seq_len = 1024
    
    print(f"\n🔧 Створення моделі...")
    print(f"   dim: {dim}")
    print(f"   depth: {depth}")
    print(f"   seq_len: {seq_len}")
    print(f"   vocab_size: {vocab_size}")
    
    # Створити backbone
    backbone = TransformerBackbone(
        dim=dim,
        depth=depth,
        seq_len=seq_len,
        pretrained=True,
        model_name='gpt2'
    )
    
    # Створити модель
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=backbone,
        max_recursion_depth=20,
        adaptive_recursion=True
    )
    
    # Завантажити чекпоінт
    checkpoint_path = "checkpoints/best_loss.ckpt"
    print(f"\n📥 Завантаження чекпоінту: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Перевірити структуру чекпоінту
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"   Знайдено model_state_dict з {len(model_state)} параметрами")
        else:
            model_state = checkpoint
            print(f"   Використовуємо весь чекпоінт як model_state_dict")
        
        model.load_state_dict(model_state)
        model.eval()
        print(f"✅ Модель завантажена з чекпоінту")
        
        # Простий тест генерації
        print(f"\n🧪 Тест генерації...")
        test_inputs = [
            "Привіт! Як справи?",
            "Що таке штучний інтелект?",
            "Напиши простий код на Python:",
            "Поясни що таке рекурсія:"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n--- Тест {i} ---")
            print(f"Вхід: {test_input}")
            
            try:
                # Токенізувати вхід
                inputs = tokenizer.encode(test_input, return_tensors='pt')
                
                # Використати predict метод TRM
                with torch.no_grad():
                    predictions, steps = model.predict(
                        inputs,
                        halt_prob_thres=0.5,
                        max_deep_refinement_steps=12
                    )
                
                # Декодувати результат
                generated_text = tokenizer.decode(predictions, skip_special_tokens=True)
                response = generated_text.strip()
                
                print(f"Відповідь: {response}")
                
                # Перевірка валідності
                if len(response) > 0:
                    print(f"✅ Генерація успішна")
                else:
                    print(f"⚠️ Порожня відповідь")
                    
            except Exception as e:
                print(f"❌ Помилка генерації: {e}")
        
        print(f"\n✅ Тестування завершено!")
        
    except Exception as e:
        print(f"❌ Помилка завантаження чекпоінту: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_from_checkpoint()
