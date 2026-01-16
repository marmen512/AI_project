#!/usr/bin/env python3
"""
Простий тест для перевірки чекпоінту
"""
import sys
import torch
from pathlib import Path
import json

# Додати шлях до проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tiny_recursive_model import TinyRecursiveModel, TransformerBackbone
from tiny_recursive_model.utils import load_tokenizer
from train.constants import DEFAULT_TOKENIZER_NAME

def test_checkpoint(checkpoint_path: str):
    """Тестування чекпоінту"""
    print("=" * 60)
    print("🧪 ТЕСТУВАННЯ ЧЕКПОІНТУ")
    print("=" * 60)
    
    # Завантажити чекпоінт
    print(f"📥 Завантаження чекпоінту: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Чекпоінт завантажено")
        
        # Показати інформацію про чекпоінт
        print(f"\n📊 Інформація про чекпоінт:")
        if 'epoch' in checkpoint:
            print(f"   Епоха: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"   Крок: {checkpoint['step']}")
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']:.4f}")
        if 'best_loss' in checkpoint:
            print(f"   Найкращий loss: {checkpoint['best_loss']:.4f}")
        
        # Перевірити структуру моделі
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n🏗️ Структура моделі:")
            for key in list(model_state.keys())[:10]:  # Показати перші 10 ключів
                print(f"   {key}: {model_state[key].shape}")
            if len(model_state.keys()) > 10:
                print(f"   ... та ще {len(model_state.keys()) - 10} параметрів")
        
        # Спробувати створити модель з конфігурації
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n⚙️ Конфігурація моделі:")
            print(f"   dim: {config.get('dim', 'N/A')}")
            print(f"   depth: {config.get('depth', 'N/A')}")
            print(f"   seq_len: {config.get('seq_len', 'N/A')}")
            print(f"   vocab_size: {config.get('vocab_size', 'N/A')}")
            
            # Спробувати створити модель
            try:
                print(f"\n🔧 Створення моделі...")
                
                # Завантажити токенізатор
                tokenizer, _, _ = load_tokenizer(DEFAULT_TOKENIZER_NAME)
                vocab_size = len(tokenizer)
                
                # Створити backbone
                if config.get('use_transformer', False):
                    backbone = TransformerBackbone(
                        dim=config['dim'],
                        depth=config['depth'],
                        seq_len=config['seq_len'],
                        vocab_size=vocab_size,
                        transformer_model=config.get('transformer_model', 'gpt2')
                    )
                else:
                    from tiny_recursive_model import MLPMixer1D
                    backbone = MLPMixer1D(
                        dim=config['dim'],
                        depth=config['depth'],
                        seq_len=config['seq_len']
                    )
                
                # Створити модель
                model = TinyRecursiveModel(
                    dim=config['dim'],
                    num_tokens=vocab_size,
                    network=backbone,
                    max_recursion_depth=config.get('max_recursion_depth', 20),
                    adaptive_recursion=config.get('adaptive_recursion', True)
                )
                
                # Завантажити стан моделі
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                print(f"✅ Модель створена та завантажена")
                
                # Простий тест генерації
                print(f"\n🧪 Тест генерації...")
                test_input = "Привіт! Як справи?"
                
                # Токенізувати вхід
                inputs = tokenizer.encode(test_input, return_tensors='pt')
                
                # Генерація
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Декодувати результат
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   Вхід: {test_input}")
                print(f"   Результат: {generated_text}")
                
                print(f"\n✅ Тест пройшов успішно!")
                
            except Exception as e:
                print(f"❌ Помилка створення моделі: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Помилка завантаження чекпоінту: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Тестувати найкращий чекпоінт
    checkpoint_path = "checkpoints/best_loss.ckpt"
    if Path(checkpoint_path).exists():
        test_checkpoint(checkpoint_path)
    else:
        print(f"❌ Чекпоінт не знайдено: {checkpoint_path}")
        print("Доступні чекпоінти:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob("*.ckpt"):
                print(f"   {ckpt.name}")
