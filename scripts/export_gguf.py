"""
GGUF Export Script
Експорт TRM моделі в GGUF формат для використання з llama.cpp та сумісними інструментами
"""
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from export.gguf_converter import export_trm_to_gguf
from export.quantization import quantize_model
from tiny_recursive_model.utils import load_tokenizer
from inference.model_inference import load_trained_model
from train.constants import DEFAULT_TOKENIZER_NAME


def main():
    """CLI для експорту моделі"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export TRM model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Базовий експорт:
  python scripts/export_gguf.py --model checkpoints/model.pt --output models/exported/model.gguf
  
  # З квантизацією:
  python scripts/export_gguf.py --model checkpoints/model.pt --output models/exported/model.gguf --quantization q4
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for GGUF file"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="trm",
        help="Model name"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["q4", "q5", "q8"],
        default=None,
        help="Quantization type (q4, q5, q8)"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Don't include tokenizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model loading"
    )
    
    args = parser.parse_args()
    
    # Перевірити наявність файлу моделі
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ Модель не знайдено: {model_path}")
        sys.exit(1)
    
    output_path = project_root / args.output
    
    print("=" * 80)
    print("📦 ЕКСПОРТ МОДЕЛІ В GGUF")
    print("=" * 80)
    print(f"📥 Модель: {model_path}")
    print(f"📤 Вихід: {output_path}")
    if args.quantization:
        print(f"🔧 Квантизація: {args.quantization}")
    print()
    
    # Завантажити модель
    print("📥 Завантаження моделі...")
    try:
        tokenizer, _, _ = load_tokenizer(DEFAULT_TOKENIZER_NAME)
        
        inference = load_trained_model(
            model_path=str(model_path),
            device=args.device,
            tokenizer_name=DEFAULT_TOKENIZER_NAME
        )
        
        model = inference.model
        print(f"✅ Модель завантажено")
        
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Завантажити config якщо вказано
    cfg = {}
    if args.config:
        config_path = project_root / args.config
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            print(f"✅ Конфігурація завантажена: {config_path}")
        except Exception as e:
            print(f"⚠️  Помилка завантаження config: {e}")
            print("   Використовуються значення за замовчуванням")
    else:
        # Створити базову конфігурацію з моделі
        cfg = {
            'model': {
                'dim': getattr(model, 'dim', 256),
                'depth': getattr(model, 'depth', 4),
                'vocab_size': getattr(model, 'vocab_size', 50257),
            },
            'training': {},
            'curriculum': {}
        }
    
    # Експортувати в GGUF
    print(f"\n🚀 Експорт в GGUF формат...")
    try:
        gguf_path = export_trm_to_gguf(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            path=output_path
        )
        print(f"✅ GGUF файл збережено: {gguf_path}")
        
    except Exception as e:
        print(f"❌ Помилка експорту: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Квантизація якщо потрібна
    if args.quantization:
        print(f"\n🔧 Квантизація ({args.quantization})...")
        try:
            # Створити тимчасовий PyTorch checkpoint для квантизації
            temp_checkpoint = output_path.with_suffix('.temp.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg
            }, temp_checkpoint)
            
            quantized_path = output_path.with_suffix(f'.{args.quantization}.pt')
            quantize_model(
                model_path=str(temp_checkpoint),
                output_path=str(quantized_path),
                quantization_type=args.quantization
            )
            
            print(f"✅ Квантизована модель збережена: {quantized_path}")
            print(f"⚠️  Примітка: Для експорту квантизованої моделі в GGUF потрібні додаткові інструменти")
            
            # Видалити тимчасовий файл
            temp_checkpoint.unlink()
            
        except Exception as e:
            print(f"⚠️  Помилка квантизації: {e}")
            print("   Базовий GGUF файл все одно збережено")
            import traceback
            traceback.print_exc()
    
    # Експорт tokenizer якщо потрібно
    if not args.no_tokenizer:
        print(f"\n🔤 Експорт tokenizer...")
        try:
            from export.tokenizer_export import export_tokenizer
            tokenizer_path = output_path.with_suffix('.tokenizer.json')
            export_tokenizer(tokenizer, tokenizer_path)
            print(f"✅ Tokenizer збережено: {tokenizer_path}")
        except Exception as e:
            print(f"⚠️  Помилка експорту tokenizer: {e}")
    
    print("\n" + "=" * 80)
    print("✅ ЕКСПОРТ ЗАВЕРШЕНО УСПІШНО!")
    print("=" * 80)
    print(f"📁 GGUF файл: {gguf_path}")
    if not args.no_tokenizer:
        print(f"📁 Tokenizer: {output_path.with_suffix('.tokenizer.json')}")
    if args.quantization:
        print(f"📁 Квантизована модель: {output_path.with_suffix(f'.{args.quantization}.pt')}")
    print()


if __name__ == "__main__":
    main()

