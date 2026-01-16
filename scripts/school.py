"""
Скрипт для запуску навчання "Школа" - GPT навчає TRM
"""
import sys
from pathlib import Path

# Додати project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse

from tiny_recursive_model import TinyRecursiveModel, TransformerBackbone
from tiny_recursive_model.utils import load_tokenizer
from school.kindergarten import KindergartenLearning


def main():
    parser = argparse.ArgumentParser(description="Навчання TRM через GPT вчительку")
    parser.add_argument("--days", type=int, default=100, help="Кількість днів навчання")
    parser.add_argument("--teacher-model", type=str, default="gpt2", help="GPT модель для вчительки")
    parser.add_argument("--topics-file", type=str, default=None, help="JSON файл з темами")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'])
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Конфігурація моделі")
    
    args = parser.parse_args()
    
    print("="*60)
    print("[SCHOOL] Система навчання 'Садочок'")
    print("="*60)
    
    # Завантажити конфігурацію
    try:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Помилка завантаження конфігурації: {e}")
        return
    
    # Завантажити tokenizer
    print("\n[LOAD] Завантаження tokenizer...")
    try:
        tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
        print(f"[OK] Tokenizer завантажено (vocab_size={vocab_size})")
    except Exception as e:
        print(f"[ERROR] Помилка завантаження tokenizer: {e}")
        return
    
    # Створити TRM модель (дитина)
    print("\n[BUILD] Створення TRM моделі (дитина)...")
    try:
        model_cfg = cfg.get('model', {})
        
        # Використати Transformer backbone
        use_transformer = model_cfg.get('use_transformer', True)
        if use_transformer:
            network = TransformerBackbone(
                dim=model_cfg.get('dim', 768),
                depth=model_cfg.get('depth', 12),
                seq_len=model_cfg.get('seq_len', 1024),
                pretrained=model_cfg.get('transformer_pretrained', True),
                model_name=model_cfg.get('transformer_model', "gpt2"),
                cache_dir=model_cfg.get('transformer_cache_dir', "models/pretrained")
            )
            dim = network.dim
        else:
            from tiny_recursive_model import MLPMixer1D
            network = MLPMixer1D(
                dim=model_cfg.get('dim', 256),
                depth=model_cfg.get('depth', 4),
                seq_len=model_cfg.get('seq_len', 256)
            )
            dim = model_cfg.get('dim', 256)
        
        student_model = TinyRecursiveModel(
            dim=dim,
            num_tokens=vocab_size,
            network=network,
            num_refinement_blocks=model_cfg.get('num_refinement_blocks', 3),
            num_latent_refinements=model_cfg.get('num_latent_refinements', 6),
            max_recursion_depth=model_cfg.get('max_recursion_depth', 20),
            adaptive_recursion=model_cfg.get('adaptive_recursion', False)
        )
        
        print(f"[OK] TRM модель створена (dim={dim})")
    except Exception as e:
        print(f"[ERROR] Помилка створення моделі: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Створити систему навчання
    print("\n[BUILD] Створення системи навчання...")
    try:
        school = KindergartenLearning(
            student_model=student_model,
            tokenizer=tokenizer,
            teacher_model_name=args.teacher_model,
            curriculum_topics_file=args.topics_file,
            device=args.device,
            save_dir="school_progress"
        )
    except Exception as e:
        print(f"[ERROR] Помилка створення системи навчання: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Запустити навчання
    print("\n[START] Початок навчання...")
    try:
        school.full_education(days=args.days)
        print("\n[OK] Навчання завершено!")
    except KeyboardInterrupt:
        print("\n[WARN] Навчання перервано користувачем")
    except Exception as e:
        print(f"\n[ERROR] Помилка під час навчання: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

