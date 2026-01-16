"""
Навчання TRM з використанням Phi-3.5 як teacher model
Використовує knowledge distillation для переносу знань

⚠️ DEPRECATED: Цей файл не використовується runtime.bootstrap
Використовуйте: scripts/train_model.py → runtime.bootstrap
Конфігурація через config/config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False
    print("⚠️ llama-cpp-python не встановлено. Встановіть: pip install llama-cpp-python")

from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.utils import tokenize_and_pad, prepare_code_input, load_tokenizer


class Phi3Teacher:
    """Teacher model на основі Phi-3.5"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048):
        if not HAS_LLAMA:
            raise ImportError("llama-cpp-python не встановлено")
        
        print(f"Завантаження Phi-3.5 teacher model...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            verbose=False
        )
        print("✅ Phi-3.5 teacher завантажена")
    
    def generate_soft_labels(self, context: str, query: str, num_samples: int = 5) -> List[str]:
        """
        Генерація кількох варіантів відповіді для soft labels
        
        Args:
            context: Контекст коду
            query: Запит
            num_samples: Скільки варіантів згенерувати
            
        Returns:
            Список відповідей
        """
        prompt = f"""{context}

[QUERY]
{query}

[RESPONSE]
"""
        
        responses = []
        for _ in range(num_samples):
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9,
                stop=["[QUERY]", "\n\n\n"],
                echo=False
            )
            responses.append(response['choices'][0]['text'].strip())
        
        return responses


class DistillationDataset(Dataset):
    """Dataset для knowledge distillation"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        teacher: Phi3Teacher = None,
        max_seq_len: int = 2048,
        use_teacher: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.teacher = teacher
        self.use_teacher = use_teacher and teacher is not None
        
        # Завантажити дані
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Завантажено {len(self.data)} прикладів")
        if self.use_teacher:
            print("✅ Використовується Phi-3.5 teacher для distillation")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        context = item.get('context', '')
        query = item.get('query', '')
        completion = item.get('completion', '')
        
        # Якщо використовуємо teacher, генерувати soft labels
        if self.use_teacher and idx < len(self.data) // 2:  # Тільки для половини даних
            try:
                teacher_responses = self.teacher.generate_soft_labels(context, query, num_samples=3)
                # Використати найкращу відповідь від teacher
                completion = teacher_responses[0] if teacher_responses else completion
            except:
                pass  # Якщо помилка, використати оригінальну completion
        
        # Створити вхід (використовуємо спільну утиліту)
        input_text = prepare_code_input(context, query)
        
        # Визначити pad_token_id
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else 0
        
        # Токенізувати та додати padding (використовуємо спільну утиліту)
        input_ids = tokenize_and_pad(
            self.tokenizer,
            input_text,
            self.max_seq_len,
            pad_token_id=pad_token_id
        )
        
        output_ids = tokenize_and_pad(
            self.tokenizer,
            completion,
            self.max_seq_len,
            pad_token_id=pad_token_id
        )
        
        return input_ids, output_ids


def train_with_phi3(
    phi3_model_path: str,
    data_path: str,
    model_save_path: str = "trm_phi3_enhanced.pt",
    dim: int = 512,
    depth: int = 4,
    seq_len: int = 2048,
    batch_size: int = 4,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    use_distillation: bool = True,
    cpu: bool = False
):
    """
    Навчання TRM з використанням Phi-3.5
    
    Args:
        phi3_model_path: Шлях до Phi-3.5 GGUF
        data_path: Шлях до датасету
        model_save_path: Де зберегти модель
        dim: Розмірність моделі
        depth: Глибина MLP Mixer
        seq_len: Довжина послідовності
        batch_size: Розмір батчу
        epochs: Кількість епох
        learning_rate: Learning rate
        use_distillation: Використовувати knowledge distillation
        cpu: Використовувати CPU
    """
    print("=" * 60)
    print("Навчання TRM з Phi-3.5 teacher")
    print("=" * 60)
    
    # Завантажити токенізатор (використовуємо спільну утиліту)
    print("\n1. Завантаження токенізатора...")
    try:
        tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
        print(f"✅ Tokenizer завантажено (vocab_size={vocab_size})")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return None, None
    
    # Ініціалізувати teacher
    teacher = None
    if use_distillation and HAS_LLAMA:
        try:
            teacher = Phi3Teacher(phi3_model_path)
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити Phi-3.5: {e}")
            print("   Продовжую без distillation")
            use_distillation = False
    
    # Створити dataset
    print("\n2. Створення dataset...")
    dataset = DistillationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        teacher=teacher,
        max_seq_len=seq_len,
        use_teacher=use_distillation
    )
    
    # Створити модель
    print("\n3. Ініціалізація TRM...")
    network = MLPMixer1D(
        dim=dim,
        depth=depth,
        seq_len=seq_len
    )
    
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=vocab_size,
        network=network,
        num_refinement_blocks=3,
        num_latent_refinements=6,
        halt_loss_weight=1.0
    )
    
    print(f"   Параметри: dim={dim}, depth={depth}, seq_len={seq_len}")
    
    # Створити trainer
    print("\n4. Створення trainer...")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_recurrent_steps=12,
        halt_prob_thres=0.5,
        warmup_steps=1000,
        cpu=cpu
    )
    
    # Навчання
    print("\n5. Початок навчання...")
    print("-" * 60)
    trainer()
    
    # Зберегти
    print("\n6. Збереження моделі...")
    if trainer.accelerator.is_main_process:
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Модель збережено: {model_save_path}")
    
    print("\n" + "=" * 60)
    print("Навчання завершено!")
    print("=" * 60)
    
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Навчання TRM з Phi-3.5")
    parser.add_argument("--phi3-model", type=str, default="../models/phi-3.5-mini-instruct-q4_k_m.gguf",
                       help="Шлях до Phi-3.5 GGUF (за замовчуванням: ../models/phi-3.5-mini-instruct-q4_k_m.gguf)")
    parser.add_argument("--data", type=str, default="phi3_enhanced_dataset.json",
                       help="Шлях до датасету")
    parser.add_argument("--save", type=str, default="trm_phi3_enhanced.pt",
                       help="Шлях для збереження")
    parser.add_argument("--dim", type=int, default=512,
                       help="Розмірність")
    parser.add_argument("--depth", type=int, default=4,
                       help="Глибина")
    parser.add_argument("--seq-len", type=int, default=2048,
                       help="Довжина послідовності")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Епохи")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--no-distillation", action="store_true",
                       help="Не використовувати distillation")
    parser.add_argument("--cpu", action="store_true",
                       help="Використовувати CPU")
    
    args = parser.parse_args()
    
    train_with_phi3(
        phi3_model_path=args.phi3_model,
        data_path=args.data,
        model_save_path=args.save,
        dim=args.dim,
        depth=args.depth,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_distillation=not args.no_distillation,
        cpu=args.cpu
    )

