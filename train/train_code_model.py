"""
Скрипт для навчання Tiny Recursive Model на коді (як Cursor)

⚠️ DEPRECATED: Цей файл не використовується runtime.bootstrap
Використовуйте: scripts/train_model.py → runtime.bootstrap
Конфігурація через config/config.yaml
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
from typing import Optional

from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.utils import tokenize_and_pad, prepare_code_input, load_tokenizer
from train.constants import (
    DEFAULT_DIM, DEFAULT_DEPTH, DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MAX_RECURRENT_STEPS,
    DEFAULT_NUM_REFINEMENT_BLOCKS, DEFAULT_NUM_LATENT_REFINEMENTS,
    DEFAULT_HALT_LOSS_WEIGHT, DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_TOKENIZER_NAME, CACHE_SIZE_LIMIT, DEFAULT_CACHE_TOKENS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS, DEFAULT_HALT_PROB_THRES
)
from train.exceptions import DatasetNotFoundError
from train.model_factory import create_model


class CodeDataset(Dataset):
    """
    Dataset для навчання на коді.
    Очікує JSON файл з парами "context" -> "completion"
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
        cache_tokens: bool = DEFAULT_CACHE_TOKENS  # Кешувати токенізовані дані для прискорення
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.cache_tokens = cache_tokens
        
        # Завантажити дані
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Підтримка метаданих: якщо словник з 'data', витягнути список
        if isinstance(data, dict) and 'data' in data:
            self.data = data['data']
        elif isinstance(data, list):
            self.data = data
        else:
            self.data = []
        
        print(f"Завантажено {len(self.data)} прикладів")
        
        # Кеш для токенізованих даних (за потреби)
        self._cached_tokens = {} if cache_tokens else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Перевірити кеш
        if self._cached_tokens is not None and idx in self._cached_tokens:
            return self._cached_tokens[idx]
        
        item = self.data[idx]
        
        # Об'єднати контекст і запит
        context = item.get('context', '')
        query = item.get('query', '')
        completion = item.get('completion', '')
        
        # Створити вхід: контекст + запит
        input_text = prepare_code_input(context, query)
        
        # Токенізувати
        input_ids = self.tokenizer.encode(
            input_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)
        
        output_ids = self.tokenizer.encode(
            completion,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)
        
        # Оптимізований padding: використовувати F.pad для швидшості
        input_pad_len = self.max_seq_len - input_ids.shape[0]
        if input_pad_len > 0:
            input_ids = torch.nn.functional.pad(
                input_ids, 
                (0, input_pad_len), 
                value=self.pad_token_id
            )
        else:
            input_ids = input_ids[:self.max_seq_len]
        
        output_pad_len = self.max_seq_len - output_ids.shape[0]
        if output_pad_len > 0:
            output_ids = torch.nn.functional.pad(
                output_ids, 
                (0, output_pad_len), 
                value=self.pad_token_id
            )
        else:
            output_ids = output_ids[:self.max_seq_len]
        
        result = (input_ids, output_ids)
        
        # Зберегти в кеш (обмежити розмір кешу)
        if self._cached_tokens is not None and len(self._cached_tokens) < CACHE_SIZE_LIMIT:
            self._cached_tokens[idx] = result
        
        return result


def create_sample_dataset(output_path: str = "sample_code_data.json"):
    """
    Створює приклад датасету для тестування
    """
    sample_data = [
        {
            "context": "def add(a, b):\n    return a + b",
            "query": "Додай docstring",
            "completion": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b"
        },
        {
            "context": "class Calculator:\n    def __init__(self):\n        self.value = 0",
            "query": "Додай метод multiply",
            "completion": "class Calculator:\n    def __init__(self):\n        self.value = 0\n    \n    def multiply(self, x):\n        self.value *= x\n        return self.value"
        },
        {
            "context": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result",
            "query": "Зроби через list comprehension",
            "completion": "def process_data(data):\n    return [item * 2 for item in data]"
        },
        {
            "context": "import os\n\ndef read_file(path):\n    with open(path) as f:\n        return f.read()",
            "query": "Додай обробку помилок",
            "completion": "import os\n\ndef read_file(path):\n    try:\n        with open(path) as f:\n            return f.read()\n    except FileNotFoundError:\n        return None\n    except Exception as e:\n        print(f\"Error: {e}\")\n        return None"
        },
        {
            "context": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "query": "Додай мемоізацію",
            "completion": "from functools import lru_cache\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nfibonacci = lru_cache(maxsize=None)(fibonacci)"
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Створено приклад датасету: {output_path}")
    return output_path


def train(
    data_path: str = "sample_code_data.json",
    model_save_path: str = "trained_code_model.pt",
    vocab_size: Optional[int] = None,  # Буде визначено з tokenizer
    dim: int = DEFAULT_DIM,
    depth: int = DEFAULT_DEPTH,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_recurrent_steps: int = DEFAULT_MAX_RECURRENT_STEPS,
    use_gpu: bool = True,
    resume_from_checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    grad_accum: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    num_workers: int | None = None,
    mixed_precision: str | None = None
):
    """
    Основна функція навчання
    """
    print("=" * 60)
    print("Навчання Tiny Recursive Model для коду")
    print("=" * 60)
    
    # Перевірити наявність даних
    if not os.path.exists(data_path):
        raise DatasetNotFoundError(data_path)
    
    # Ініціалізувати токенізатор (використовуємо спільну утиліту)
    print("\n1. Завантаження токенізатора...")
    tokenizer, actual_vocab_size, pad_token_id = load_tokenizer(DEFAULT_TOKENIZER_NAME)
    # Використати vocab_size з tokenizer якщо не вказано
    if vocab_size is None:
        vocab_size = actual_vocab_size
    print(f"   Використовується tokenizer (vocab_size={vocab_size})")
    
    # Створити dataset
    print("\n2. Створення dataset...")
    dataset = CodeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=seq_len,
        pad_token_id=pad_token_id
    )
    
    # Створити модель
    print("\n3. Ініціалізація моделі...")
    model = create_model(
        dim=dim,
        vocab_size=vocab_size,
        depth=depth,
        seq_len=seq_len,
        num_refinement_blocks=DEFAULT_NUM_REFINEMENT_BLOCKS,
        num_latent_refinements=DEFAULT_NUM_LATENT_REFINEMENTS,
        halt_loss_weight=DEFAULT_HALT_LOSS_WEIGHT
    )
    
    print(f"   Параметри моделі:")
    print(f"   - dim: {dim}")
    print(f"   - vocab_size: {vocab_size}")
    print(f"   - seq_len: {seq_len}")
    print(f"   - depth: {depth}")
    
    # Створити trainer з оптимізованими параметрами
    print("\n4. Створення trainer...")
    from train.trainer_factory import create_trainer
    
    trainer = create_trainer(
        model=model,
        dataset=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_recurrent_steps=max_recurrent_steps,
        use_gpu=use_gpu,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        gradient_accumulation_steps=grad_accum,
        num_workers=num_workers,
        mixed_precision=mixed_precision
    )
    
    # Навчання
    print("\n5. Початок навчання...")
    if resume_from_checkpoint:
        print(f"   Продовження з checkpoint: {resume_from_checkpoint}")
    print("-" * 60)
    trainer(resume_from_checkpoint=resume_from_checkpoint)
    
    # Зберегти модель
    print("\n6. Збереження моделі...")
    if trainer.accelerator.is_main_process:
        model_save_path_obj = Path(model_save_path)
        model_save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model.state_dict(), model_save_path_obj)
            print(f"   Модель збережено: {model_save_path_obj}")
        except Exception as e:
            print(f"   ❌ Помилка збереження моделі: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("Навчання завершено!")
    print("=" * 60)
    
    return model, tokenizer


def test_model(model, tokenizer, test_input: str = "def hello():\n    print('world')", seq_len: int = 512):
    """
    Тестування навченої моделі
    """
    print("\n" + "=" * 60)
    print("Тестування моделі")
    print("=" * 60)
    
    # Токенізувати вхід
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(test_input, max_length=seq_len, truncation=True, return_tensors='pt')
        # Додати padding якщо потрібно
        if input_ids.shape[1] < seq_len:
            pad_len = seq_len - input_ids.shape[1]
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padding = torch.full((1, pad_len), pad_token, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
        input_ids = input_ids[:, :seq_len]  # Обрізати якщо занадто довгий
    else:
        input_ids = torch.tensor([[tokenizer.vocab.get(c, 0) for c in test_input[:seq_len]]], dtype=torch.long)
        # Додати padding
        if input_ids.shape[1] < seq_len:
            pad_len = seq_len - input_ids.shape[1]
            padding = torch.full((1, pad_len), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
    
    # Передбачення
    print(f"\nВхід: {test_input}")
    print("\nГенерація відповіді...")
    
    try:
        with torch.no_grad():
            pred_tokens, exit_steps = model.predict(
                input_ids,
                max_deep_refinement_steps=DEFAULT_MAX_RECURRENT_STEPS,
                halt_prob_thres=DEFAULT_HALT_PROB_THRES
            )
        
        # Декодувати
        if hasattr(tokenizer, 'decode'):
            # Прибрати padding tokens перед декодуванням
            pred_tokens_clean = pred_tokens[0].cpu().numpy()
            # Знайти перший padding token або обрізати до розумної довжини
            output = tokenizer.decode(pred_tokens_clean[:min(100, len(pred_tokens_clean))])
        else:
            output = ''.join([tokenizer.inv_vocab.get(int(t), '?') for t in pred_tokens[0][:100]])
        
        print(f"\nВихід: {output}")
        print(f"Кроки уточнення: {exit_steps[0].item()}")
    except Exception as e:
        print(f"\nПомилка при тестуванні: {e}")
        print("Модель навчена, але потребує додаткового налаштування для інференсу")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Навчання TRM на коді")
    parser.add_argument("--data", type=str, default="sample_code_data.json",
                       help="Шлях до JSON файлу з даними")
    parser.add_argument("--save", type=str, default="trained_code_model.pt",
                       help="Шлях для збереження моделі")
    parser.add_argument("--dim", type=int, default=512,
                       help="Розмірність ембеддингів")
    parser.add_argument("--depth", type=int, default=4,
                       help="Глибина MLP Mixer")
    parser.add_argument("--seq-len", type=int, default=2048,
                       help="Максимальна довжина послідовності")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Розмір батчу")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Кількість епох")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--cpu", action="store_true",
                       help="Використовувати CPU замість GPU")
    parser.add_argument("--create-sample", action="store_true",
                       help="Створити приклад датасету")
    parser.add_argument("--resume", type=str, default=None,
                       help="Шлях до checkpoint для продовження навчання (наприклад: checkpoints/checkpoint_latest.pt)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Папка для збереження checkpoint'ів (за замовчуванням: checkpoints)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                       help="Зберігати checkpoint кожні N батчів (за замовчуванням: 100)")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Кількість кроків для gradient accumulation (за замовчуванням: 4)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Кількість worker'ів для DataLoader (None = автоматично)")
    parser.add_argument("--mixed-precision", type=str, choices=['fp16', 'bf16'], default=None, nargs='?',
                       help="Mixed precision training (fp16, bf16, або None)")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.data)
        exit(0)
    
    # Навчання
    model, tokenizer = train(
        data_path=args.data,
        model_save_path=args.save,
        dim=args.dim,
        depth=args.depth,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_gpu=not args.cpu,
        resume_from_checkpoint=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision
    )
    
    # Тестування
    test_model(model, tokenizer, seq_len=args.seq_len)

