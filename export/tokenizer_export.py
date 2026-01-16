"""
Експорт tokenizer в JSON формат сумісний з GGUF
"""
import json
from pathlib import Path
from typing import Any, Dict


def export_tokenizer(tokenizer: Any, path: str | Path) -> Path:
    """
    Експортувати tokenizer в JSON формат сумісний з GGUF
    
    Args:
        tokenizer: Tokenizer instance (HuggingFace tokenizer)
        path: Шлях для збереження JSON файлу
    
    Returns:
        Шлях до збереженого файлу
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Визначити тип tokenizer
    tokenizer_type = "gpt2"  # За замовчуванням
    if hasattr(tokenizer, 'name_or_path'):
        if 'gpt2' in tokenizer.name_or_path.lower():
            tokenizer_type = "gpt2"
        elif 'bert' in tokenizer.name_or_path.lower():
            tokenizer_type = "bert"
        elif 'roberta' in tokenizer.name_or_path.lower():
            tokenizer_type = "roberta"
    
    # Отримати vocab
    vocab = {}
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'get_vocab'):
        vocab = tokenizer.tokenizer.get_vocab()
    
    # Отримати спеціальні токени
    unk_token = getattr(tokenizer, 'unk_token', None)
    bos_token = getattr(tokenizer, 'bos_token', None)
    eos_token = getattr(tokenizer, 'eos_token', None)
    pad_token = getattr(tokenizer, 'pad_token', None)
    
    # Якщо токени не знайдені, спробувати через tokenizer.tokenizer
    if not unk_token and hasattr(tokenizer, 'tokenizer'):
        unk_token = getattr(tokenizer.tokenizer, 'unk_token', None)
    if not bos_token and hasattr(tokenizer, 'tokenizer'):
        bos_token = getattr(tokenizer.tokenizer, 'bos_token', None)
    if not eos_token and hasattr(tokenizer, 'tokenizer'):
        eos_token = getattr(tokenizer.tokenizer, 'eos_token', None)
    if not pad_token and hasattr(tokenizer, 'tokenizer'):
        pad_token = getattr(tokenizer.tokenizer, 'pad_token', None)
    
    # Створити дані для експорту
    data = {
        "type": tokenizer_type,
        "vocab": vocab,
        "unk_token": unk_token,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
    }
    
    # Додати додаткову інформацію якщо доступна
    if hasattr(tokenizer, 'vocab_size'):
        data["vocab_size"] = tokenizer.vocab_size
    if hasattr(tokenizer, 'model_max_length'):
        data["model_max_length"] = tokenizer.model_max_length
    
    # Зберегти в JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer збережено: {path}")
    return path

