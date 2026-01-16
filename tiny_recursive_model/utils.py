"""
Спільні утиліти для обробки даних та токенізації
"""
import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional


def pad_sequence(
    sequence: torch.Tensor,
    max_len: int,
    pad_value: int = 0,
    pad_left: bool = False
) -> torch.Tensor:
    """
    Оптимізоване padding послідовності
    
    Args:
        sequence: Тензор з послідовністю токенів
        max_len: Максимальна довжина
        pad_value: Значення для padding
        pad_left: Якщо True, padding зліва, інакше справа
        
    Returns:
        Западедена послідовність
    """
    seq_len = sequence.shape[0]
    
    if seq_len >= max_len:
        return sequence[:max_len]
    
    pad_len = max_len - seq_len
    
    if pad_left:
        return F.pad(sequence, (pad_len, 0), value=pad_value)
    else:
        return F.pad(sequence, (0, pad_len), value=pad_value)


def tokenize_and_pad(
    tokenizer,
    text: str,
    max_seq_len: int,
    pad_token_id: Optional[int] = None,
    truncation: bool = True
) -> torch.Tensor:
    """
    Токенізувати текст та додати padding
    
    Args:
        tokenizer: Токенізатор
        text: Текст для токенізації
        max_seq_len: Максимальна довжина послідовності
        pad_token_id: ID токену для padding (None = автоматично)
        truncation: Обрізати якщо довше max_seq_len
        
    Returns:
        Тензор з токенами
    """
    # Визначити pad_token_id
    if pad_token_id is None:
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 0
    
    # Токенізувати
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(
            text,
            max_length=max_seq_len if truncation else None,
            truncation=truncation,
            return_tensors='pt'
        ).squeeze(0)
    else:
        # Простий символьний tokenizer
        input_ids = torch.tensor([tokenizer.vocab.get(c, 0) for c in text[:max_seq_len if truncation else len(text)]], dtype=torch.long)
    
    # Padding
    return pad_sequence(input_ids, max_seq_len, pad_value=pad_token_id)


def prepare_code_input(
    context: str,
    query: str,
    query_marker: str = "\n\n[QUERY]\n"
) -> str:
    """
    Підготувати вхідний текст з контексту та запиту
    
    Args:
        context: Контекст коду
        query: Запит користувача
        query_marker: Маркер для розділення контексту та запиту
        
    Returns:
        Об'єднаний текст
    """
    return f"{context}{query_marker}{query}"


def load_tokenizer(model_name: str = "gpt2") -> tuple:
    """
    Завантажити токенізатор з обробкою помилок
    
    Args:
        model_name: Назва моделі для токенізатора
        
    Returns:
        tuple: (tokenizer, vocab_size, pad_token_id)
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id
        return tokenizer, vocab_size, pad_token_id
    except Exception:
        # Fallback на простий символьний tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {chr(i): i for i in range(256)}
                self.vocab['<PAD>'] = 0
                self.vocab['<EOS>'] = 256
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
                self.pad_token_id = 0
            
            def encode(self, text, max_length=None, truncation=False, return_tensors=None):
                ids = [self.vocab.get(c, 0) for c in text[:max_length or len(text)]]
                if return_tensors == 'pt':
                    return torch.tensor([ids], dtype=torch.long)
                return ids
            
            def __len__(self):
                return len(self.vocab)
            
            def decode(self, token_ids):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.cpu().numpy()
                return ''.join([self.inv_vocab.get(int(t), '?') for t in token_ids])
        
        tokenizer = SimpleTokenizer()
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id
        return tokenizer, vocab_size, pad_token_id





















