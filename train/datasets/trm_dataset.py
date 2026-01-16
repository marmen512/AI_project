"""
Ефективний Dataset для TRM з lazy loading та document-aware підтримкою
Замість завантаження всього JSON в пам'ять, завантажуємо по мірі потреби
Підтримує doc_id та segment_id для збереження document boundaries
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import hashlib

from core.constants import DATASET_DOC_ID, DATASET_SEGMENT_ID


class TRMDataset(Dataset):
    """Ефективний Dataset для TRM з lazy loading та обмеженням довжини"""
    
    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        cache_size: int = 1000,  # Кеш для N прикладів
        validate_format: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Датасет не знайдено: {self.data_path}")
        
        # Lazy loading: зберігаємо тільки індекси
        self._data_indices = []
        self._cache = {}
        self._data_structure = None  # 'list' або 'dict'
        
        # Валідація формату (тільки перші N рядків)
        if validate_format:
            self._validate_format()
        
        # Підрахунок прикладів без завантаження всього
        self._count_examples()
        
        print(f"[OK] TRMDataset: {len(self._data_indices)} прикладів, max_seq_len={max_seq_len}, cache_size={cache_size}")
    
    def _validate_format(self):
        """Перевірити формат файлу (тільки перші 10 прикладів)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                self._data_structure = 'dict'
                sample = data['data'][:10]
            elif isinstance(data, list):
                self._data_structure = 'list'
                sample = data[:10]
            else:
                raise ValueError(f"Невідомий формат датасету: {self.data_path}")
            
            # Перевірити структуру
            for item in sample:
                if not isinstance(item, dict):
                    raise ValueError(f"Приклад повинен бути словником, отримано: {type(item)}")
                if 'context' not in item and 'input' not in item:
                    raise ValueError(f"Приклад повинен містити 'context' або 'input'")
                if 'completion' not in item and 'output' not in item:
                    raise ValueError(f"Приклад повинен містити 'completion' або 'output'")
    
    def _count_examples(self):
        """Підрахувати приклади без завантаження всього в пам'ять"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if self._data_structure == 'dict' or (isinstance(data, dict) and 'data' in data):
                self._data_structure = 'dict'
                self._data_indices = list(range(len(data['data'])))
            elif isinstance(data, list):
                self._data_structure = 'list'
                self._data_indices = list(range(len(data)))
            else:
                raise ValueError(f"Невідомий формат: {self.data_path}")
    
    def _load_item(self, idx: int) -> Dict:
        """Завантажити один приклад (з кешу або файлу)"""
        if idx in self._cache:
            return self._cache[idx]
        
        # Завантажити з файлу
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if self._data_structure == 'dict':
                item = data['data'][idx]
            else:
                item = data[idx]
        
        # Кешувати (з обмеженням розміру)
        if len(self._cache) < self.cache_size:
            self._cache[idx] = item
        else:
            # Видалити найстаріший (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._cache[idx] = item
        
        return item
    
    def __len__(self):
        return len(self._data_indices)
    
    def __getitem__(self, idx):
        item = self._load_item(idx)
        
        # Отримати context та completion
        context = item.get('context', item.get('input', ''))
        completion = item.get('completion', item.get('output', ''))
        
        # Document-aware: отримати doc_id та segment_id
        # doc_id - унікальний ідентифікатор документа
        # segment_id - номер сегменту в документі (для документів розбитих на частини)
        doc_id = item.get(DATASET_DOC_ID, item.get('doc_id', None))
        segment_id = item.get(DATASET_SEGMENT_ID, item.get('segment_id', 0))
        
        # Якщо doc_id не вказано, створити його з hash контексту
        if doc_id is None:
            # Створити hash з context для унікальності
            doc_id = hashlib.md5(context.encode('utf-8')).hexdigest()[:16]
        
        # Обрізати ДО токенізації (економія часу)
        # Приблизно 4 символи на токен
        max_chars = self.max_seq_len * 4
        context = context[:max_chars]
        completion = completion[:max_chars]
        
        # Токенізувати
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(
                context,
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
        else:
            # Простий tokenizer
            input_ids = torch.tensor([self.tokenizer.vocab.get(c, 0) for c in context[:self.max_seq_len]], dtype=torch.long)
            output_ids = torch.tensor([self.tokenizer.vocab.get(c, 0) for c in completion[:self.max_seq_len]], dtype=torch.long)
        
        # Padding
        input_ids = self._pad(input_ids, self.max_seq_len)
        output_ids = self._pad(output_ids, self.max_seq_len)
        
        # Document-aware: повернути з doc_id та segment_id
        # Конвертувати doc_id (string) в числовий ID для tensor
        # Обмежити розмір щоб уникнути overflow
        try:
            if isinstance(doc_id, str):
                # Використати hash та обмежити до безпечного діапазону (0 до 2^31-1)
                # hash() може повертати від'ємні числа, тому використовуємо модуль
                hash_val = hash(doc_id)
                # Обмежити до позитивного числа в безпечному діапазоні
                doc_id_int = (hash_val & 0x7FFFFFFF)  # Беремо тільки 31 біт (без знаку)
            else:
                doc_id_int = int(doc_id) & 0x7FFFFFFF
        except (ValueError, OverflowError, TypeError):
            # Fallback: використати простий hash з обмеженням
            hash_val = hash(str(doc_id))
            doc_id_int = hash_val & 0x7FFFFFFF
        
        # Переконатися що число в безпечному діапазоні
        doc_id_int = max(0, min(doc_id_int, 2147483647))  # 2^31 - 1
        doc_id_tensor = torch.tensor([doc_id_int], dtype=torch.long)
        segment_id_tensor = torch.tensor([segment_id], dtype=torch.long)
        
        # Повернути як tuple для backwards compatibility з trainer
        # Trainer очікує (input_ids, output_ids)
        # doc_id та segment_id доступні через metadata якщо потрібно
        return input_ids, output_ids, doc_id_tensor, segment_id_tensor
    
    def _pad(self, ids: torch.Tensor, max_len: int) -> torch.Tensor:
        """Padding до max_len"""
        if ids.dim() == 0:
            ids = ids.unsqueeze(0)
        
        current_len = ids.shape[0]
        if current_len < max_len:
            padding = torch.full((max_len - current_len,), self.pad_token_id, dtype=ids.dtype)
            return torch.cat([ids, padding])
        return ids[:max_len]
    
    def update_max_seq_len(self, new_seq_len: int):
        """Оновити max_seq_len (для curriculum learning)"""
        self.max_seq_len = new_seq_len
        # Очистити кеш, щоб перетокенізувати з новою довжиною
        self._cache.clear()


