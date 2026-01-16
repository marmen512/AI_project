"""
Утиліти для роботи з датасетами
"""
import torch
from torch.utils.data import Dataset, random_split
from typing import Tuple, Optional
import warnings

# Імпортувати безпечне розділення по doc_id
try:
    from train.datasets.splits import split_by_doc_id, validate_split_integrity
except ImportError:
    split_by_doc_id = None
    validate_split_integrity = None

# Імпортувати DatasetManifest для contamination control
try:
    from data.dataset_manifest import DatasetManifest
except ImportError:
    DatasetManifest = None


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    val_ratio: Optional[float] = None,
    seed: int = 42,
    safe_split: bool = True  # Використовувати split_by_doc_id якщо доступно
) -> Tuple[Dataset, Dataset]:
    """
    Розділити датасет на train та validation.
    
    Якщо safe_split=True та dataset підтримує doc_id, використовує split_by_doc_id
    для запобігання eval leakage. Інакше використовує random_split.
    
    Args:
        dataset: Датасет для розділення
        train_ratio: Частка датасету для навчання (0.0-1.0)
        val_ratio: Частка датасету для валідації (None = 1 - train_ratio)
        seed: Seed для генератора випадкових чисел
        safe_split: Використовувати безпечне розділення по doc_id (якщо доступно)
    
    Returns:
        (train_dataset, val_dataset)
    """
    if val_ratio is None:
        val_ratio = 1.0 - train_ratio
    
    assert 0 < train_ratio < 1, f"train_ratio має бути між 0 та 1, отримано {train_ratio}"
    assert 0 < val_ratio < 1, f"val_ratio має бути між 0 та 1, отримано {val_ratio}"
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, f"train_ratio + val_ratio має дорівнювати 1.0"
    
    # Спробувати використати безпечне розділення по doc_id
    if safe_split and split_by_doc_id is not None:
        try:
            train_dataset, val_dataset = split_by_doc_id(dataset, train_ratio=train_ratio)
            # Валідувати що немає overlap
            if validate_split_integrity is not None:
                is_valid = validate_split_integrity(train_dataset, val_dataset)
                if not is_valid:
                    warnings.warn(
                        "⚠️ Валідація split_integrity не пройдена! Можливий eval leakage. "
                        "Перевірте що doc_id правильно використовуються в dataset.",
                        UserWarning
                    )
            return train_dataset, val_dataset
        except (ValueError, AttributeError, TypeError) as e:
            # Якщо dataset не підтримує doc_id, fallback до random_split
            warnings.warn(
                f"⚠️ Dataset не підтримує doc_id для безпечного split: {e}. "
                "Використовується random_split (можливий eval leakage).",
                UserWarning
            )
            pass
    
    # Fallback до random_split
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset

