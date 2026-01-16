"""
Factory для створення datasets
Підтримує TRMDataset, RAGDatasetWrapper, DatasetMixer, PuzzleDataset
"""
from typing import Dict, Any, Optional
from pathlib import Path
from torch.utils.data import Dataset

from train.datasets.trm_dataset import TRMDataset


def build_dataset(
    cfg: Dict[str, Any],
    tokenizer,
    pad_token_id: int
) -> Dataset:
    """
    Створити dataset з конфігурації
    
    Args:
        cfg: Конфігурація з ключами:
            - dataset.type: Тип dataset ("trm", "rag", "mixer", "puzzle") (default: "trm")
            - dataset.path: Шлях до датасету (обов'язково для TRMDataset)
            - dataset.max_seq_len: Максимальна довжина послідовності (default: 256)
            - dataset.cache_size: Розмір кешу (default: 1000)
            - dataset.validate_format: Валідувати формат (default: True)
            - rag.enabled: Увімкнути RAG (default: False)
            - rag.backend: Backend для RAG ("memory", "faiss") (default: "memory")
        tokenizer: Tokenizer instance
        pad_token_id: ID padding token'а
    
    Returns:
        Dataset instance
    """
    dataset_cfg = cfg.get('dataset', {})
    dataset_type = dataset_cfg.get('type', 'trm').lower()
    
    if dataset_type == 'trm':
        return _build_trm_dataset(cfg, tokenizer, pad_token_id)
    elif dataset_type == 'rag':
        return _build_rag_dataset(cfg, tokenizer, pad_token_id)
    elif dataset_type == 'mixer':
        return _build_mixer_dataset(cfg, tokenizer, pad_token_id)
    elif dataset_type == 'puzzle':
        return _build_puzzle_dataset(cfg, tokenizer, pad_token_id)
    else:
        raise ValueError(f"Невідомий тип dataset: {dataset_type}")


def _build_trm_dataset(
    cfg: Dict[str, Any],
    tokenizer,
    pad_token_id: int
) -> TRMDataset:
    """Створити TRMDataset"""
    dataset_cfg = cfg.get('dataset', {})
    dataset_path = dataset_cfg.get('path')
    
    if dataset_path is None:
        raise ValueError("dataset.path повинен бути вказаний для TRMDataset")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Датасет не знайдено: {dataset_path}")
    
    max_seq_len = dataset_cfg.get('max_seq_len', 256)
    cache_size = dataset_cfg.get('cache_size', 1000)
    validate_format = dataset_cfg.get('validate_format', True)
    
    return TRMDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        cache_size=cache_size,
        validate_format=validate_format
    )


def _build_rag_dataset(
    cfg: Dict[str, Any],
    tokenizer,
    pad_token_id: int
) -> Dataset:
    """Створити RAGDatasetWrapper"""
    # TODO: Після створення RAGDatasetWrapper
    # from train.datasets.rag_dataset_wrapper import RAGDatasetWrapper
    # from rag.memory_store import MemoryVectorStore
    # from rag.faiss_store import FAISSVectorStore
    
    rag_cfg = cfg.get('rag', {})
    backend = rag_cfg.get('backend', 'memory').lower()
    
    # Тимчасово використовуємо TRMDataset
    # Після створення RAGDatasetWrapper замінити
    return _build_trm_dataset(cfg, tokenizer, pad_token_id)


def _build_mixer_dataset(
    cfg: Dict[str, Any],
    tokenizer,
    pad_token_id: int
) -> Dataset:
    """Створити DatasetMixer для multi-dataset curriculum"""
    # TODO: Після створення DatasetMixer
    # from train.datasets.dataset_mixer import DatasetMixer
    
    # Тимчасово використовуємо TRMDataset
    return _build_trm_dataset(cfg, tokenizer, pad_token_id)


def _build_puzzle_dataset(
    cfg: Dict[str, Any],
    tokenizer,
    pad_token_id: int
) -> Dataset:
    """Створити PuzzleDataset"""
    # TODO: Після створення PuzzleDataset
    # from train.datasets.puzzle_dataset import PuzzleDataset
    
    # Тимчасово використовуємо TRMDataset
    return _build_trm_dataset(cfg, tokenizer, pad_token_id)

