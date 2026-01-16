"""
Factory для створення моделей
Перенесено з train/model_factory.py (без runtime-логіки)
"""
from typing import Dict, Any, Optional
from pathlib import Path
import torch

from tiny_recursive_model import TinyRecursiveModel
from train.model_factory import create_model as _create_model
from train.constants import DEFAULT_DIM, DEFAULT_DEPTH, DEFAULT_SEQ_LEN


def build_model(cfg: Dict[str, Any]) -> TinyRecursiveModel:
    """
    Створити TRM модель з конфігурації
    
    Args:
        cfg: Конфігурація моделі з ключами:
            - dim: Розмірність (default: 256)
            - vocab_size: Розмір словника (обов'язково)
            - depth: Глибина (default: 4)
            - seq_len: Довжина послідовності (default: 256)
            - max_recursion_depth: Максимальна глибина рекурсії (default: 20)
    
    Returns:
        Надініціалізована TRM модель
    """
    model_cfg = cfg.get('model', {})
    
    # Обов'язкові параметри
    vocab_size = model_cfg.get('vocab_size')
    if vocab_size is None:
        raise ValueError("vocab_size повинен бути вказаний в конфігурації")
    
    # Опціональні параметри з дефолтами
    dim = model_cfg.get('dim', DEFAULT_DIM)
    depth = model_cfg.get('depth', DEFAULT_DEPTH)
    seq_len = model_cfg.get('seq_len', DEFAULT_SEQ_LEN)
    max_recursion_depth = model_cfg.get('max_recursion_depth', 20)
    
    # Параметри Transformer (якщо використовується)
    use_transformer = model_cfg.get('use_transformer', False)
    transformer_model = model_cfg.get('transformer_model', 'gpt2')
    transformer_pretrained = model_cfg.get('transformer_pretrained', True)
    transformer_cache_dir = model_cfg.get('transformer_cache_dir', None)
    
    # Інші параметри моделі
    num_refinement_blocks = model_cfg.get('num_refinement_blocks', 3)
    num_latent_refinements = model_cfg.get('num_latent_refinements', 6)
    halt_loss_weight = model_cfg.get('halt_loss_weight', 1.0)
    adaptive_recursion = model_cfg.get('adaptive_recursion', False)
    timeout_seconds = model_cfg.get('timeout_seconds', None)
    thinking_cost_weight = model_cfg.get('thinking_cost_weight', 0.01)
    
    # Створити модель
    model = _create_model(
        dim=dim,
        vocab_size=vocab_size,
        depth=depth,
        seq_len=seq_len,
        max_recursion_depth=max_recursion_depth,
        num_refinement_blocks=num_refinement_blocks,
        num_latent_refinements=num_latent_refinements,
        halt_loss_weight=halt_loss_weight,
        adaptive_recursion=adaptive_recursion,
        timeout_seconds=timeout_seconds,
        thinking_cost_weight=thinking_cost_weight,
        use_transformer=use_transformer,
        transformer_model=transformer_model,
        transformer_pretrained=transformer_pretrained,
        transformer_cache_dir=transformer_cache_dir
    )
    
    return model

