"""
Фабрика для створення моделей TRM
"""
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
import torch

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, TransformerBackbone

from train.constants import (
    DEFAULT_DIM, DEFAULT_DEPTH, DEFAULT_SEQ_LEN,
    DEFAULT_NUM_REFINEMENT_BLOCKS, DEFAULT_NUM_LATENT_REFINEMENTS,
    DEFAULT_HALT_LOSS_WEIGHT, DEFAULT_VOCAB_SIZE
)
from train.exceptions import ModelConfigError


def create_model(
    dim: int = DEFAULT_DIM,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    depth: int = DEFAULT_DEPTH,
    seq_len: int = DEFAULT_SEQ_LEN,
    num_refinement_blocks: int = DEFAULT_NUM_REFINEMENT_BLOCKS,
    num_latent_refinements: int = DEFAULT_NUM_LATENT_REFINEMENTS,
    halt_loss_weight: float = DEFAULT_HALT_LOSS_WEIGHT,
    max_recursion_depth: int = 20,  # НОВИЙ: guard для рекурсії
    adaptive_recursion: bool = False,  # Адаптивна рекурсія
    timeout_seconds: Optional[float] = None,  # Timeout для recursion
    thinking_cost_weight: float = 0.01,  # Вага thinking cost
    network: Optional['MLPMixer1D'] = None,  # Можна передати готовий network
    use_transformer: bool = False,  # НОВИЙ: використати Transformer замість MLPMixer
    transformer_model: str = "gpt2",  # НОВИЙ: назва GPT-2 моделі
    transformer_pretrained: bool = True,  # НОВИЙ: завантажити готовий GPT-2
    transformer_cache_dir: Optional[str] = None,  # НОВИЙ: де зберігати завантажені моделі
    **kwargs
) -> 'TinyRecursiveModel':
    """
    Створити TRM модель з заданими параметрами
    
    Args:
        dim: Розмірність ембеддингів
        vocab_size: Розмір словника
        depth: Глибина MLP Mixer або Transformer
        seq_len: Довжина послідовності
        num_refinement_blocks: Кількість блоків уточнення
        num_latent_refinements: Кількість уточнень латентних змінних
        halt_loss_weight: Вага loss для раннього виходу
        max_recursion_depth: Максимальна глибина рекурсії (guard)
        adaptive_recursion: Чи використовувати адаптивну рекурсію
        timeout_seconds: Timeout для recursion (None = без timeout)
        thinking_cost_weight: Вага thinking cost в loss
        network: Готовий network (опціонально)
        use_transformer: Використати Transformer замість MLPMixer
        transformer_model: Назва GPT-2 моделі ("gpt2", "gpt2-medium", тощо)
        transformer_pretrained: Завантажити готовий GPT-2
        transformer_cache_dir: Де зберігати завантажені моделі
        **kwargs: Додаткові параметри
    
    Returns:
        Надініціалізована TRM модель
    """
    # Lazy import to avoid circular dependency
    from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, TransformerBackbone
    
    try:
        if network is None:
            if use_transformer:
                # Використати Transformer замість MLPMixer
                network = TransformerBackbone(
                    dim=dim,
                    depth=depth,
                    seq_len=seq_len,
                    pretrained=transformer_pretrained,
                    model_name=transformer_model,
                    cache_dir=transformer_cache_dir
                )
                # Оновити dim з реальної моделі (якщо використовується pretrained)
                if transformer_pretrained:
                    dim = network.dim
                    depth = network.depth
                    seq_len = network.seq_len
            else:
                # Старий підхід: MLPMixer
                network = MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
        
        model = TinyRecursiveModel(
            dim=dim,
            num_tokens=vocab_size,
            network=network,
            num_refinement_blocks=num_refinement_blocks,
            num_latent_refinements=num_latent_refinements,
            halt_loss_weight=halt_loss_weight,
            max_recursion_depth=max_recursion_depth,
            adaptive_recursion=adaptive_recursion,
            timeout_seconds=timeout_seconds,
            thinking_cost_weight=thinking_cost_weight,
            **kwargs
        )
        return model
    except Exception as e:
        raise ModelConfigError(f"Помилка створення моделі: {e}") from e


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_config: Dict[str, Any],
    device: str = 'cpu'
) -> 'TinyRecursiveModel':
    """
    Завантажити модель з checkpoint'у
    
    Args:
        checkpoint_path: Шлях до checkpoint'у
        model_config: Конфігурація моделі
        device: Пристрій для завантаження
    
    Returns:
        Завантажена модель
    """
    # Lazy import to avoid circular dependency
    from tiny_recursive_model import TinyRecursiveModel
    from train.exceptions import CheckpointError, ModelLoadError
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise CheckpointError(str(checkpoint_path), "Checkpoint не знайдено")
    
    try:
        # Створити модель
        model = create_model(**model_config)
        
        # Завантажити ваги
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        return model
    except Exception as e:
        raise ModelLoadError(str(checkpoint_path), f"Помилка завантаження: {e}") from e


def get_model_config_dict(
    dim: int,
    vocab_size: int,
    seq_len: int,
    depth: int,
    num_refinement_blocks: int = DEFAULT_NUM_REFINEMENT_BLOCKS,
    num_latent_refinements: int = DEFAULT_NUM_LATENT_REFINEMENTS,
    halt_loss_weight: float = DEFAULT_HALT_LOSS_WEIGHT,
    max_recursion_depth: int = 20,
    adaptive_recursion: bool = False,
    timeout_seconds: Optional[float] = None,
    thinking_cost_weight: float = 0.01
) -> Dict[str, Any]:
    """
    Створити словник конфігурації моделі
    
    Returns:
        Словник з конфігурацією моделі
    """
    return {
        'dim': dim,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'depth': depth,
        'num_refinement_blocks': num_refinement_blocks,
        'num_latent_refinements': num_latent_refinements,
        'halt_loss_weight': halt_loss_weight,
        'max_recursion_depth': max_recursion_depth,
        'adaptive_recursion': adaptive_recursion,
        'timeout_seconds': timeout_seconds,
        'thinking_cost_weight': thinking_cost_weight
    }

