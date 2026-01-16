"""
Модуль для автоматичного продовження навчання
Перенесено з config/training_resume.py
"""
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import hashlib
import torch

from core.types import TrainState


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Знайти останній checkpoint
    
    Args:
        checkpoint_dir: Директорія з checkpoint'ами
    
    Returns:
        Шлях до останнього checkpoint'у або None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Спочатку перевірити checkpoint_latest.pt
    latest_file = checkpoint_dir / "checkpoint_latest.pt"
    if latest_file.exists():
        return latest_file
    
    # Шукати за ім'ям файлу
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        # Сортувати за часом модифікації
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    # Також шукати ckpt_step_*.pt (новий формат)
    step_checkpoints = list(checkpoint_dir.glob("ckpt_step_*.pt"))
    if step_checkpoints:
        # Сортувати за номером кроку
        def get_step(path: Path) -> int:
            try:
                # Виділити номер з назви файлу (ckpt_step_123.pt -> 123)
                return int(path.stem.split('_')[-1])
            except:
                return 0
        
        step_checkpoints.sort(key=get_step, reverse=True)
        return step_checkpoints[0]
    
    return None


def get_checkpoint_info(checkpoint_path: Path) -> Dict:
    """
    Отримати інформацію про checkpoint
    
    Args:
        checkpoint_path: Шлях до checkpoint'у
    
    Returns:
        Інформація про checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Спробувати завантажити TrainState якщо він є
        if 'train_state' in checkpoint:
            train_state = TrainState.from_dict(checkpoint['train_state'])
            return {
                'step': train_state.step,
                'epoch': train_state.epoch,
                'batch_idx': train_state.batch_idx,
                'batch_count': train_state.step,
                'loss': train_state.loss,
                'recursion_depth': train_state.recursion_depth,
                'is_final': checkpoint.get('is_final', False),
                'train_state': train_state.to_dict()
            }
        else:
            # Fallback для старих checkpoint'ів
            return {
                'step': checkpoint.get('step', checkpoint.get('batch_count', 0)),
                'epoch': checkpoint.get('epoch', 0),
                'batch_idx': checkpoint.get('batch_idx', 0),
                'batch_count': checkpoint.get('batch_count', 0),
                'epochs': checkpoint.get('epochs', 0),
                'batch_size': checkpoint.get('batch_size', 0),
                'loss': checkpoint.get('loss', None),
                'is_final': checkpoint.get('is_final', False),
            }
    except Exception as e:
        return {'error': str(e)}


def load_train_state(checkpoint_path: Path) -> Optional[TrainState]:
    """
    Завантажити TrainState з checkpoint
    
    Args:
        checkpoint_path: Шлях до checkpoint'у
    
    Returns:
        TrainState або None
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'train_state' in checkpoint:
            return TrainState.from_dict(checkpoint['train_state'])
    except Exception:
        pass
    return None


def should_auto_resume(
    cfg: Dict,
    interactive: bool = False
) -> bool:
    """
    Визначити чи потрібно автоматично продовжувати навчання
    
    Args:
        cfg: Конфігурація
        interactive: Чи запущено в інтерактивному режимі
    
    Returns:
        True якщо потрібно автоматично продовжувати
    """
    # Перевірити налаштування в config
    auto_resume = cfg.get('training', {}).get('auto_resume', False)
    
    if not auto_resume:
        return False
    
    # В інтерактивному режимі завжди питати
    if interactive:
        return False
    
    # В service режимі - автоматично
    return True


def maybe_resume(
    trainer,
    cfg: Dict,
    logger,
    checkpoint_dir: Optional[str] = None
) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Перевірити та завантажити checkpoint для продовження навчання
    
    Args:
        trainer: Trainer instance
        cfg: Конфігурація
        logger: Logger instance
        checkpoint_dir: Директорія з checkpoint'ами
    
    Returns:
        (should_resume, checkpoint_path, checkpoint_info)
    """
    if checkpoint_dir is None:
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
    
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        return False, None, None
    
    checkpoint_info = get_checkpoint_info(latest_checkpoint)
    
    # Перевірити чи checkpoint фінальний
    if checkpoint_info.get('is_final', False):
        return False, None, None
    
    # Перевірити чи потрібно автоматично продовжувати
    if not should_auto_resume(cfg, interactive=False):
        return False, None, None
    
    # Завантажити checkpoint
    try:
        if hasattr(trainer, 'resume_from_checkpoint'):
            trainer.resume_from_checkpoint(str(latest_checkpoint))
        elif hasattr(trainer, 'load_checkpoint'):
            trainer.load_checkpoint(str(latest_checkpoint))
        else:
            # Загальний спосіб завантаження
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            if 'model' in checkpoint:
                trainer.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                trainer.optim.load_state_dict(checkpoint['optimizer'])
        
        logger.info(f"✅ Продовжено навчання з checkpoint: {latest_checkpoint.name}")
        logger.info(f"   Step: {checkpoint_info.get('step', 'N/A')}, Loss: {checkpoint_info.get('loss', 'N/A')}")
        
        return True, latest_checkpoint, checkpoint_info
    except Exception as e:
        logger.warning(f"⚠️ Не вдалося завантажити checkpoint: {e}")
        return False, None, None

