"""
Автоматичні checkpoints з cleanup старих checkpoint'ів та best model tracking
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json

from core.constants import (
    CHECKPOINT_BEST_LOSS, 
    CHECKPOINT_BEST_EVAL, 
    CHECKPOINT_BEST_ENTROPY
)


class CheckpointManager:
    """
    Менеджер для автоматичних checkpoint'ів
    Підтримує cleanup старих checkpoint'ів та resume graph через metrics.jsonl
    """
    
    def __init__(self, path: str = "checkpoints", keep_last: int = 5):
        """
        Ініціалізація менеджера
        
        Args:
            path: Шлях до директорії з checkpoint'ами
            keep_last: Скільки останніх checkpoint'ів зберігати
        """
        self.path = Path(path)
        self.keep_last = keep_last
        self.path.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        extra: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Зберегти checkpoint
        
        Args:
            step: Номер кроку
            model: Модель
            optimizer: Optimizer
            extra: Додаткові дані для збереження
        
        Returns:
            Шлях до збереженого checkpoint'у
        """
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "extra": extra or {}
        }
        
        # Зберегти checkpoint
        p = self.path / f"ckpt_step_{step}.pt"
        torch.save(ckpt, p)
        
        # Оновити checkpoint_latest.pt
        latest_path = self.path / "checkpoint_latest.pt"
        torch.save(ckpt, latest_path)
        
        # Cleanup старих checkpoint'ів
        self._cleanup()
        
        return p
    
    def latest(self) -> Optional[Path]:
        """
        Знайти останній checkpoint
        
        Returns:
            Шлях до останнього checkpoint'у або None
        """
        # Спочатку перевірити checkpoint_latest.pt
        latest_path = self.path / "checkpoint_latest.pt"
        if latest_path.exists():
            return latest_path
        
        # Шукати ckpt_step_*.pt
        ckpts = sorted(self.path.glob("ckpt_step_*.pt"))
        if ckpts:
            return ckpts[-1]
        
        # Також шукати checkpoint_*.pt (старий формат)
        old_ckpts = sorted(self.path.glob("checkpoint_*.pt"))
        if old_ckpts:
            return old_ckpts[-1]
        
        return None
    
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> int:
        """
        Завантажити останній checkpoint
        
        Args:
            model: Модель для завантаження
            optimizer: Optimizer для завантаження
        
        Returns:
            Номер кроку (0 якщо checkpoint не знайдено)
        """
        ckpt_path = self.latest()
        if not ckpt_path:
            return 0
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("step", 0)
    
    def _cleanup(self):
        """
        Видалити старі checkpoint'и, залишивши тільки keep_last останніх
        """
        # Знайти всі checkpoint'и
        ckpts = sorted(self.path.glob("ckpt_step_*.pt"))
        
        # Залишити тільки останні keep_last
        if len(ckpts) > self.keep_last:
            for p in ckpts[:-self.keep_last]:
                try:
                    p.unlink()
                except Exception:
                    pass  # Ігнорувати помилки видалення
    
    def save_best_loss(
        self,
        loss: float,
        model_state_dict: Dict,
        optimizer_state_dict: Dict,
        train_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Зберегти best loss checkpoint (PRIMARY метрика)
        
        Args:
            loss: Поточний loss
            model_state_dict: State dict моделі
            optimizer_state_dict: State dict optimizer
            train_state: TrainState для збереження
        
        Returns:
            True якщо checkpoint збережено (новий best)
        """
        best_path = self.path / CHECKPOINT_BEST_LOSS
        
        # Перевірити чи є попередній best
        current_best = None
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location='cpu')
                current_best = best_ckpt.get('loss')
            except:
                pass
        
        # Зберегти якщо це новий best (менший loss)
        if current_best is None or loss < current_best:
            ckpt = {
                'loss': loss,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
            }
            if train_state:
                ckpt['train_state'] = train_state
            
            torch.save(ckpt, best_path)
            return True
        return False
    
    def save_best_eval_score(
        self,
        eval_score: float,
        model_state_dict: Dict,
        optimizer_state_dict: Dict,
        train_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Зберегти best eval score checkpoint (PRIMARY метрика)
        
        Args:
            eval_score: Поточний eval score
            model_state_dict: State dict моделі
            optimizer_state_dict: State dict optimizer
            train_state: TrainState для збереження
        
        Returns:
            True якщо checkpoint збережено (новий best)
        """
        best_path = self.path / CHECKPOINT_BEST_EVAL
        
        # Перевірити чи є попередній best
        current_best = None
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location='cpu')
                current_best = best_ckpt.get('eval_score')
            except:
                pass
        
        # Зберегти якщо це новий best (більший score)
        if current_best is None or eval_score > current_best:
            ckpt = {
                'eval_score': eval_score,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
            }
            if train_state:
                ckpt['train_state'] = train_state
            
            torch.save(ckpt, best_path)
            return True
        return False
    
    def save_best_entropy(
        self,
        entropy: float,
        model_state_dict: Dict,
        optimizer_state_dict: Dict,
        train_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Зберегти best entropy checkpoint (SECONDARY метрика)
        
        Args:
            entropy: Поточний entropy
            model_state_dict: State dict моделі
            optimizer_state_dict: State dict optimizer
            train_state: TrainState для збереження
        
        Returns:
            True якщо checkpoint збережено (новий best)
        """
        best_path = self.path / CHECKPOINT_BEST_ENTROPY
        
        # Перевірити чи є попередній best
        current_best = None
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location='cpu')
                current_best = best_ckpt.get('entropy')
            except:
                pass
        
        # Зберегти якщо це новий best (більший entropy - більша невизначеність)
        # Але це secondary метрика, тому не критично
        if current_best is None or entropy > current_best:
            ckpt = {
                'entropy': entropy,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
            }
            if train_state:
                ckpt['train_state'] = train_state
            
            torch.save(ckpt, best_path)
            return True
        return False
    
    def get_best_loss_checkpoint(self) -> Optional[Path]:
        """Отримати шлях до best loss checkpoint"""
        path = self.path / CHECKPOINT_BEST_LOSS
        return path if path.exists() else None
    
    def get_best_eval_checkpoint(self) -> Optional[Path]:
        """Отримати шлях до best eval score checkpoint"""
        path = self.path / CHECKPOINT_BEST_EVAL
        return path if path.exists() else None
    
    def get_best_entropy_checkpoint(self) -> Optional[Path]:
        """Отримати шлях до best entropy checkpoint"""
        path = self.path / CHECKPOINT_BEST_ENTROPY
        return path if path.exists() else None
    
    def get_all_best_checkpoints(self) -> Dict[str, Optional[Path]]:
        """
        Отримати всі best checkpoints.
        
        Returns:
            Dict з шляхами до best checkpoints:
            {
                'best_loss': Path or None,
                'best_eval': Path or None,
                'best_entropy': Path or None
            }
        """
        return {
            'best_loss': self.get_best_loss_checkpoint(),
            'best_eval': self.get_best_eval_checkpoint(),
            'best_entropy': self.get_best_entropy_checkpoint()
        }
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Валідувати checkpoint файл.
        
        Args:
            checkpoint_path: Шлях до checkpoint
            
        Returns:
            True якщо checkpoint валідний, False інакше
        """
        if not checkpoint_path.exists():
            return False
        
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            # Перевірити обов'язкові поля
            required_fields = ['model_state_dict', 'optimizer_state_dict']
            return all(field in ckpt for field in required_fields)
        except Exception:
            return False

