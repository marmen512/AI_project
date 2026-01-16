"""
Автоматичний curriculum tuner
Адаптивно налаштовує curriculum на основі метрик навчання
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Етап curriculum"""
    seq_len: int
    batch_size: int
    learning_rate: float


class AutoCurriculumTuner:
    """
    Автоматичний налаштувач curriculum
    Адаптивно змінює параметри на основі метрик
    """
    
    def __init__(
        self,
        loss_drop_threshold: float = 0.1,
        min_speed_threshold: float = 1.0,  # tokens/sec
        max_seq_len: int = 512,
        min_batch_size: int = 1,
        seq_len_multiplier: float = 2.0,
        batch_size_step: int = 1
    ):
        """
        Ініціалізація tuner'а
        
        Args:
            loss_drop_threshold: Поріг зниження loss для збільшення seq_len
            min_speed_threshold: Мінімальна швидкість для зменшення batch_size
            max_seq_len: Максимальна довжина послідовності
            min_batch_size: Мінімальний розмір батчу
            seq_len_multiplier: Множник для збільшення seq_len
            batch_size_step: Крок зміни batch_size
        """
        self.loss_drop_threshold = loss_drop_threshold
        self.min_speed_threshold = min_speed_threshold
        self.max_seq_len = max_seq_len
        self.min_batch_size = min_batch_size
        self.seq_len_multiplier = seq_len_multiplier
        self.batch_size_step = batch_size_step
        
        # Історія метрик
        self.metrics_history = []
    
    def update(
        self,
        metrics: Dict[str, Any],
        stage: CurriculumStage
    ) -> CurriculumStage:
        """
        Оновити stage на основі метрик
        
        Args:
            metrics: Метрики навчання з ключами:
                - loss: Поточний loss
                - tokens_per_sec: Швидкість генерації токенів
                - prev_loss: Попередній loss (опціонально)
            stage: Поточний curriculum stage
        
        Returns:
            Оновлений stage
        """
        # Зберегти метрики в історію
        self.metrics_history.append(metrics.copy())
        
        # Обмежити історію
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        new_stage = CurriculumStage(
            seq_len=stage.seq_len,
            batch_size=stage.batch_size,
            learning_rate=stage.learning_rate
        )
        
        # Перевірити чи потрібно збільшити seq_len
        current_loss = metrics.get('loss', float('inf'))
        prev_loss = metrics.get('prev_loss')
        
        if prev_loss is not None:
            loss_drop = prev_loss - current_loss
            if loss_drop >= self.loss_drop_threshold:
                # Збільшити seq_len
                new_seq_len = int(stage.seq_len * self.seq_len_multiplier)
                new_stage.seq_len = min(new_seq_len, self.max_seq_len)
                print(f"📈 Збільшено seq_len: {stage.seq_len} → {new_stage.seq_len} (loss drop: {loss_drop:.4f})")
        
        # Перевірити чи потрібно зменшити batch_size
        tokens_per_sec = metrics.get('tokens_per_sec', float('inf'))
        if tokens_per_sec < self.min_speed_threshold:
            # Зменшити batch_size
            new_batch_size = max(stage.batch_size - self.batch_size_step, self.min_batch_size)
            if new_batch_size < stage.batch_size:
                new_stage.batch_size = new_batch_size
                print(f"📉 Зменшено batch_size: {stage.batch_size} → {new_stage.batch_size} (швидкість: {tokens_per_sec:.2f} tokens/sec)")
        
        return new_stage
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Отримати рекомендації на основі історії метрик
        
        Returns:
            Словник з рекомендаціями
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Останні 10 записів
        
        avg_loss = sum(m.get('loss', 0) for m in recent_metrics) / len(recent_metrics)
        avg_speed = sum(m.get('tokens_per_sec', 0) for m in recent_metrics) / len(recent_metrics)
        
        recommendations = {
            'avg_loss': avg_loss,
            'avg_speed': avg_speed,
            'should_increase_seq_len': avg_loss < self.loss_drop_threshold,
            'should_decrease_batch_size': avg_speed < self.min_speed_threshold,
        }
        
        return recommendations

