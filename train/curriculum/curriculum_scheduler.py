"""
Curriculum Scheduler для керування етапами навчання
Контролює послідовність зміни seq_len, batch_size, dim тощо
Curriculum як policy (складність → progression → reasoning length)
Dataset filters для типів задач (не curriculum)
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CurriculumStage:
    """
    Етап curriculum навчання (як policy, не жорстке керування типами задач)
    Curriculum керує складністю через progression, dataset filters керують типами задач
    """
    name: str
    seq_len: int
    dim: int
    batch: int
    epochs: int
    task_difficulty: Optional[float] = None  # Складність задачі (0.0 - 1.0) для policy
    max_recursion: Optional[int] = None  # Максимальна глибина рекурсії для цього етапу


class CurriculumScheduler:
    """
    Планувальник етапів curriculum навчання
    
    Контролює перехід між етапами на основі кількості пройдених епох.
    Кожен етап має свої параметри: seq_len, dim, batch, epochs.
    """
    
    def __init__(self, stages: List[dict]):
        """
        Ініціалізація scheduler зі списку stages
        
        Args:
            stages: Список словників з параметрами кожного stage
                   [{name, seq_len, dim, batch, epochs}, ...]
        """
        if not stages:
            raise ValueError("CurriculumScheduler requires at least one stage")
        
        self.stages = [CurriculumStage(**s) for s in stages]
        self.stage_idx = 0
        self.epoch_in_stage = 0
    
    @property
    def current(self) -> CurriculumStage:
        """Повертає поточний етап curriculum"""
        return self.stages[self.stage_idx]
    
    def on_epoch_end(self):
        """
        Викликається в кінці кожної епохи
        Перевіряє чи потрібно перейти до наступного stage
        """
        self.epoch_in_stage += 1
        current_stage = self.stages[self.stage_idx]
        
        # Якщо завершено всі епохи поточного stage
        if self.epoch_in_stage >= current_stage.epochs:
            # Перейти до наступного stage якщо він є
            if self.stage_idx < len(self.stages) - 1:
                self.stage_idx += 1
                self.epoch_in_stage = 0
                # Повертаємося до початку циклу, але вже з новим stage
            else:
                # Останній stage - залишаємося на ньому
                pass
    
    def describe(self) -> str:
        """
        Отримати опис поточного stage для логування
        
        Returns:
            Рядок з описом поточного stage
        """
        s = self.current
        return (f"[CURRICULUM] {s.name} "
                f"seq={s.seq_len} dim={s.dim} batch={s.batch} "
                f"epoch={self.epoch_in_stage+1}/{s.epochs} (stage {self.stage_idx+1}/{len(self.stages)})")
    
    def is_final_stage(self) -> bool:
        """Перевіряє чи поточний stage є останнім"""
        return self.stage_idx == len(self.stages) - 1
    
    def get_current_task_difficulty(self) -> Optional[float]:
        """Отримати поточну складність задачі (policy)"""
        stage = self.current
        return stage.task_difficulty if hasattr(stage, 'task_difficulty') else None
    
    def get_current_max_recursion(self) -> Optional[int]:
        """Отримати поточну максимальну глибину рекурсії"""
        stage = self.current
        return stage.max_recursion if hasattr(stage, 'max_recursion') else None

