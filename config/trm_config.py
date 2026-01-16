"""
TRM-оптимальна конфігурація навчання
Замість LLM-стайл параметрів, використовуємо параметри оптимізовані для TRM
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class TRMConfig:
    """Конфігурація для TRM навчання - оптимізована для рекурсивних моделей"""
    
    # Версія конфігурації
    config_version: str = "1.0"  # Версія конфігурації для міграції
    
    # Модель (TRM-оптимальні розміри)
    dim: int = 256  # Замість 1024 - менше для швидкості
    depth: int = 4
    seq_len: int = 256  # Замість 4096 - набагато менше для TRM
    vocab_size: Optional[int] = None  # Автоматично з tokenizer
    
    # Навчання
    batch_size: int = 4  # Замість 1 - більше для ефективності
    effective_batch_size: int = 16  # batch_size * gradient_accumulation
    epochs: int = 10
    learning_rate: float = 1e-4
    
    # Рекурсія
    max_recurrent_steps: int = 12
    halt_prob_thres: float = 0.5
    max_recursion_depth: int = 20  # НОВИЙ: guard для рекурсії в моделі
    adaptive_recursion: bool = False  # Увімкнути adaptive recursion gate (потрібен thinking cost)
    
    # Curriculum learning (для TRM важливо!)
    curriculum_enabled: bool = True
    curriculum_start_len: int = 64
    curriculum_max_len: int = 256
    curriculum_stages: int = 4  # Кількість етапів
    curriculum_epochs_per_stage: int = 3  # Скільки епох на кожному рівні
    
    # Оптимізація
    warmup_steps: int = 1000
    weight_decay: float = 1.0
    
    # Thinking cost (для оптимізації мислення, не просто мінімізації кроків)
    thinking_cost_weight: float = 0.01  # Вага thinking cost в loss
    
    # Dataset
    cache_size: int = 1000  # Кеш для lazy loading
    validate_format: bool = True
    
    @property
    def gradient_accumulation_steps(self) -> int:
        """Автоматично обчислити з effective_batch_size"""
        return max(1, self.effective_batch_size // self.batch_size)
    
    def validate(self):
        """Перевірити коректність конфігурації"""
        assert self.dim > 0, "dim повинен бути > 0"
        assert 0 < self.halt_prob_thres <= 1, "halt_prob_thres повинен бути в (0, 1]"
        assert self.max_recurrent_steps > 0, "max_recurrent_steps повинен бути > 0"
        assert self.seq_len >= self.curriculum_start_len, "seq_len повинен бути >= curriculum_start_len"
        assert self.curriculum_max_len >= self.curriculum_start_len, "curriculum_max_len повинен бути >= curriculum_start_len"
        assert self.batch_size > 0, "batch_size повинен бути > 0"
        assert self.effective_batch_size >= self.batch_size, "effective_batch_size повинен бути >= batch_size"
    
    def to_dict(self) -> dict:
        """Конвертувати в словник"""
        return {
            'config_version': self.config_version,
            'dim': self.dim,
            'depth': self.depth,
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'effective_batch_size': self.effective_batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'max_recurrent_steps': self.max_recurrent_steps,
            'halt_prob_thres': self.halt_prob_thres,
            'max_recursion_depth': self.max_recursion_depth,
            'curriculum_enabled': self.curriculum_enabled,
            'curriculum_start_len': self.curriculum_start_len,
            'curriculum_max_len': self.curriculum_max_len,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TRMConfig':
        """Створити зі словника"""
        config_version = d.get('config_version', '1.0')
        # Міграція конфігурації (для майбутніх версій)
        if config_version == '1.0':
            pass  # Поточна версія
        # Тут можна додати міграції для майбутніх версій
        
        # Фільтрувати тільки валідні поля
        valid_fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        auto_detect: bool = True,
        **overrides
    ) -> 'TRMConfig':
        """
        Створити конфігурацію з автоматичним визначенням параметрів на основі датасету
        
        Args:
            dataset_path: Шлях до датасету
            auto_detect: Автоматично визначити параметри на основі датасету
            **overrides: Параметри для перевизначення
        
        Returns:
            TRMConfig з автоматично визначеними параметрами
        """
        if not auto_detect:
            return cls(**overrides)
        
        dataset_path = Path(dataset_path)
        
        # Отримати інформацію про датасет
        dataset_size = 0
        dataset_samples = 0
        
        if dataset_path.exists():
            dataset_size = dataset_path.stat().st_size
            
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    dataset_samples = len(data)
                elif isinstance(data, dict) and 'data' in data:
                    dataset_samples = len(data['data'])
            except:
                pass
        
        # Оцінити доступну пам'ять
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        except:
            available_memory_gb = 8.0
        
        # Автоматично визначити параметри
        # Batch size на основі розміру датасету та пам'яті
        if dataset_samples < 100:
            batch_size = 2
            epochs = 20
        elif dataset_samples < 500:
            batch_size = 4
            epochs = 15
        elif dataset_samples < 2000:
            batch_size = 4 if available_memory_gb < 8 else 8
            epochs = 12
        elif dataset_samples < 10000:
            batch_size = 8 if available_memory_gb < 16 else 16
            epochs = 10
        else:
            batch_size = 16 if available_memory_gb < 16 else 32
            epochs = 8
        
        # Learning rate на основі batch_size
        learning_rate = 1e-4
        if batch_size >= 16:
            learning_rate = 2e-4
        elif batch_size <= 2:
            learning_rate = 5e-5
        
        # Gradient accumulation
        effective_batch_size = 16
        if batch_size < 4:
            effective_batch_size = 8
        elif batch_size >= 16:
            effective_batch_size = 32
        
        # Warmup steps
        batches_per_epoch = max(1, dataset_samples // batch_size)
        warmup_steps = min(2000, max(100, batches_per_epoch * 2))
        
        # Max recurrent steps
        max_recurrent_steps = 12
        if dataset_samples > 10000:
            max_recurrent_steps = 16
        elif dataset_samples < 100:
            max_recurrent_steps = 8
        
        # Створити конфігурацію з автоматично визначеними параметрами
        config = cls(
            batch_size=batch_size,
            effective_batch_size=effective_batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            max_recurrent_steps=max_recurrent_steps,
            warmup_steps=warmup_steps,
            **overrides
        )
        
        return config
    
    def print_summary(self, dataset_path: Optional[str | Path] = None) -> None:
        """Вивести підсумок конфігурації"""
        print("\n" + "=" * 70)
        print("⚙️  КОНФІГУРАЦІЯ TRM НАВЧАННЯ")
        print("=" * 70)
        
        if dataset_path:
            dataset_path = Path(dataset_path)
            if dataset_path.exists():
                dataset_size = dataset_path.stat().st_size
                print(f"\n📊 Датасет:")
                print(f"   - Файл: {dataset_path.name}")
                print(f"   - Розмір: {dataset_size / (1024*1024):.2f} MB")
        
        print(f"\n🎯 Параметри моделі:")
        print(f"   - dim: {self.dim}")
        print(f"   - depth: {self.depth}")
        print(f"   - seq_len: {self.seq_len}")
        if self.curriculum_enabled:
            print(f"   - curriculum: {self.curriculum_start_len} → {self.curriculum_max_len}")
        
        print(f"\n🎓 Параметри навчання:")
        print(f"   - epochs: {self.epochs}")
        print(f"   - batch_size: {self.batch_size}")
        print(f"   - effective_batch_size: {self.effective_batch_size}")
        print(f"   - gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        print(f"   - learning_rate: {self.learning_rate:.2e}")
        print(f"   - warmup_steps: {self.warmup_steps}")
        print(f"   - max_recurrent_steps: {self.max_recurrent_steps}")
        print(f"   - halt_prob_thres: {self.halt_prob_thres}")
        print(f"   - max_recursion_depth: {self.max_recursion_depth}")
        
        print("=" * 70 + "\n")


@dataclass
class CurriculumStage:
    """Етап curriculum learning (як policy, не жорстке керування типами задач)"""
    seq_len: int
    epochs: int
    description: str
    task_difficulty: Optional[float] = None  # Складність задачі (0.0 - 1.0)
    max_recursion: Optional[int] = None  # Максимальна глибина рекурсії для цього етапу
    # Примітка: типи задач не керуються тут - це робиться через dataset filters


class CurriculumScheduler:
    """Планувальник curriculum learning для TRM"""
    
    def __init__(
        self,
        start_len: int = 64,
        max_len: int = 256,
        stages: int = 4,
        epochs_per_stage: int = 3
    ):
        self.stages = self._create_stages(start_len, max_len, stages, epochs_per_stage)
        self.current_stage = 0
        self.current_epoch_in_stage = 0
    
    def _create_stages(self, start: int, max: int, n: int, epochs: int) -> list[CurriculumStage]:
        """Створити етапи curriculum"""
        if n == 1:
            return [CurriculumStage(seq_len=max, epochs=epochs, description=f"Stage 1: seq_len={max}")]
        
        step = (max - start) / (n - 1)
        stages = []
        for i in range(n):
            seq_len = int(start + step * i)
            stages.append(CurriculumStage(
                seq_len=seq_len,
                epochs=epochs,
                description=f"Stage {i+1}/{n}: seq_len={seq_len}"
            ))
        return stages
    
    def get_current_seq_len(self) -> int:
        """Отримати поточну seq_len"""
        return self.stages[self.current_stage].seq_len
    
    def advance_epoch(self):
        """Перейти до наступної епохи"""
        self.current_epoch_in_stage += 1
        if self.current_epoch_in_stage >= self.stages[self.current_stage].epochs:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.current_epoch_in_stage = 0
                print(f"📈 Curriculum: {self.stages[self.current_stage].description}")
    
    def is_complete(self) -> bool:
        """Чи завершено curriculum"""
        return (self.current_stage == len(self.stages) - 1 and 
                self.current_epoch_in_stage >= self.stages[-1].epochs)
    
    def get_current_stage_info(self) -> str:
        """Отримати інформацію про поточний етап"""
        stage = self.stages[self.current_stage]
        return f"{stage.description} (epoch {self.current_epoch_in_stage + 1}/{stage.epochs})"


