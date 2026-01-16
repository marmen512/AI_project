"""
Модуль для автоматичного визначення параметрів навчання
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
import os


class TrainingConfig:
    """Базова конфігурація навчання"""
    
    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        max_recurrent_steps: int = 12,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 1000,
        **kwargs
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_recurrent_steps = max_recurrent_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертувати в словник"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_recurrent_steps': self.max_recurrent_steps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'warmup_steps': self.warmup_steps,
            **self.extra_params
        }


class AutoTrainingConfig:
    """
    Автоматичне визначення параметрів навчання на основі датасету та ресурсів
    DEPRECATED: Використовуйте config.yaml з runtime.bootstrap замість цього
    """
    import warnings
    warnings.warn(
        "AutoTrainingConfig застарів. Використовуйте config.yaml з runtime.bootstrap",
        DeprecationWarning,
        stacklevel=2
    )
    
    def __init__(self, dataset_path: str | Path, models_dir: str | Path = None):
        """
        Ініціалізація з автоматичним визначенням параметрів
        
        Args:
            dataset_path: Шлях до датасету
            models_dir: Шлях до папки з моделями (для оцінки доступних ресурсів)
        """
        self.dataset_path = Path(dataset_path)
        self.models_dir = Path(models_dir) if models_dir else None
        
        # Завантажити датасет для аналізу (без side-effects)
        self.dataset_size = self._get_dataset_size()
        self.dataset_samples = self._count_samples()
        
        # Визначити параметри автоматично (без side-effects)
        self.config = self._auto_configure()
    
    def _get_dataset_size(self) -> int:
        """Отримати розмір датасету в байтах"""
        if not self.dataset_path.exists():
            return 0
        return self.dataset_path.stat().st_size
    
    def _count_samples(self) -> int:
        """Підрахувати кількість прикладів у датасеті"""
        if not self.dataset_path.exists():
            return 0
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict) and 'data' in data:
                return len(data['data'])
            return 0
        except:
            return 0
    
    def _estimate_available_memory(self) -> float:
        """Оцінити доступну пам'ять (GB)"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except:
            # За замовчуванням припускаємо 8GB
            return 8.0
    
    def _auto_configure(self) -> TrainingConfig:
        """Автоматично визначити оптимальні параметри"""
        samples = self.dataset_samples
        dataset_size_mb = self.dataset_size / (1024 * 1024)
        available_memory_gb = self._estimate_available_memory()
        
        # Визначити batch_size на основі розміру датасету та пам'яті
        if samples < 100:
            batch_size = 2
            epochs = 20  # Більше епох для малих датасетів
        elif samples < 500:
            batch_size = 4
            epochs = 15
        elif samples < 2000:
            batch_size = 4 if available_memory_gb < 8 else 8
            epochs = 12
        elif samples < 10000:
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
        
        # Gradient accumulation для стабільного навчання
        gradient_accumulation_steps = max(1, 16 // batch_size)
        
        # Warmup steps на основі розміру датасету
        batches_per_epoch = max(1, samples // batch_size)
        warmup_steps = min(2000, max(100, batches_per_epoch * 2))
        
        # Max recurrent steps
        max_recurrent_steps = 12
        if samples > 10000:
            max_recurrent_steps = 16
        elif samples < 100:
            max_recurrent_steps = 8
        
        return TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_recurrent_steps=max_recurrent_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps
        )
    
    def get_config(self) -> TrainingConfig:
        """Отримати конфігурацію"""
        return self.config
    
    def print_summary(self) -> None:
        """Вивести підсумок автоматичної конфігурації"""
        print("\n" + "=" * 70)
        print("⚙️  АВТОМАТИЧНА КОНФІГУРАЦІЯ НАВЧАННЯ")
        print("=" * 70)
        print(f"\n📊 Аналіз датасету:")
        print(f"   - Файл: {self.dataset_path.name}")
        print(f"   - Розмір: {self.dataset_size / (1024*1024):.2f} MB")
        print(f"   - Прикладів: {self.dataset_samples:,}")
        
        print(f"\n🎯 Рекомендовані параметри:")
        config = self.config
        print(f"   - Епох: {config.epochs}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Learning rate: {config.learning_rate:.2e}")
        print(f"   - Max recurrent steps: {config.max_recurrent_steps}")
        print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"   - Warmup steps: {config.warmup_steps}")
        
        # Оцінка часу
        batches_per_epoch = max(1, self.dataset_samples // config.batch_size)
        total_batches = batches_per_epoch * config.epochs
        estimated_time_min = total_batches * 0.5  # Приблизно 0.5 сек на батч
        
        print(f"\n⏱️  Оцінка:")
        print(f"   - Батчів на епоху: {batches_per_epoch}")
        print(f"   - Загалом батчів: {total_batches}")
        print(f"   - Орієнтовний час: ~{estimated_time_min/60:.1f} хвилин")
        print("=" * 70 + "\n")
    
    def save_config(self, output_file: str = "training_config.json") -> None:
        """Зберегти конфігурацію в JSON"""
        config_dict = {
            'dataset': {
                'path': str(self.dataset_path),
                'size_mb': self.dataset_size / (1024 * 1024),
                'samples': self.dataset_samples
            },
            'config': self.config.to_dict()
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Конфігурація збережена: {output_path}")


def main():
    """Тестування автоматичної конфігурації"""
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Шукати датасет у стандартних місцях
        project_root = Path(__file__).parent.parent
        possible_datasets = [
            project_root / "phi3_training_dataset.json",
            project_root / "datasets" / "train.json",
            Path("phi3_training_dataset.json"),
        ]
        
        dataset_path = None
        for path in possible_datasets:
            if path.exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            print("❌ Датасет не знайдено. Вкажіть шлях: python training_config.py <dataset_path>")
            return
    
    auto_config = AutoTrainingConfig(dataset_path)
    auto_config.print_summary()
    auto_config.save_config()


if __name__ == "__main__":
    main()





















