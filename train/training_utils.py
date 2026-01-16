"""
Утиліти для навчання
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import torch

from train.exceptions import DatasetNotFoundError
from train.constants import DEFAULT_SEQ_LEN
from data import DatasetManager
# DEPRECATED: Використовуйте runtime.resume замість config.training_resume
from runtime.resume import find_latest_checkpoint, get_checkpoint_info
# Backwards compatibility
try:
    from config.training_resume import TrainingResume
except ImportError:
    TrainingResume = None  # Fallback


def resolve_dataset_path(
    dataset_name_or_path: Optional[str],
    dataset_manager: DatasetManager
) -> Path:
    """
    Вирішити шлях до датасету
    
    Args:
        dataset_name_or_path: Ім'я датасету або шлях
        dataset_manager: Менеджер датасетів
    
    Returns:
        Шлях до датасету
    
    Raises:
        DatasetNotFoundError: Якщо датасет не знайдено
    """
    if dataset_name_or_path is None:
        datasets = dataset_manager.list_datasets()
        if not datasets:
            raise DatasetNotFoundError("", "Датасети не знайдено. Додайте датасет в папку datasets/")
        return Path(datasets[0]['path'])
    
    if Path(dataset_name_or_path).exists():
        return Path(dataset_name_or_path)
    
    dataset = dataset_manager.get_dataset(dataset_name_or_path)
    if not dataset:
        raise DatasetNotFoundError(dataset_name_or_path)
    return Path(dataset['path'])


def handle_duplicate_training(
    training_resume: TrainingResume,
    dataset_path: Path,
    model_config: Dict[str, Any],
    teacher_model_path: Optional[str],
    auto_resume: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Обробити перевірку дублювання навчання
    
    Returns:
        (is_cancelled, reason, previous_training_info)
    """
    is_duplicate, reason, previous_training = training_resume.check_duplicate_training(
        dataset_path,
        model_config,
        teacher_model_path=teacher_model_path
    )
    
    if is_duplicate:
        print(f"\n⚠️  УВАГА: {reason}")
        print("   Модель вже навчена на цьому датасеті з такою конфігурацією та teacher моделлю.")
        # Автоматично продовжувати якщо це не інтерактивний режим
        import sys
        if auto_resume or not sys.stdin.isatty():
            # Автоматичний режим - продовжувати завжди
            response = 'y'
            print("   Автоматичне продовження навчання (неінтерактивний режим)")
        else:
            response = input("   Продовжити навчання? (y/n): ").strip().lower()
        if response != 'y':
            print("   Навчання скасовано")
            return True, None, None
    
    return False, reason, previous_training


def handle_resume_checkpoint(
    training_resume: TrainingResume,
    previous_training: Optional[Dict],
    teacher_model_path: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    model_config: Optional[Dict] = None,
    auto_resume: bool = False
) -> Optional[str]:
    """
    Обробити логіку продовження з checkpoint
    
    Returns:
        Шлях до checkpoint'у або None
    """
    resume_from_checkpoint = None
    
    if previous_training:
        print(f"\n📚 ІНФОРМАЦІЯ: Знайдено попереднє навчання")
        print(f"   Попереднє навчання: teacher модель '{previous_training.get('teacher_model', 'невідомо')}'")
        print(f"   Поточне навчання: teacher модель '{Path(teacher_model_path).stem if teacher_model_path else 'невідомо'}'")
        print(f"   → Буде донавчання на нових даних від нової teacher моделі")
        
        if previous_training.get('checkpoint_path'):
            prev_checkpoint = Path(previous_training['checkpoint_path'])
            if prev_checkpoint.exists():
                print(f"   💡 Знайдено checkpoint від попереднього навчання")
                import sys
                if auto_resume or not sys.stdin.isatty():
                    response = 'y'
                    print("   Автоматичне використання попереднього checkpoint (неінтерактивний режим)")
                else:
                    response = input("   Використати попередній checkpoint як базу для донавчання? (y/n): ").strip().lower()
                if response == 'y':
                    resume_from_checkpoint = str(prev_checkpoint)
                    print(f"   ✅ Буде використано попередній checkpoint для донавчання")
    
    # Перевірити чи є checkpoint для продовження після екстреного переривання
    if resume_from_checkpoint is None:
        # Перевірити чи є незавершене навчання (екстрене переривання)
        should_resume, resume_checkpoint, checkpoint_info = training_resume.should_resume(
            dataset_path=dataset_path,
            model_config=model_config
        )
        
        if should_resume and resume_checkpoint and checkpoint_info:
            # Знайдено незавершене навчання
            is_final = checkpoint_info.get('is_final', False)
            epoch = checkpoint_info.get('epoch', 0)
            batch_count = checkpoint_info.get('batch_count', 0)
            total_epochs = checkpoint_info.get('epochs', 0)
            loss = checkpoint_info.get('loss', None)
            
            print(f"\n🔄 ЗНАЙДЕНО НЕЗАВЕРШЕНЕ НАВЧАННЯ (екстрене переривання):")
            print(f"   📍 Епоха: {epoch}/{total_epochs if total_epochs > 0 else '?'}")
            print(f"   📊 Батчів оброблено: {batch_count}")
            if loss is not None:
                print(f"   📉 Останній loss: {loss:.6f}")
            print(f"   💾 Checkpoint: {Path(resume_checkpoint).name}")
            print(f"   ✅ Автоматично продовжуємо навчання...")
            
            # Завжди автоматично продовжувати незавершене навчання
            resume_from_checkpoint = str(resume_checkpoint)
    
    return resume_from_checkpoint


def save_model_and_config(
    model: Any,  # TinyRecursiveModel (використовуємо Any через циклічні імпорти)
    model_save_path: Union[str, Path],
    model_config: Dict[str, Any],
    training_config: Any  # TrainingConfig (використовуємо Any через циклічні імпорти)
) -> None:
    """
    Зберегти модель та конфігурацію
    
    Args:
        model: Навчена модель
        model_save_path: Шлях для збереження моделі
        model_config: Конфігурація моделі
        training_config: Конфігурація навчання
    """
    import json
    
    # Конвертувати в Path якщо потрібно
    model_save_path = Path(model_save_path)
    
    # Створити директорію якщо не існує
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(model.state_dict(), model_save_path)
    except Exception as e:
        raise IOError(f"Не вдалося зберегти модель: {e}") from e
    
    config_path = Path(str(model_save_path).replace('.pt', '_config.json'))
    
    # Додати інформацію про тип backbone
    backbone_type = 'unknown'
    if hasattr(model, 'network'):
        if hasattr(model.network, '__class__'):
            if 'Transformer' in model.network.__class__.__name__:
                backbone_type = 'transformer'
            elif 'MLPMixer' in model.network.__class__.__name__:
                backbone_type = 'mlpmixer'
    
    full_config = {
        **model_config,
        'backbone_type': backbone_type,  # Додати тип backbone
        'training_config': training_config.to_dict() if hasattr(training_config, 'to_dict') else training_config
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Не вдалося зберегти конфігурацію: {e}") from e
    
    print(f"   ✅ Модель збережено: {model_save_path}")
    print(f"   ✅ Конфігурація збережена: {config_path}")


def get_final_loss_from_checkpoint(checkpoint_dir: str) -> Optional[float]:
    """
    Отримати останній loss з checkpoint'у
    
    Args:
        checkpoint_dir: Директорія з checkpoint'ами
    
    Returns:
        Loss або None
    """
    latest_checkpoint = Path(checkpoint_dir) / "checkpoint_latest.pt"
    if latest_checkpoint.exists():
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            return checkpoint.get('loss', checkpoint.get('final_loss', None))
        except Exception as e:
            # Логувати помилку, але не переривати виконання
            print(f"⚠️ Не вдалося завантажити loss з checkpoint: {e}")
    return None

