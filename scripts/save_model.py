"""
Скрипт для збереження моделі, чекпоінтів та датасету в окрему папку
для подальшого донавчання
"""
import sys
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def get_model_name_from_dataset(dataset_path: Path) -> str:
    """Визначити назву моделі з назви датасету"""
    return dataset_path.stem  # Без розширення


def cleanup_training_files(project_root: Path, keep_model: Optional[Path] = None):
    """
    Очистити файли навчання
    
    Args:
        project_root: Корінь проекту
        keep_model: Модель, яку не потрібно видаляти (якщо вона в models/trained/)
    """
    cleaned = []
    
    # Очистити checkpoints/
    checkpoints_dir = project_root / "checkpoints"
    if checkpoints_dir.exists():
        try:
            shutil.rmtree(checkpoints_dir)
            checkpoints_dir.mkdir(exist_ok=True)
            cleaned.append("checkpoints/")
        except Exception as e:
            print(f"⚠️  Помилка очищення checkpoints/: {e}")
    
    # Очистити logs/
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        try:
            # Видалити всі .log та .json файли, але зберегти структуру
            for log_file in logs_dir.glob("*.log"):
                log_file.unlink()
            for json_file in logs_dir.glob("*.json"):
                json_file.unlink()
            # Видалити символічне посилання якщо воно є
            latest_link = logs_dir / "training_latest.log"
            if latest_link.exists() and latest_link.is_symlink():
                latest_link.unlink()
            cleaned.append("logs/")
        except Exception as e:
            print(f"⚠️  Помилка очищення logs/: {e}")
    
    # Очистити temp/
    temp_dir = project_root / "temp"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            cleaned.append("temp/")
        except Exception as e:
            print(f"⚠️  Помилка очищення temp/: {e}")
    
    # Очистити models/trained/ (крім keep_model)
    models_trained_dir = project_root / "models" / "trained"
    if models_trained_dir.exists():
        try:
            for model_file in models_trained_dir.glob("*.pt"):
                if keep_model and model_file.resolve() == keep_model.resolve():
                    continue  # Пропустити збережену модель
                model_file.unlink()
            # Видалити config файли
            for config_file in models_trained_dir.glob("*_config.json"):
                config_file.unlink()
            cleaned.append("models/trained/")
        except Exception as e:
            print(f"⚠️  Помилка очищення models/trained/: {e}")
    
    if cleaned:
        print(f"\n✅ Очищено: {', '.join(cleaned)}")
    else:
        print("\n⚠️  Немає файлів для очищення")


def find_latest_model(models_dir: Path = None) -> Optional[Path]:
    """Знайти останню навчену модель"""
    if models_dir is None:
        models_dir = project_root / "models" / "trained"
    
    if not models_dir.exists():
        return None
    
    # Шукати .pt файли
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        return None
    
    # Повернути найновіший
    return max(model_files, key=lambda p: p.stat().st_mtime)


def find_latest_checkpoint(checkpoints_dir: Path = None) -> Optional[Path]:
    """Знайти останній checkpoint"""
    if checkpoints_dir is None:
        checkpoints_dir = project_root / "checkpoints"
    
    if not checkpoints_dir.exists():
        return None
    
    checkpoint_file = checkpoints_dir / "checkpoint_latest.pt"
    if checkpoint_file.exists():
        return checkpoint_file
    
    # Шукати інші checkpoint'и
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        return None
    
    return max(checkpoint_files, key=lambda p: p.stat().st_mtime)


def load_model_config(model_path: Path) -> Optional[dict]:
    """Завантажити конфігурацію моделі"""
    # Спробувати знайти config файл
    config_path = model_path.parent / f"{model_path.stem}_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Спробувати завантажити з checkpoint (якщо він в тій же директорії що і модель)
    # Або з глобального checkpoints/
    checkpoint_dirs = [
        model_path.parent.parent / "checkpoints",  # Якщо модель в models/trained/
        project_root / "checkpoints"  # Глобальний checkpoints/
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint = find_latest_checkpoint(checkpoint_dir)
        if checkpoint and checkpoint.exists():
            try:
                data = torch.load(checkpoint, map_location='cpu')
                if 'config' in data:
                    return data['config']
            except:
                pass
    
    return None


def copy_checkpoints(source_dir: Path, dest_dir: Path):
    """Скопіювати всі чекпоінти"""
    if not source_dir.exists():
        print(f"⚠️  Директорія чекпоінтів не знайдена: {source_dir}")
        return
    
    dest_checkpoints = dest_dir / "checkpoints"
    dest_checkpoints.mkdir(parents=True, exist_ok=True)
    
    checkpoint_files = list(source_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        print("⚠️  Чекпоінти не знайдені")
        return
    
    copied = 0
    for checkpoint in checkpoint_files:
        try:
            shutil.copy2(checkpoint, dest_checkpoints / checkpoint.name)
            copied += 1
        except Exception as e:
            print(f"⚠️  Помилка копіювання {checkpoint.name}: {e}")
    
    print(f"✅ Скопійовано {copied} чекпоінтів")


def copy_dataset(dataset_path: Path, dest_dir: Path):
    """Скопіювати датасет"""
    if not dataset_path.exists():
        print(f"⚠️  Датасет не знайдено: {dataset_path}")
        return
    
    dest_dataset_dir = dest_dir / "dataset"
    dest_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if dataset_path.is_file():
            shutil.copy2(dataset_path, dest_dataset_dir / dataset_path.name)
        else:
            # Якщо це директорія, скопіювати всю
            shutil.copytree(dataset_path, dest_dataset_dir / dataset_path.name, dirs_exist_ok=True)
        
        print(f"✅ Датасет скопійовано: {dataset_path.name}")
    except Exception as e:
        print(f"⚠️  Помилка копіювання датасету: {e}")


def create_model_readme(model_dir: Path, model_name: str, model_path: Optional[Path], 
                       dataset_path: Optional[Path], config: Optional[dict]):
    """Створити README з інструкціями для донавчання"""
    readme_path = model_dir / "README.md"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# Модель: {model_name}\n\n")
        f.write(f"**Дата створення:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Структура\n\n")
        f.write("```\n")
        f.write(f"{model_name}/\n")
        if model_path:
            f.write("├── model.pt                    # Модель\n")
        f.write("├── model_config.json           # Конфігурація моделі\n")
        f.write("├── training_config.json        # Конфігурація навчання\n")
        f.write("├── checkpoints/                # Чекпоінти для продовження\n")
        if dataset_path:
            f.write("├── dataset/                    # Датасет для навчання\n")
        f.write("└── README.md                   # Цей файл\n")
        f.write("```\n\n")
        
        f.write("## Донавчання моделі\n\n")
        f.write("### 1. Продовження навчання з чекпоінту\n\n")
        f.write("```bash\n")
        f.write("python scripts/train_model.py \\\n")
        if dataset_path:
            f.write(f"    --dataset {model_dir}/dataset/{dataset_path.name} \\\n")
        f.write(f"    --resume {model_dir}/checkpoints/checkpoint_latest.pt \\\n")
        f.write("    --checkpoint-dir checkpoints \\\n")
        f.write("    --checkpoint-interval 100\n")
        f.write("```\n\n")
        
        f.write("### 2. Донавчання на новому датасеті\n\n")
        f.write("**Важливо:** Для донавчання на новому датасеті потрібно спочатку завантажити ваги моделі.\n\n")
        f.write("**Варіант A: Використати checkpoint як базу**\n")
        f.write("```bash\n")
        f.write("python scripts/train_model.py \\\n")
        f.write("    --dataset path/to/new_dataset.json \\\n")
        f.write(f"    --resume {model_dir}/checkpoints/checkpoint_latest.pt \\\n")
        f.write("    --checkpoint-dir checkpoints \\\n")
        f.write("    --checkpoint-interval 100\n")
        f.write("```\n\n")
        if model_path:
            f.write("**Варіант B: Завантажити модель через Python**\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write("from train.model_factory import create_model\n")
            f.write("from config.trm_config import TRMConfig\n\n")
            f.write("# Завантажити конфігурацію\n")
            f.write("import json\n")
            f.write("with open('model_config.json', 'r') as f:\n")
            f.write("    model_config = json.load(f)\n\n")
            f.write("# Створити модель\n")
            f.write("model = create_model(**model_config)\n\n")
            f.write("# Завантажити ваги\n")
            f.write("model.load_state_dict(torch.load('model.pt', map_location='cpu'))\n")
            f.write("```\n\n")
            f.write("Потім використайте цю модель для навчання на новому датасеті.\n\n")
        
        if config:
            f.write("## Конфігурація\n\n")
            f.write("### Параметри моделі\n\n")
            if 'dim' in config:
                f.write(f"- **dim:** {config.get('dim')}\n")
            if 'depth' in config:
                f.write(f"- **depth:** {config.get('depth')}\n")
            if 'seq_len' in config:
                f.write(f"- **seq_len:** {config.get('seq_len')}\n")
            if 'vocab_size' in config:
                f.write(f"- **vocab_size:** {config.get('vocab_size')}\n")
            
            f.write("\n### Параметри навчання\n\n")
            training_config = config.get('training_config', {})
            if isinstance(training_config, dict):
                if 'batch_size' in training_config:
                    f.write(f"- **batch_size:** {training_config.get('batch_size')}\n")
                if 'learning_rate' in training_config:
                    f.write(f"- **learning_rate:** {training_config.get('learning_rate')}\n")
                if 'epochs' in training_config:
                    f.write(f"- **epochs:** {training_config.get('epochs')}\n")
        
        f.write("\n## Примітки\n\n")
        f.write("- Всі чекпоінти збережені в `checkpoints/` для можливості продовження навчання\n")
        if dataset_path:
            f.write(f"- Датасет збережено в `dataset/` для відтворення навчання\n")
        f.write("- Конфігурація моделі та навчання збережена в JSON файлах\n")
        f.write("- Для донавчання використовуйте `--resume` з шляхом до checkpoint'у\n")
    
    print(f"✅ README створено: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Зберегти модель, чекпоінти та датасет в окрему папку для донавчання"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Назва моделі (буде використано як назва папки). Якщо не вказано, буде використано назву датасету"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Шлях до моделі (.pt файл). Якщо не вказано, буде знайдено останню модель"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Директорія з чекпоінтами (за замовчуванням: checkpoints/)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Шлях до датасету для збереження"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Директорія для збереження моделей (за замовчуванням: saved_models/)"
    )
    
    args = parser.parse_args()
    
    # Визначити назву моделі
    model_name = args.model_name
    if model_name is None:
        # Спробувати визначити з датасету
        if args.dataset_path:
            dataset_path = Path(args.dataset_path)
            if not dataset_path.is_absolute():
                dataset_path = project_root / dataset_path
            if dataset_path.exists():
                model_name = get_model_name_from_dataset(dataset_path)
                print(f"📝 Назва моделі визначена з датасету: {model_name}")
            else:
                print("❌ Помилка: --model-name не вказано і датасет не знайдено")
                print("   Вкажіть --model-name або --dataset-path")
                return
        else:
            print("❌ Помилка: --model-name не вказано")
            print("   Вкажіть --model-name або --dataset-path для автоматичного визначення")
            return
    
    # Створити директорію для збереження
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_dir = output_dir / model_name
    if model_dir.exists():
        response = input(f"⚠️  Папка {model_dir} вже існує. Перезаписати? (y/N): ")
        if response.lower() != 'y':
            print("Скасовано")
            return
        shutil.rmtree(model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Збереження моделі: {model_name}")
    print(f"📁 Директорія: {model_dir}\n")
    
    # Знайти або скопіювати модель
    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    else:
        model_path = find_latest_model()
    
    if model_path and model_path.exists():
        try:
            shutil.copy2(model_path, model_dir / "model.pt")
            print(f"✅ Модель скопійовано: {model_path.name}")
        except Exception as e:
            print(f"⚠️  Помилка копіювання моделі: {e}")
            model_path = None
    else:
        print("⚠️  Модель не знайдена, буде збережено тільки чекпоінти та датасет")
    
    # Завантажити конфігурацію
    config = None
    if model_path:
        config = load_model_config(model_path)
        if config:
            # Зберегти конфігурацію
            model_config = {k: v for k, v in config.items() if k != 'training_config'}
            training_config = config.get('training_config', {})
            
            with open(model_dir / "model_config.json", 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            
            with open(model_dir / "training_config.json", 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
            
            print("✅ Конфігурація збережена")
    
    # Скопіювати чекпоінти
    checkpoint_dir = project_root / (args.checkpoint_dir or "checkpoints")
    copy_checkpoints(checkpoint_dir, model_dir)
    
    # Скопіювати датасет
    dataset_path = None
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        copy_dataset(dataset_path, model_dir)
    
    # Створити README
    create_model_readme(model_dir, model_name, model_path, dataset_path, config)
    
    print(f"\n✅ Модель збережено успішно!")
    print(f"📁 Шлях: {model_dir}")
    print(f"\n💡 Для донавчання дивіться інструкції в {model_dir}/README.md")
    
    # Запитати про очищення файлів навчання
    print("\n" + "=" * 80)
    response = input("🧹 Очистити файли навчання (checkpoints/, logs/, temp/, models/trained/)? (y/N): ")
    if response.lower() == 'y':
        keep_model = None
        if model_path and model_path.exists():
            # Зберегти модель якщо вона в models/trained/
            if model_path.parent.name == "trained":
                keep_model = model_path
        cleanup_training_files(project_root, keep_model=keep_model)
    else:
        print("   Файли навчання залишено без змін")


if __name__ == "__main__":
    main()

