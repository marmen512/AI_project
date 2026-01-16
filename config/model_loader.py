"""
Модуль для роботи з GGUF моделями
Автоматично знаходить та керує GGUF моделями
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import json


class GGUFModelManager:
    """Менеджер для роботи з GGUF моделями"""
    
    def __init__(self, models_dir: str | Path = None):
        """
        Ініціалізація менеджера моделей
        
        Args:
            models_dir: Шлях до папки з GGUF моделями (None = автоматично знайти)
        """
        if models_dir is None:
            # Шукати в кількох місцях
            project_root = Path(__file__).parent.parent
            possible_dirs = [
                project_root / "models" / "gguf",  # Нова структура
                project_root / "models",            # Стара структура
                project_root.parent / "models",     # Зовнішня папка
                Path("models") / "gguf",
                Path("models"),
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists() and any(dir_path.glob("*.gguf")):
                    models_dir = dir_path
                    break
            
            if models_dir is None:
                # Створити за замовчуванням
                models_dir = project_root / "models" / "gguf"
                models_dir.mkdir(parents=True, exist_ok=True)
            
            # Також перевірити стару структуру для міграції
            old_dir = project_root / "models" / "gguf"
            if old_dir.exists():
                pass  # Вже знайдено
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._models_cache = None
    
    def discover_models(self) -> List[Dict[str, any]]:
        """
        Знайти всі GGUF моделі в директорії
        
        Returns:
            Список словників з інформацією про моделі
        """
        models = []
        
        # Шукати .gguf файли
        for gguf_file in self.models_dir.glob("*.gguf"):
            model_info = {
                'path': str(gguf_file.absolute()),
                'name': gguf_file.stem,
                'filename': gguf_file.name,
                'size_mb': gguf_file.stat().st_size / (1024 * 1024),
                'size_gb': gguf_file.stat().st_size / (1024 * 1024 * 1024),
            }
            models.append(model_info)
        
        # Також шукати в підпапках
        for subdir in self.models_dir.iterdir():
            if subdir.is_dir():
                for gguf_file in subdir.glob("*.gguf"):
                    model_info = {
                        'path': str(gguf_file.absolute()),
                        'name': gguf_file.stem,
                        'filename': gguf_file.name,
                        'size_mb': gguf_file.stat().st_size / (1024 * 1024),
                        'size_gb': gguf_file.stat().st_size / (1024 * 1024 * 1024),
                        'subdir': subdir.name,
                    }
                    models.append(model_info)
        
        self._models_cache = models
        return models
    
    def get_models(self, refresh: bool = False) -> List[Dict[str, any]]:
        """
        Отримати список моделей (з кешуванням)
        
        Args:
            refresh: Оновити кеш
        
        Returns:
            Список моделей
        """
        if self._models_cache is None or refresh:
            return self.discover_models()
        return self._models_cache
    
    def count_models(self) -> int:
        """Підрахувати кількість знайдених моделей"""
        return len(self.get_models())
    
    def get_model_by_name(self, name: str) -> Optional[Dict[str, any]]:
        """
        Знайти модель за ім'ям
        
        Args:
            name: Ім'я моделі (частина або повне)
        
        Returns:
            Інформація про модель або None
        """
        models = self.get_models()
        name_lower = name.lower()
        
        for model in models:
            if name_lower in model['name'].lower() or name_lower in model['filename'].lower():
                return model
        
        return None
    
    def get_default_model(self) -> Optional[Dict[str, any]]:
        """
        Отримати модель за замовчуванням (перша знайдена або найменша)
        
        Returns:
            Інформація про модель або None
        """
        models = self.get_models()
        if not models:
            return None
        
        # Пріоритет: phi-3, потім найменша модель
        for model in models:
            if 'phi' in model['name'].lower():
                return model
        
        # Повернути найменшу модель
        return min(models, key=lambda m: m['size_mb'])
    
    def list_models(self) -> None:
        """Вивести список всіх знайдених моделей"""
        models = self.get_models()
        
        if not models:
            print(f"❌ GGUF моделі не знайдено в {self.models_dir}")
            print(f"   Додайте .gguf файли в папку: {self.models_dir}")
            return
        
        print(f"\n📦 Знайдено {len(models)} GGUF моделей в {self.models_dir}:")
        print("-" * 70)
        
        for i, model in enumerate(models, 1):
            size_str = f"{model['size_gb']:.2f} GB" if model['size_gb'] >= 1 else f"{model['size_mb']:.1f} MB"
            print(f"{i}. {model['name']}")
            print(f"   📁 {model['path']}")
            print(f"   📊 Розмір: {size_str}")
            if 'subdir' in model:
                print(f"   📂 Підпапка: {model['subdir']}")
            print()
    
    def save_models_info(self, output_file: str = "models_info.json") -> None:
        """Зберегти інформацію про моделі в JSON"""
        models = self.get_models()
        info = {
            'models_dir': str(self.models_dir),
            'count': len(models),
            'models': models
        }
        
        # Створити папку temp якщо не існує
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = temp_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Інформація про моделі збережена: {output_path}")


def main():
    """Тестування менеджера моделей"""
    manager = GGUFModelManager()
    manager.list_models()
    
    if manager.count_models() > 0:
        default = manager.get_default_model()
        if default:
            print(f"\n🎯 Модель за замовчуванням: {default['name']}")
    
    manager.save_models_info("models_info.json")


if __name__ == "__main__":
    main()

