"""
Менеджер для роботи з навченими моделями
Автоматично знаходить та керує моделями з models/trained/
"""
from pathlib import Path
from typing import List, Dict, Optional
import json


class ModelManager:
    """Менеджер для навчених моделей"""
    
    def __init__(self, models_dir: str | Path = None):
        """
        Ініціалізація менеджера моделей
        
        Args:
            models_dir: Шлях до папки з моделями (None = автоматично знайти)
        """
        if models_dir is None:
            # Шукати в кількох місцях
            project_root = Path(__file__).parent.parent
            possible_dirs = [
                project_root / "models" / "trained",  # Нова структура
                project_root / "trained_models",       # Стара структура
                project_root.parent / "models",        # Зовнішня папка
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    models_dir = dir_path
                    break
            
            if models_dir is None:
                # Створити за замовчуванням
                models_dir = project_root / "models" / "trained"
                models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._models_cache = None
    
    def discover_models(self) -> List[Dict]:
        """
        Знайти всі навчені моделі в директорії
        
        Returns:
            Список словників з інформацією про моделі
        """
        models = []
        
        # Шукати .pt, .pth, .ckpt файли
        for ext in ['*.pt', '*.pth', '*.ckpt']:
            for model_file in self.models_dir.glob(ext):
                # Перевірити чи є конфігурація
                config_path = model_file.with_suffix('.json')
                if not config_path.exists():
                    # Спробувати знайти config з _config суфіксом
                    config_path = model_file.parent / f"{model_file.stem}_config.json"
                
                model_info = {
                    'path': str(model_file.absolute()),
                    'name': model_file.stem,
                    'filename': model_file.name,
                    'size_mb': model_file.stat().st_size / (1024 * 1024),
                    'modified': model_file.stat().st_mtime,
                    'config_path': str(config_path) if config_path.exists() else None
                }
                
                # Завантажити конфігурацію якщо є
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            model_info['config'] = config
                            # Витягнути тип backbone якщо є
                            if 'use_transformer' in config:
                                model_info['backbone_type'] = 'transformer' if config.get('use_transformer') else 'mlpmixer'
                            else:
                                model_info['backbone_type'] = 'unknown'
                    except Exception as e:
                        model_info['config'] = None
                        model_info['backbone_type'] = 'unknown'
                else:
                    model_info['config'] = None
                    model_info['backbone_type'] = 'unknown'
                
                models.append(model_info)
        
        # Сортувати за датою модифікації (новіші спочатку)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        self._models_cache = models
        return models
    
    def get_models(self, refresh: bool = False) -> List[Dict]:
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
    
    def get_default_model(self) -> Optional[Dict]:
        """
        Отримати модель за замовчуванням (остання)
        
        Returns:
            Інформація про модель або None
        """
        models = self.get_models()
        return models[0] if models else None
    
    def get_model_by_name(self, name: str) -> Optional[Dict]:
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
    
    def count_models(self) -> int:
        """Підрахувати кількість знайдених моделей"""
        return len(self.get_models())
    
    def list_models(self) -> None:
        """Вивести список всіх знайдених моделей"""
        models = self.get_models()
        
        if not models:
            print(f"❌ Моделі не знайдено в {self.models_dir}")
            print(f"   Додайте .pt файли в папку: {self.models_dir}")
            return
        
        print(f"\n📦 Знайдено {len(models)} моделей в {self.models_dir}:")
        print("-" * 70)
        
        for i, model in enumerate(models, 1):
            size_str = f"{model['size_mb']:.2f} MB"
            backbone = model.get('backbone_type', 'unknown')
            print(f"{i}. {model['name']}")
            print(f"   📁 {model['path']}")
            print(f"   📊 Розмір: {size_str}")
            print(f"   🔧 Backbone: {backbone}")
            if model.get('config'):
                print(f"   ⚙️  Конфігурація: {model['config_path']}")
            print()

