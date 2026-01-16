"""
Модуль для керування датасетами
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Any


class DatasetManager:
    """Менеджер для роботи з датасетами"""
    
    def __init__(self, datasets_dir: str | Path = None):
        """
        Ініціалізація менеджера датасетів
        
        Args:
            datasets_dir: Шлях до папки з датасетами (None = автоматично)
        """
        if datasets_dir is None:
            project_root = Path(__file__).parent.parent
            # Шукати в кількох місцях
            possible_dirs = [
                project_root / "temp" / "datasets",
                project_root / "datasets",
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    datasets_dir = dir_path
                    break
            
            if datasets_dir is None:
                datasets_dir = project_root / "temp" / "datasets"
                datasets_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Знайти всі датасети
        
        Returns:
            Список інформації про датасети
        """
        datasets = []
        
        for json_file in self.datasets_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Підтримка метаданих
                if isinstance(data, dict) and 'data' in data:
                    samples = len(data['data'])
                    metadata = data.get('metadata', {})
                elif isinstance(data, dict) and 'metadata' in data:
                    samples = len(data.get('data', []))
                    metadata = data.get('metadata', {})
                elif isinstance(data, list):
                    samples = len(data)
                    metadata = {}
                else:
                    samples = 0
                    metadata = {}
                
                dataset_info = {
                    'path': str(json_file.absolute()),
                    'name': json_file.stem,
                    'filename': json_file.name,
                    'size_mb': json_file.stat().st_size / (1024 * 1024),
                    'samples': samples,
                    'modified': json_file.stat().st_mtime,
                    'teacher_model': metadata.get('teacher_model_name') if metadata else None
                }
                datasets.append(dataset_info)
            except Exception as e:
                # Пропустити файли які не є валідними JSON
                continue
        
        # Сортувати за датою (новіші спочатку)
        datasets.sort(key=lambda x: x['modified'], reverse=True)
        
        return datasets
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Отримати датасет за ім'ям
        
        Args:
            name: Ім'я датасету (частина або повне)
        
        Returns:
            Інформація про датасет або None
        """
        datasets = self.list_datasets()
        name_lower = name.lower()
        
        for dataset in datasets:
            if name_lower in dataset['name'].lower():
                return dataset
        
        return None
    
    def load_dataset(self, name_or_path: str) -> List[Dict[str, str]]:
        """
        Завантажити датасет
        
        Args:
            name_or_path: Ім'я датасету або шлях до файлу
        
        Returns:
            Список прикладів
        """
        # Перевірити чи це шлях
        if Path(name_or_path).exists():
            dataset_path = Path(name_or_path)
        else:
            dataset = self.get_dataset(name_or_path)
            if not dataset:
                raise FileNotFoundError(f"Датасет не знайдено: {name_or_path}")
            dataset_path = Path(dataset['path'])
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Підтримка метаданих
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            return []
    
    def save_dataset(self, data: List[Dict[str, str]], name: str) -> Path:
        """
        Зберегти датасет
        
        Args:
            data: Список прикладів
            name: Ім'я датасету (без .json)
        
        Returns:
            Шлях до збереженого файлу
        """
        if not name.endswith('.json'):
            name += '.json'
        
        output_path = self.datasets_dir / name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def print_summary(self) -> None:
        """Вивести підсумок всіх датасетів"""
        datasets = self.list_datasets()
        
        if not datasets:
            print(f"❌ Датасети не знайдено в {self.datasets_dir}")
            print(f"   Додайте .json файли в папку: {self.datasets_dir}")
            return
        
        print(f"\n📚 Знайдено {len(datasets)} датасетів в {self.datasets_dir}:")
        print("-" * 70)
        
        total_samples = 0
        for i, dataset in enumerate(datasets, 1):
                print(f"{i}. {dataset['name']}")
                print(f"   📁 {dataset['filename']}")
                print(f"   📊 Розмір: {dataset['size_mb']:.2f} MB")
                print(f"   📝 Прикладів: {dataset['samples']:,}")
                if dataset.get('teacher_model'):
                    print(f"   🎓 Teacher модель: {dataset['teacher_model']}")
                print()
                total_samples += dataset['samples']
        
        print(f"📊 Загалом: {total_samples:,} прикладів")
        print()


def main():
    """CLI для керування датасетами"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Керування датасетами")
    parser.add_argument("--list", action="store_true", help="Список датасетів")
    parser.add_argument("--info", type=str, help="Інформація про конкретний датасет")
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.info:
        dataset = manager.get_dataset(args.info)
        if dataset:
            print(f"\n📊 Інформація про датасет '{dataset['name']}':")
            print(f"   Шлях: {dataset['path']}")
            print(f"   Розмір: {dataset['size_mb']:.2f} MB")
            print(f"   Прикладів: {dataset['samples']:,}")
        else:
            print(f"❌ Датасет не знайдено: {args.info}")
    else:
        manager.print_summary()


if __name__ == "__main__":
    main()

