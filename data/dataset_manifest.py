"""
Dataset split manifest для контролю contamination
Жорстке розділення pretrain/instruction/eval режимів
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from data.dataset_fingerprint import DatasetFingerprint, compute_file_fingerprint


@dataclass
class DatasetSplit:
    """Інформація про split датасету"""
    name: str  # 'pretrain', 'instruction', 'eval'
    path: str
    fingerprint: str
    num_samples: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetManifest:
    """
    Manifest для контролю dataset splits та contamination
    """
    
    def __init__(self, manifest_path: Optional[Path] = None):
        """
        Ініціалізація
        
        Args:
            manifest_path: Шлях до manifest файлу
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.splits: Dict[str, DatasetSplit] = {}
        self._loaded_doc_ids: Set[str] = set()  # Для перевірки на leakage
    
    def add_split(
        self,
        name: str,
        path: Path,
        fingerprint: Optional[str] = None,
        num_samples: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Додати split до manifest
        
        Args:
            name: Назва split ('pretrain', 'instruction', 'eval')
            path: Шлях до файлу split
            fingerprint: Fingerprint файлу (якщо None, обчислиться автоматично)
            num_samples: Кількість семплів (опціонально)
            metadata: Додаткові метадані
        """
        if fingerprint is None:
            fingerprint = compute_file_fingerprint(path)
        
        split = DatasetSplit(
            name=name,
            path=str(path),
            fingerprint=fingerprint,
            num_samples=num_samples or 0,
            metadata=metadata or {}
        )
        
        self.splits[name] = split
    
    def verify_split(self, name: str, path: Path) -> bool:
        """
        Перевірити чи split відповідає manifest
        
        Args:
            name: Назва split
            path: Шлях до файлу
        
        Returns:
            True якщо split відповідає manifest
        """
        if name not in self.splits:
            return False
        
        split = self.splits[name]
        expected_fingerprint = split.fingerprint
        
        return DatasetFingerprint(path).verify(expected_fingerprint)
    
    def check_contamination(
        self,
        split_name: str,
        doc_ids: List[str]
    ) -> Dict[str, any]:
        """
        Перевірити на contamination (eval leakage)
        
        Args:
            split_name: Назва split для перевірки
            doc_ids: Список doc_id для перевірки
        
        Returns:
            Dict з результатами перевірки
        """
        result = {
            'is_clean': True,
            'contaminated_docs': [],
            'contamination_source': None
        }
        
        # Eval split не повинен містити doc_id з pretrain/instruction
        if split_name == 'eval':
            # Перевірити чи є doc_id які вже були в інших splits
            contaminated = [doc_id for doc_id in doc_ids if doc_id in self._loaded_doc_ids]
            if contaminated:
                result['is_clean'] = False
                result['contaminated_docs'] = contaminated
                result['contamination_source'] = 'pretrain/instruction leakage'
        
        # Додати doc_ids до набору завантажених
        self._loaded_doc_ids.update(doc_ids)
        
        return result
    
    def save(self, output_path: Optional[Path] = None):
        """Зберегти manifest в файл"""
        if output_path is None:
            output_path = self.manifest_path
        
        if output_path is None:
            raise ValueError("output_path повинен бути вказаний")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest_data = {
            'splits': {name: asdict(split) for name, split in self.splits.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    @classmethod
    def load(cls, manifest_path: Path) -> 'DatasetManifest':
        """Завантажити manifest з файлу"""
        manifest = cls(manifest_path=manifest_path)
        
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        for name, split_data in data.get('splits', {}).items():
            split = DatasetSplit(**split_data)
            manifest.splits[name] = split
        
        return manifest
    
    def get_split_info(self, name: str) -> Optional[DatasetSplit]:
        """Отримати інформацію про split"""
        return self.splits.get(name)
    
    def list_splits(self) -> List[str]:
        """Отримати список всіх splits"""
        return list(self.splits.keys())

