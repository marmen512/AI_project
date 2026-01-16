"""
Hash-based контроль dataset файлів для запобігання contamination
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List


def compute_file_fingerprint(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Обчислити fingerprint файлу
    
    Args:
        file_path: Шлях до файлу
        algorithm: Алгоритм хешування ('sha256', 'md5')
    
    Returns:
        Hex digest fingerprint
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        # Читати файл частинами для ефективності
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_data_fingerprint(data: List[Dict] | Dict, algorithm: str = 'sha256') -> str:
    """
    Обчислити fingerprint даних (JSON структури)
    
    Args:
        data: Дані (список або словник)
        algorithm: Алгоритм хешування
    
    Returns:
        Hex digest fingerprint
    """
    hash_func = getattr(hashlib, algorithm)()
    
    # Конвертувати в JSON string для хешування
    # Використовуємо sort_keys для детерміністичності
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    hash_func.update(json_str.encode('utf-8'))
    
    return hash_func.hexdigest()


def verify_fingerprint(file_path: Path, expected_fingerprint: str, algorithm: str = 'sha256') -> bool:
    """
    Перевірити fingerprint файлу
    
    Args:
        file_path: Шлях до файлу
        expected_fingerprint: Очікуваний fingerprint
        algorithm: Алгоритм хешування
    
    Returns:
        True якщо fingerprint збігається
    """
    if not file_path.exists():
        return False
    
    actual_fingerprint = compute_file_fingerprint(file_path, algorithm)
    return actual_fingerprint == expected_fingerprint


class DatasetFingerprint:
    """
    Клас для роботи з fingerprints датасетів
    """
    
    def __init__(self, file_path: Path, algorithm: str = 'sha256'):
        """
        Ініціалізація
        
        Args:
            file_path: Шлях до файлу датасету
            algorithm: Алгоритм хешування
        """
        self.file_path = Path(file_path)
        self.algorithm = algorithm
        self._fingerprint: Optional[str] = None
    
    def compute(self) -> str:
        """Обчислити fingerprint"""
        if self._fingerprint is None:
            self._fingerprint = compute_file_fingerprint(self.file_path, self.algorithm)
        return self._fingerprint
    
    def verify(self, expected: str) -> bool:
        """Перевірити fingerprint"""
        return verify_fingerprint(self.file_path, expected, self.algorithm)
    
    def save(self, output_path: Path):
        """Зберегти fingerprint в файл"""
        fingerprint = self.compute()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'file': str(self.file_path),
                'fingerprint': fingerprint,
                'algorithm': self.algorithm
            }, f, indent=2)
    
    @classmethod
    def load(cls, fingerprint_path: Path) -> Dict:
        """Завантажити fingerprint з файлу"""
        with open(fingerprint_path, 'r') as f:
            return json.load(f)

