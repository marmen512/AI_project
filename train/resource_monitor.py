"""
Модуль для моніторингу ресурсів під час навчання
Відстежує CPU, пам'ять, GPU та виявляє аномалії
"""
import time
import psutil
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

# Налаштувати logger
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Монітор ресурсів системи"""
    
    def __init__(
        self, 
        log_dir: Path = None, 
        log_interval: int = 10,
        cpu_warning_threshold: float = 95.0,
        memory_warning_threshold: float = 90.0,
        gpu_memory_warning_threshold: float = 90.0,
        slow_batch_threshold: float = 300.0
    ):
        """
        Ініціалізація монітора
        
        Args:
            log_dir: Папка для логів (за замовчуванням: logs/)
            log_interval: Логувати кожні N батчів
            cpu_warning_threshold: Поріг попередження для CPU (%)
            memory_warning_threshold: Поріг попередження для пам'яті (%)
            gpu_memory_warning_threshold: Поріг попередження для GPU пам'яті (%)
            slow_batch_threshold: Поріг для виявлення повільних батчів (секунди)
        """
        if log_dir is None:
            log_dir = Path("logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_interval = log_interval
        self.log_file = self.log_dir / f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Статистика
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.batch_times = []
        
        # Пороги для попереджень (з параметрів або config)
        self.cpu_warning_threshold = cpu_warning_threshold
        self.memory_warning_threshold = memory_warning_threshold
        self.gpu_memory_warning_threshold = gpu_memory_warning_threshold
        self.slow_batch_threshold = slow_batch_threshold
        
        # Ініціалізувати лог-файл
        self._init_log_file()
    
    def _init_log_file(self):
        """Ініціалізувати лог-файл"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Resource Monitor Log - Started at {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
    
    def _log(self, message: str):
        """Записати повідомлення в лог"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def get_cpu_usage(self) -> float:
        """Отримати використання CPU (%)"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Отримати використання пам'яті"""
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / (1024 ** 3),
            'available_gb': mem.available / (1024 ** 3),
            'total_gb': mem.total / (1024 ** 3)
        }
    
    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Отримати використання GPU (якщо доступно)"""
        if not torch.cuda.is_available():
            return None
        
        try:
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            
            return {
                'memory_allocated_gb': gpu_memory,
                'memory_reserved_gb': gpu_memory_reserved,
                'memory_total_gb': gpu_memory_total,
                'memory_percent': (gpu_memory_reserved / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
            }
        except Exception as e:
            self._log(f"⚠️ Помилка отримання інформації про GPU: {e}")
            return None
    
    def check_resources(self, batch_idx: int, epoch: int, batch_time: Optional[float] = None) -> Dict:
        """
        Перевірити ресурси та залогувати
        
        Args:
            batch_idx: Індекс батча
            epoch: Номер епохи
            batch_time: Час обробки батча (секунди)
        
        Returns:
            Словник з інформацією про ресурси та попередженнями
        """
        cpu_usage = self.get_cpu_usage()
        memory_info = self.get_memory_usage()
        gpu_info = self.get_gpu_usage()
        
        # Зберегти зразки
        self.cpu_samples.append(cpu_usage)
        self.memory_samples.append(memory_info['percent'])
        if gpu_info:
            self.gpu_samples.append(gpu_info['memory_percent'])
        if batch_time:
            self.batch_times.append(batch_time)
        
        # Перевірити попередження
        warnings = []
        
        if cpu_usage > self.cpu_warning_threshold:
            warning_msg = f"Високе використання CPU: {cpu_usage:.1f}% (Епоха: {epoch}, Батч: {batch_idx})"
            warnings.append(f"Високе використання CPU: {cpu_usage:.1f}%")
            logger.warning(warning_msg)
            self._log(f"⚠️ {warning_msg}")
        
        if memory_info['percent'] > self.memory_warning_threshold:
            warning_msg = f"Високе використання пам'яті: {memory_info['percent']:.1f}% ({memory_info['used_gb']:.2f} GB / {memory_info['total_gb']:.2f} GB) (Епоха: {epoch}, Батч: {batch_idx})"
            warnings.append(f"Високе використання пам'яті: {memory_info['percent']:.1f}% ({memory_info['used_gb']:.2f} GB / {memory_info['total_gb']:.2f} GB)")
            logger.warning(warning_msg)
            self._log(f"⚠️ {warning_msg}")
        
        if gpu_info and gpu_info['memory_percent'] > self.gpu_memory_warning_threshold:
            warning_msg = f"Високе використання GPU пам'яті: {gpu_info['memory_percent']:.1f}% ({gpu_info['memory_reserved_gb']:.2f} GB / {gpu_info['memory_total_gb']:.2f} GB) (Епоха: {epoch}, Батч: {batch_idx})"
            warnings.append(f"Високе використання GPU пам'яті: {gpu_info['memory_percent']:.1f}% ({gpu_info['memory_reserved_gb']:.2f} GB / {gpu_info['memory_total_gb']:.2f} GB)")
            logger.warning(warning_msg)
            self._log(f"⚠️ {warning_msg}")
        
        if batch_time and batch_time > self.slow_batch_threshold:
            warning_msg = f"Повільний батч: {batch_time:.1f} секунд (Епоха: {epoch}, Батч: {batch_idx})"
            warnings.append(f"Повільний батч: {batch_time:.1f} секунд")
            logger.warning(warning_msg)
            self._log(f"⚠️ {warning_msg}")
        
        # Логувати ресурси кожні N батчів
        if batch_idx % self.log_interval == 0:
            log_msg = f"📊 Ресурси (Епоха: {epoch}, Батч: {batch_idx}): CPU: {cpu_usage:.1f}%, RAM: {memory_info['percent']:.1f}% ({memory_info['used_gb']:.2f} GB)"
            if gpu_info:
                log_msg += f", GPU: {gpu_info['memory_percent']:.1f}% ({gpu_info['memory_reserved_gb']:.2f} GB)"
            if batch_time:
                log_msg += f", Час батча: {batch_time:.1f}s"
            self._log(log_msg)
        
        return {
            'cpu_usage': cpu_usage,
            'memory': memory_info,
            'gpu': gpu_info,
            'batch_time': batch_time,
            'warnings': warnings
        }
    
    def should_throttle(self, cpu_usage: Optional[float] = None, memory_info: Optional[Dict] = None) -> bool:
        """
        Чи потрібно throttle (знизити навантаження)
        
        Args:
            cpu_usage: Використання CPU (%) (якщо None, обчислиться)
            memory_info: Інформація про пам'ять (якщо None, обчислиться)
        
        Returns:
            True якщо потрібно throttle
        """
        if cpu_usage is None:
            cpu_usage = self.get_cpu_usage()
        if memory_info is None:
            memory_info = self.get_memory_usage()
        
        # Throttle якщо CPU або пам'ять занадто високі
        if cpu_usage > self.cpu_warning_threshold:
            return True
        if memory_info['percent'] > self.memory_warning_threshold:
            return True
        
        return False
    
    def should_shrink_batch(self, memory_info: Optional[Dict] = None) -> bool:
        """
        Чи потрібно зменшити batch size
        
        Args:
            memory_info: Інформація про пам'ять (якщо None, обчислиться)
        
        Returns:
            True якщо потрібно shrink batch
        """
        if memory_info is None:
            memory_info = self.get_memory_usage()
        
        # Shrink batch якщо пам'ять дуже висока (> 90%)
        shrink_threshold = 90.0
        if memory_info['percent'] > shrink_threshold:
            return True
        
        return False
    
    def should_pause(self, cpu_usage: Optional[float] = None, memory_info: Optional[Dict] = None) -> bool:
        """
        Чи потрібно призупинити навчання
        
        Args:
            cpu_usage: Використання CPU (%) (якщо None, обчислиться)
            memory_info: Інформація про пам'ять (якщо None, обчислиться)
        
        Returns:
            True якщо потрібно pause
        """
        if cpu_usage is None:
            cpu_usage = self.get_cpu_usage()
        if memory_info is None:
            memory_info = self.get_memory_usage()
        
        # Pause якщо пам'ять критично висока (> 95%) або система перевантажена
        pause_threshold_memory = 95.0
        if memory_info['percent'] > pause_threshold_memory:
            return True
        
        # Pause якщо CPU постійно > 98%
        if cpu_usage > 98.0:
            return True
        
        return False
    
    def get_throttle_recommendations(self) -> Dict[str, any]:
        """
        Отримати рекомендації по throttle
        
        Returns:
            Dict з рекомендаціями: throttle, shrink_batch, pause, suggested_batch_size
        """
        cpu_usage = self.get_cpu_usage()
        memory_info = self.get_memory_usage()
        gpu_info = self.get_gpu_usage()
        
        should_throttle = self.should_throttle(cpu_usage, memory_info)
        should_shrink = self.should_shrink_batch(memory_info)
        should_pause = self.should_pause(cpu_usage, memory_info)
        
        return {
            'throttle': should_throttle,
            'shrink_batch': should_shrink,
            'pause': should_pause,
            'cpu_usage': cpu_usage,
            'memory_percent': memory_info['percent'],
            'gpu_memory_percent': gpu_info['memory_percent'] if gpu_info else None
        }
    
    def get_suggested_batch_size(self, current_batch_size: int) -> int:
        """
        Отримати рекомендований batch size на основі поточного використання ресурсів.
        Викликається при OOM або високому використанні пам'яті.
        
        Args:
            current_batch_size: Поточний batch size
            
        Returns:
            Рекомендований batch size (завжди >= 1)
        """
        memory_info = self.get_memory_usage()
        gpu_info = self.get_gpu_usage()
        
        # Використовувати GPU пам'ять якщо доступна, інакше системну пам'ять
        memory_percent = gpu_info['memory_percent'] if gpu_info else memory_info['percent']
        
        # Зменшити batch size в залежності від використання пам'яті
        if memory_percent > 95.0:
            # Критично високе використання - зменшити на 75%
            suggested = max(1, int(current_batch_size * 0.25))
        elif memory_percent > 90.0:
            # Високе використання - зменшити на 50%
            suggested = max(1, int(current_batch_size * 0.5))
        elif memory_percent > 85.0:
            # Середнє-високе - зменшити на 25%
            suggested = max(1, int(current_batch_size * 0.75))
        else:
            # Нормальне використання - залишити як є
            suggested = current_batch_size
        
        return suggested
    
    def auto_throttle(self, current_batch_size: int) -> Dict[str, any]:
        """
        Автоматичний throttle - повертає рекомендації та новий batch size.
        
        Args:
            current_batch_size: Поточний batch size
            
        Returns:
            Dict з рекомендаціями та suggested_batch_size
        """
        recommendations = self.get_throttle_recommendations()
        suggested_batch_size = self.get_suggested_batch_size(current_batch_size)
        
        recommendations['suggested_batch_size'] = suggested_batch_size
        recommendations['batch_size_changed'] = suggested_batch_size != current_batch_size
        
        return recommendations
    
    def get_statistics(self) -> Dict:
        """Отримати статистику ресурсів"""
        stats = {}
        
        if self.cpu_samples:
            stats['cpu'] = {
                'avg': sum(self.cpu_samples) / len(self.cpu_samples),
                'max': max(self.cpu_samples),
                'min': min(self.cpu_samples)
            }
        
        if self.memory_samples:
            stats['memory'] = {
                'avg_percent': sum(self.memory_samples) / len(self.memory_samples),
                'max_percent': max(self.memory_samples),
                'min_percent': min(self.memory_samples)
            }
        
        if self.gpu_samples:
            stats['gpu'] = {
                'avg_percent': sum(self.gpu_samples) / len(self.gpu_samples),
                'max_percent': max(self.gpu_samples),
                'min_percent': min(self.gpu_samples)
            }
        
        if self.batch_times:
            stats['batch_times'] = {
                'avg': sum(self.batch_times) / len(self.batch_times),
                'max': max(self.batch_times),
                'min': min(self.batch_times)
            }
        
        return stats
    
    def save_statistics(self, filename: str = "resource_statistics.json"):
        """Зберегти статистику в JSON"""
        stats = self.get_statistics()
        stats['log_file'] = str(self.log_file)
        stats['samples_count'] = len(self.cpu_samples)
        
        output_path = self.log_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return output_path

