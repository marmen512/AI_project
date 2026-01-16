"""
Єдина точка входу для навчання
Bootstrap функція ініціалізує всі компоненти та запускає навчання
"""
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# Встановити UTF-8 для виводу в Windows
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

try:
    import yaml
except ImportError:
    # Якщо yaml не встановлено, використовувати json
    import json as yaml

# Додати project root до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from runtime.modes import TrainingMode, parse_mode
from runtime.resume import find_latest_checkpoint
from runtime.checkpointing import CheckpointManager
from config.trm_config import TRMConfig
from train.logging import TRMTrainingLogger
from train.resource_monitor import ResourceMonitor


def _setup_cpu_optimization(cfg: Dict[str, Any]) -> None:
    """
    Налаштувати оптимізацію CPU під Ryzen 5 3600 та NUMA
    
    Args:
        cfg: Конфігурація
    """
    cpu_cfg = cfg.get('cpu_optimization', {})
    num_threads = cpu_cfg.get('num_threads', 6)  # Ryzen 5 3600 = 6 cores
    num_interop_threads = cpu_cfg.get('num_interop_threads', 2)
    
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    
    print(f"[CPU] CPU оптимізація: threads={num_threads}, interop_threads={num_interop_threads}")


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Завантажити конфігурацію з YAML або використати дефолти
    
    Args:
        config_path: Шлях до config.yaml
    
    Returns:
        Словник конфігурації
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            if hasattr(yaml, 'safe_load'):
                cfg = yaml.safe_load(f)
            else:
                # Якщо yaml не доступний, використати json
                cfg = yaml.load(f)
        print(f"[CONFIG] Завантажено конфігурацію: {config_path}")
    else:
        # Використати дефолти з TRMConfig
        trm_config = TRMConfig()
        cfg = {
            'model': {
                'dim': trm_config.dim,
                'depth': trm_config.depth,
                'seq_len': trm_config.seq_len,
                'vocab_size': trm_config.vocab_size,
            },
            'training': {
                'batch_size': trm_config.batch_size,
                'epochs': trm_config.epochs,
                'learning_rate': trm_config.learning_rate,
                'max_recurrent_steps': trm_config.max_recurrent_steps,
                'halt_prob_thres': trm_config.halt_prob_thres,
                'checkpoint_dir': 'checkpoints',
                'checkpoint_interval': 100,
                'auto_resume': False,
            },
            'curriculum': {
                'enabled': trm_config.curriculum_enabled,
                'start_len': trm_config.curriculum_start_len,
                'max_len': trm_config.curriculum_max_len,
            },
            'cpu_optimization': {
                'num_threads': 6,
                'num_interop_threads': 2,
                'num_workers': 0,
                'numa_enabled': False,
            },
        }
        print("[CONFIG] Використано дефолтну конфігурацію")
    
    return cfg


def _setup_logging(cfg: Dict[str, Any]) -> TRMTrainingLogger:
    """
    Ініціалізувати логування
    
    Args:
        cfg: Конфігурація
    
    Returns:
        Logger instance
    """
    log_dir = cfg.get('training', {}).get('log_dir', 'logs')
    logger = TRMTrainingLogger(log_dir=log_dir)
    print(f"[LOG] Логування: {log_dir}")
    return logger


def _setup_monitoring(cfg: Dict[str, Any], logger: TRMTrainingLogger) -> Optional[ResourceMonitor]:
    """
    Ініціалізувати моніторинг ресурсів
    
    Args:
        cfg: Конфігурація
        logger: Logger instance
    
    Returns:
        ResourceMonitor instance або None
    """
    monitoring_enabled = cfg.get('training', {}).get('enable_monitoring', True)
    if not monitoring_enabled:
        return None
    
    log_dir = cfg.get('training', {}).get('log_dir', 'logs')
    log_interval = cfg.get('training', {}).get('monitoring_interval', 10)
    monitor = ResourceMonitor(log_dir=log_dir, log_interval=log_interval)
    print(f"[MONITOR] Моніторинг ресурсів: увімкнено (інтервал: {log_interval})")
    return monitor


def _build_components(cfg: Dict[str, Any], logger: TRMTrainingLogger):
    """
    Створити всі компоненти через factories
    
    Args:
        cfg: Конфігурація
        logger: Logger instance
    
    Returns:
        Tuple (model, optimizer, dataset, trainer, tokenizer, pad_token_id)
    """
    # Використовувати factories
    from factories.model_factory import build_model
    from factories.optimizer_factory import build_optimizer
    from factories.dataset_factory import build_dataset
    from tiny_recursive_model.utils import load_tokenizer
    
    # Завантажити tokenizer
    print("[LOAD] Завантаження tokenizer...")
    tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
    print(f"   [OK] Vocab size: {vocab_size}")
    
    # Оновити vocab_size в конфігурації
    if cfg.get('model', {}).get('vocab_size') is None:
        cfg.setdefault('model', {})['vocab_size'] = vocab_size
    
    # Створити модель через factory
    print("\n[BUILD] Створення моделі...")
    model = build_model(cfg)
    print(f"   [OK] Модель створена")
    
    # Створити dataset через factory
    print("\n[BUILD] Створення dataset...")
    dataset = build_dataset(cfg, tokenizer, pad_token_id)
    print(f"   [OK] Датасет: {len(dataset)} прикладів")
    
    # Створити optimizer через factory
    print("\n[BUILD] Створення optimizer...")
    optimizer, scheduler = build_optimizer(cfg, model)
    print(f"   [OK] Optimizer створено")
    
    # Створити trainer
    from train.trainer_factory import create_trainer
    training_cfg = cfg.get('training', {})
    # Конвертувати learning_rate в float якщо рядок
    learning_rate = training_cfg.get('learning_rate', 1e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    trainer = create_trainer(
        model=model,
        dataset=dataset,
        learning_rate=learning_rate,
        batch_size=training_cfg.get('batch_size', 4),
        epochs=training_cfg.get('epochs', 10),
        max_recurrent_steps=training_cfg.get('max_recurrent_steps', 12),
        halt_prob_thres=training_cfg.get('halt_prob_thres', 0.5),
        thinking_cost_weight=training_cfg.get('thinking_cost_weight', 0.01),
        use_gpu=torch.cuda.is_available(),
        checkpoint_dir=training_cfg.get('checkpoint_dir', 'checkpoints'),
        checkpoint_interval=training_cfg.get('checkpoint_interval', 100),
        training_logger=logger,
        warmup_steps=training_cfg.get('warmup_steps', 2000)
    )
    print(f"   [OK] Trainer створено")
    
    # Примітка: Trainer створює свій власний optimizer, тому переданий optimizer не використовується
    # Але scheduler можна використати, якщо Trainer підтримує зовнішній scheduler
    
    return model, optimizer, dataset, trainer, tokenizer, pad_token_id


def bootstrap(
    config_path: Optional[str] = None,
    mode: str = "new",
    resume_from: Optional[str] = None
) -> None:
    """
    Головна функція bootstrap - єдина точка входу для навчання
    
    Args:
        config_path: Шлях до config.yaml
        mode: Режим навчання ("new", "resume", "service")
        resume_from: Шлях до checkpoint для resume (опціонально)
    """
    print("=" * 80)
    print("[START] TRM TRAINING BOOTSTRAP")
    print("=" * 80)
    
    # Парсити режим
    training_mode = parse_mode(mode)
    print(f"[MODE] Режим: {training_mode.value}")
    
    # Завантажити конфігурацію
    cfg = _load_config(config_path)
    
    # Налаштувати CPU оптимізацію
    _setup_cpu_optimization(cfg)
    
    # Ініціалізувати логування
    logger = _setup_logging(cfg)
    
    # Ініціалізувати моніторинг
    monitor = _setup_monitoring(cfg, logger)
    
    # Створити компоненти
    model, optimizer, dataset, trainer, tokenizer, pad_token_id = _build_components(cfg, logger)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        path=cfg.get('training', {}).get('checkpoint_dir', 'checkpoints'),
        keep_last=cfg.get('training', {}).get('keep_checkpoints', 5)
    )
    
    # Обробити resume
    resume_checkpoint_path = None
    if training_mode == TrainingMode.RESUME or resume_from:
        if resume_from:
            checkpoint_path = Path(resume_from)
        else:
            checkpoint_path = find_latest_checkpoint(checkpoint_manager.path)
        
        if checkpoint_path and checkpoint_path.exists():
            print(f"\n⏮️  Продовження навчання з checkpoint: {checkpoint_path.name}")
            checkpoint_manager.load_latest(model, optimizer)
            resume_checkpoint_path = str(checkpoint_path)
        else:
            print("[WARN] Checkpoint не знайдено, починаємо нове навчання")
    
    # Запустити навчання
    print("\n" + "=" * 80)
    print("[START] ПОЧАТОК НАВЧАННЯ")
    print("=" * 80)
    print("")
    
    try:
        # Запустити trainer
        if hasattr(trainer, 'forward'):
            trainer.forward(resume_from_checkpoint=resume_checkpoint_path)
        elif callable(trainer):
            trainer(resume_from_checkpoint=resume_checkpoint_path)
        else:
            raise ValueError("Trainer не підтримує запуск навчання")
        
        # Зберегти метрики
        metrics_path = logger.save()
        print(f"\n[SAVE] Метрики збережено: {metrics_path}")
        
        # Зберегти статистику ресурсів
        if monitor:
            resource_stats_path = monitor.save_statistics()
            print(f"[SAVE] Статистика ресурсів: {resource_stats_path.name}")
        
        print("\n" + "=" * 80)
        print("[OK] НАВЧАННЯ ЗАВЕРШЕНО!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Навчання перервано користувачем")
        if monitor:
            monitor.save_statistics("resource_statistics_interrupted.json")
    except Exception as e:
        print(f"\n[ERROR] Помилка під час навчання: {e}")
        import traceback
        traceback.print_exc()
        if monitor:
            monitor.save_statistics("resource_statistics_error.json")
    finally:
        # Cleanup
        if monitor and hasattr(monitor, 'stop'):
            monitor.stop()
        if hasattr(logger, 'close'):
            logger.close()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TRM Training Bootstrap - єдина точка входу для навчання"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Шлях до config.yaml (за замовчуванням: config/config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="new",
        choices=["new", "resume", "service"],
        help="Режим навчання (new, resume, service)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Шлях до checkpoint для продовження"
    )
    
    args = parser.parse_args()
    
    # Якщо config не вказано, спробувати знайти за замовчуванням
    if args.config is None:
        default_config = project_root / "config" / "config.yaml"
        if default_config.exists():
            args.config = str(default_config)
    
    bootstrap(
        config_path=args.config,
        mode=args.mode,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()

