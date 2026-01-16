"""
Універсальний скрипт для навчання TRM моделі
Використовує модульну структуру та автоматичну конфігурацію

⚠️ DEPRECATED: Як прямий entry point
Використовуйте: scripts/train_model.py → runtime.bootstrap

Функція train_with_auto_config() зберігається для backwards compatibility,
але рекомендовано використовувати runtime.bootstrap напряму.
"""
import torch
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from tiny_recursive_model import TinyRecursiveModel
from config import GGUFModelManager
from config.trm_config import TRMConfig
# DEPRECATED: Використовуйте runtime.resume замість config.training_resume
from runtime.resume import find_latest_checkpoint, get_checkpoint_info
# Backwards compatibility
try:
    from config.training_resume import TrainingResume
except ImportError:
    TrainingResume = None  # Fallback
from data import DatasetManager
from tiny_recursive_model.utils import load_tokenizer
from train.constants import (
    DEFAULT_DIM, DEFAULT_DEPTH, DEFAULT_SEQ_LEN,
    DEFAULT_NUM_REFINEMENT_BLOCKS, DEFAULT_NUM_LATENT_REFINEMENTS,
    DEFAULT_CHECKPOINT_INTERVAL, DEFAULT_TOKENIZER_NAME,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_HALT_PROB_THRES, DEFAULT_HALT_LOSS_WEIGHT,
    DEFAULT_CHECKPOINT_DIR
)
from train.model_factory import create_model, get_model_config_dict
from train.trainer_factory import create_trainer
from train.curriculum import CurriculumScheduler
from train.metrics import TRMTrainingLogger
from rag import build_rag, RAGDatasetWrapper
from train.training_utils import (
    resolve_dataset_path, handle_duplicate_training,
    handle_resume_checkpoint, save_model_and_config,
    get_final_loss_from_checkpoint
)


def train_with_auto_config(
    dataset_name_or_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    auto_config: bool = True,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_gpu: Optional[bool] = None,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    resume_from_checkpoint: Optional[str] = None,
    **kwargs
) -> Tuple[Optional['TinyRecursiveModel'], Optional[object], Optional[Path]]:
    """
    Навчати TRM модель з автоматичною конфігурацією
    
    Args:
        dataset_name_or_path: Ім'я датасету або шлях до нього
        model_save_path: Куди зберегти модель (None = trained_models/auto_model.pt)
        auto_config: Використовувати автоматичну конфігурацію
        epochs: Кількість епох (None = автоматично)
        batch_size: Розмір батчу (None = автоматично)
        learning_rate: Learning rate (None = автоматично)
        use_gpu: Використовувати GPU (None = автоматично визначити)
        checkpoint_dir: Папка для checkpoint'ів
        checkpoint_interval: Кожні скільки батчів зберігати checkpoint
        **kwargs: Додаткові параметри
    """
    print("=" * 70)
    print("🚀 НАВЧАННЯ TRM МОДЕЛІ")
    print("=" * 70)
    
    # Визначити використання GPU з перевіркою сумісності
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    # Перевірити сумісність GPU перед використанням
    if use_gpu:
        print(f"\n🔍 Перевірка сумісності GPU...")
        try:
            # Тестовий тензор для перевірки
            test_tensor = torch.randn(2, 2).cuda()
            _ = test_tensor + 1  # Проста операція
            # Тест embedding операції (яка викликає помилку)
            test_embed = torch.nn.Embedding(10, 5).cuda()
            test_input = torch.randint(0, 10, (2, 3)).cuda()
            _ = test_embed(test_input)  # Ця операція викликає помилку на несумісних GPU
            del test_tensor, test_embed, test_input
            torch.cuda.empty_cache()
            print(f"   ✅ GPU сумісний: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            error_str = str(e)
            if "HIP error" in error_str or "invalid device function" in error_str:
                print(f"   ⚠️  GPU не сумісний з поточною версією PyTorch/ROCm")
                print(f"   💡 Автоматично переключаємося на CPU")
                use_gpu = False
            else:
                raise
        except Exception as e:
            print(f"   ⚠️  Помилка при перевірці GPU: {e}")
            print(f"   💡 Автоматично переключаємося на CPU")
            use_gpu = False
    
    # Система продовження навчання
    training_resume = TrainingResume(checkpoint_dir=checkpoint_dir)
    
    # Менеджер датасетів
    dataset_manager = DatasetManager()
    
    # Знайти датасет
    dataset_path = resolve_dataset_path(dataset_name_or_path, dataset_manager)
    print(f"\n📚 Використовується датасет: {dataset_path.name}")
    
    # Автоматична конфігурація
    if auto_config:
        print("\n⚙️  Автоматична конфігурація навчання...")
        # Створити TRMConfig з автоматичним визначенням параметрів
        training_config = TRMConfig.from_dataset(
            dataset_path,
            auto_detect=True,
            dim=kwargs.get('dim', DEFAULT_DIM),
            depth=kwargs.get('depth', DEFAULT_DEPTH),
            seq_len=kwargs.get('seq_len', DEFAULT_SEQ_LEN),
            epochs=epochs or None,
            batch_size=batch_size or None,
            learning_rate=learning_rate or None
        )
        training_config.print_summary(dataset_path)
        
        # Перевизначити параметри якщо вказано вручну
        if epochs is not None:
            training_config.epochs = epochs
        if batch_size is not None:
            training_config.batch_size = batch_size
        if learning_rate is not None:
            training_config.learning_rate = learning_rate
    else:
        # Використати параметри за замовчуванням
        training_config = TRMConfig(
            dim=kwargs.get('dim', DEFAULT_DIM),
            depth=kwargs.get('depth', DEFAULT_DEPTH),
            seq_len=kwargs.get('seq_len', DEFAULT_SEQ_LEN),
            epochs=epochs or DEFAULT_EPOCHS,
            batch_size=batch_size or DEFAULT_BATCH_SIZE,
            learning_rate=learning_rate or DEFAULT_LEARNING_RATE,
            max_recurrent_steps=kwargs.get('max_recurrent_steps', 12),
            halt_prob_thres=kwargs.get('halt_prob_thres', DEFAULT_HALT_PROB_THRES)
        )
    
    # Завантажити токенізатор
    print(f"\n📥 Завантаження токенізатора...")
    tokenizer, vocab_size, pad_token_id = load_tokenizer(DEFAULT_TOKENIZER_NAME)
    if tokenizer is None:
        raise ValueError(f"Не вдалося завантажити токенізатор: {DEFAULT_TOKENIZER_NAME}")
    print(f"   ✅ Vocab size: {vocab_size}")
    
    # Створити датасет
    print(f"\n📚 Завантаження датасету...")
    from train.datasets.trm_dataset import TRMDataset
    from train.dataset_utils import split_dataset
    seq_len = kwargs.get('seq_len', DEFAULT_SEQ_LEN)
    full_dataset = TRMDataset(
        data_path=str(dataset_path),
        tokenizer=tokenizer,
        max_seq_len=seq_len,
        pad_token_id=pad_token_id,
        cache_size=kwargs.get('cache_size', 1000),
        validate_format=kwargs.get('validate_format', True)
    )
    print(f"   ✅ Датасет: {len(full_dataset)} прикладів")
    
    # Розділити на train/validation
    train_ratio = kwargs.get('train_ratio', 0.9)
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=train_ratio)
    print(f"   📊 Train: {len(train_dataset)} прикладів, Validation: {len(val_dataset)} прикладів")
    dataset = train_dataset  # Використовуємо train_dataset для навчання
    
    # Додати RAG якщо увімкнено
    rag_config = kwargs.get('rag', None)
    if rag_config and rag_config.get('enabled', False):
        print(f"\n🧠 Ініціалізація RAG...")
        # Створити документи з датасету для індексації
        # Для початку використаємо тексти з датасету
        documents = []
        try:
            # Спробувати витягнути тексти з датасету (перші 1000 для швидкості)
            sample_size = min(1000, len(dataset))
            for i in range(sample_size):
                item = dataset[i]
                if isinstance(item, dict):
                    text = item.get('input', item.get('context', ''))
                    if text:
                        documents.append(text)
                elif isinstance(item, tuple) and len(item) > 0:
                    if isinstance(item[0], str):
                        documents.append(item[0])
        except Exception as e:
            print(f"   ⚠️ Не вдалося витягнути документи з датасету: {e}")
        
        if documents:
            rag_retriever = build_rag(rag_config, documents)
            dataset = RAGDatasetWrapper(dataset, rag_retriever, k=rag_config.get('k', 5))
            print(f"   ✅ RAG увімкнено, датасет обгорнуто в RAGDatasetWrapper")
        else:
            print(f"   ⚠️ RAG увімкнено, але документи не знайдено. Продовжуємо без RAG.")
    
    # Параметри моделі
    dim = kwargs.get('dim', DEFAULT_DIM)
    depth = kwargs.get('depth', DEFAULT_DEPTH)
    
    # Отримати teacher модель з метаданих датасету
    dataset_metadata = training_resume.get_dataset_metadata(dataset_path)
    teacher_model_path = dataset_metadata.get('teacher_model_path') or dataset_metadata.get('teacher_model_name')
    
    # Створити конфігурацію моделі
    model_config = get_model_config_dict(
        dim=dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
        depth=depth,
        num_refinement_blocks=kwargs.get('num_refinement_blocks', DEFAULT_NUM_REFINEMENT_BLOCKS),
        num_latent_refinements=kwargs.get('num_latent_refinements', DEFAULT_NUM_LATENT_REFINEMENTS),
        max_recursion_depth=kwargs.get('max_recursion_depth', getattr(training_config, 'max_recursion_depth', 20)),
        adaptive_recursion=kwargs.get('adaptive_recursion', getattr(training_config, 'adaptive_recursion', False)),
        timeout_seconds=kwargs.get('timeout_seconds', None),
        thinking_cost_weight=kwargs.get('thinking_cost_weight', getattr(training_config, 'thinking_cost_weight', 0.01))
    )
    
    # Перевірити чи не навчаємося повторно на тому самому
    auto_resume = kwargs.get('auto_resume', True)  # За замовчуванням автоматично продовжувати
    is_cancelled, reason, previous_training = handle_duplicate_training(
        training_resume, dataset_path, model_config, teacher_model_path, auto_resume=auto_resume
    )
    if is_cancelled:
        return None, None, None
    
    # Визначити чи буде донавчання або продовження після екстреного переривання
    # Якщо checkpoint вказано явно, використати його, інакше автоматично визначити
    if resume_from_checkpoint is None:
        resume_from_checkpoint = handle_resume_checkpoint(
            training_resume, previous_training, teacher_model_path, 
            dataset_path=dataset_path, model_config=model_config,
            auto_resume=auto_resume
        )
    elif resume_from_checkpoint and not Path(resume_from_checkpoint).exists():
        print(f"⚠️  Checkpoint не знайдено: {resume_from_checkpoint}")
        print("   Буде розпочато нове навчання")
        resume_from_checkpoint = None
    
    # Створити модель
    print(f"\n🏗️  Створення моделі...")
    model = create_model(
        dim=dim,
        vocab_size=vocab_size,
        depth=depth,
        seq_len=seq_len,
        num_refinement_blocks=kwargs.get('num_refinement_blocks', DEFAULT_NUM_REFINEMENT_BLOCKS),
        num_latent_refinements=kwargs.get('num_latent_refinements', DEFAULT_NUM_LATENT_REFINEMENTS),
        halt_loss_weight=kwargs.get('halt_loss_weight', DEFAULT_HALT_LOSS_WEIGHT),
        max_recursion_depth=kwargs.get('max_recursion_depth', getattr(training_config, 'max_recursion_depth', 20)),
        adaptive_recursion=kwargs.get('adaptive_recursion', getattr(training_config, 'adaptive_recursion', False)),
        timeout_seconds=kwargs.get('timeout_seconds', None),
        thinking_cost_weight=kwargs.get('thinking_cost_weight', getattr(training_config, 'thinking_cost_weight', 0.01))
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Параметрів: {total_params:,}")
    
    # Визначити шлях для збереження
    if model_save_path is None:
        trained_models_dir = Path(__file__).parent.parent / "models" / "trained"
        trained_models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = trained_models_dir / f"trm_{Path(dataset_path).stem}.pt"
    else:
        model_save_path = Path(model_save_path)
    
    # Створити curriculum scheduler якщо увімкнено
    curriculum_scheduler = None
    curriculum_config = kwargs.get('curriculum', None)
    if curriculum_config and curriculum_config.get('enabled', False):
        stages = curriculum_config.get('stages', [])
        if stages:
            curriculum_scheduler = CurriculumScheduler(stages)
            print(f"\n📚 Створено CurriculumScheduler з {len(stages)} етапами")
            print(f"   {curriculum_scheduler.describe()}")
    
    # Створити training logger для метрик
    log_dir = kwargs.get('log_dir', 'logs')
    training_logger = TRMTrainingLogger(Path(log_dir) / 'training_metrics.jsonl')
    print(f"\n📊 Створено TRMTrainingLogger: {Path(log_dir) / 'training_metrics.jsonl'}")
    
    # Створити trainer
    print(f"\n🎓 Створення trainer...")
    trainer = create_trainer(
        model=model,
        dataset=dataset,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        max_recurrent_steps=training_config.max_recurrent_steps,
        use_gpu=use_gpu,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        halt_prob_thres=training_config.halt_prob_thres,
        mixed_precision=kwargs.get('mixed_precision', None),
        curriculum_scheduler=curriculum_scheduler,
        training_logger=training_logger
    )
    
    # Навчання
    print(f"\n🚀 Початок навчання...")
    print("\n" + "=" * 70)
    print("📋 ПАРАМЕТРИ НАВЧАННЯ:")
    print("=" * 70)
    print(f"   📚 Датасет: {dataset_path.name}")
    print(f"   🎯 Модель:")
    print(f"      - dim: {dim}")
    print(f"      - depth: {depth}")
    print(f"      - seq_len: {seq_len}")
    print(f"      - vocab_size: {vocab_size}")
    print(f"      - num_refinement_blocks: {kwargs.get('num_refinement_blocks', DEFAULT_NUM_REFINEMENT_BLOCKS)}")
    print(f"      - num_latent_refinements: {kwargs.get('num_latent_refinements', DEFAULT_NUM_LATENT_REFINEMENTS)}")
    print(f"   🎓 Навчання:")
    print(f"      - epochs: {training_config.epochs}")
    print(f"      - batch_size: {training_config.batch_size}")
    print(f"      - learning_rate: {training_config.learning_rate}")
    print(f"      - gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"      - warmup_steps: {training_config.warmup_steps}")
    print(f"      - max_recurrent_steps: {training_config.max_recurrent_steps}")
    print(f"      - halt_prob_thres: {training_config.halt_prob_thres}")
    print(f"      - halt_loss_weight: {kwargs.get('halt_loss_weight', DEFAULT_HALT_LOSS_WEIGHT)}")
    print(f"   💾 Checkpoint:")
    print(f"      - checkpoint_dir: {checkpoint_dir}")
    print(f"      - checkpoint_interval: {checkpoint_interval}")
    print(f"   🔧 Інші параметри:")
    print(f"      - use_gpu: {use_gpu}")
    print(f"      - mixed_precision: {kwargs.get('mixed_precision', 'None')}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   📊 Загальна кількість параметрів: {total_params:,}")
    print("=" * 70)
    if resume_from_checkpoint:
        print(f"\n   ⏮️  Продовження з checkpoint: {Path(resume_from_checkpoint).name}")
    print("-" * 70)
    
    # Навчання з автоматичним fallback на CPU
    final_loss = None
    try:
        trainer(resume_from_checkpoint=resume_from_checkpoint)
        final_loss = get_final_loss_from_checkpoint(checkpoint_dir)
    except RuntimeError as e:
        error_str = str(e)
        if use_gpu and ("HIP error" in error_str or "invalid device function" in error_str):
            print(f"\n⚠️  Помилка GPU: {error_str}")
            print(f"💡 Автоматично перезапускаємо навчання на CPU...")
            print("-" * 70)
            
            # Перемістити модель на CPU
            model = model.cpu()
            torch.cuda.empty_cache()
            
            # Перестворити trainer з CPU
            trainer = create_trainer(
                model=model,
                dataset=train_dataset,
                learning_rate=training_config.learning_rate,
                batch_size=training_config.batch_size,
                epochs=training_config.epochs,
                max_recurrent_steps=training_config.max_recurrent_steps,
                use_gpu=False,  # Примусово CPU
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                warmup_steps=training_config.warmup_steps,
                halt_prob_thres=training_config.halt_prob_thres,
                mixed_precision=None  # Без mixed precision на CPU
            )
            
            # Оновити параметри для відображення
            use_gpu = False
            print(f"\n🔄 Параметри оновлено:")
            print(f"   - use_gpu: {use_gpu} (CPU)")
            print(f"   - mixed_precision: None")
            print("-" * 70)
            
            # Повторний запуск на CPU
            try:
                trainer(resume_from_checkpoint=resume_from_checkpoint)
                final_loss = get_final_loss_from_checkpoint(checkpoint_dir)
            except Exception as e2:
                print(f"\n❌ Помилка під час навчання на CPU: {e2}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n❌ Помилка під час навчання: {e}")
            import traceback
            traceback.print_exc()
    except KeyboardInterrupt:
        print("\n⚠️ Навчання перервано користувачем")
    except Exception as e:
        print(f"\n❌ Помилка під час навчання: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 70)
    
    # Зберегти запис про навчання
    if final_loss is not None:
        try:
            latest_checkpoint = Path(checkpoint_dir) / "checkpoint_latest.pt"
            training_resume.save_training_record(
                dataset_path,
                model_config,
                final_loss,
                latest_checkpoint if latest_checkpoint.exists() else None,
                teacher_model_path=teacher_model_path
            )
        except Exception as e:
            print(f"⚠️ Не вдалося зберегти запис про навчання: {e}")
            import traceback
            traceback.print_exc()
    
    # Зберегти модель (тільки на головному процесі)
    print(f"\n💾 Збереження моделі...")
    if trainer.accelerator.is_main_process:
        try:
            save_model_and_config(model, model_save_path, model_config, training_config)
            print(f"   ✅ Модель збережено: {model_save_path}")
        except Exception as e:
            print(f"   ❌ Помилка збереження моделі: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ НАВЧАННЯ ЗАВЕРШЕНО!")
    print("=" * 70)
    
    return model, tokenizer, model_save_path


def main():
    """CLI для навчання"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Навчання TRM моделі")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Ім'я датасету або шлях до файлу")
    parser.add_argument("--save", type=str, default=None,
                       help="Шлях для збереження моделі")
    parser.add_argument("--no-auto-config", action="store_true",
                       help="Не використовувати автоматичну конфігурацію")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Кількість епох")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Розмір батчу")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--dim", type=int, default=512,
                       help="Розмірність моделі")
    parser.add_argument("--depth", type=int, default=4,
                       help="Глибина MLP Mixer")
    parser.add_argument("--seq-len", type=int, default=2048,
                       help="Довжина послідовності")
    parser.add_argument("--cpu", action="store_true",
                       help="Використовувати CPU")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Папка для checkpoint'ів")
    
    args = parser.parse_args()
    
    train_with_auto_config(
        dataset_name_or_path=args.dataset,
        model_save_path=args.save,
        auto_config=not args.no_auto_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.cpu,
        checkpoint_dir=args.checkpoint_dir,
        dim=args.dim,
        depth=args.depth,
        seq_len=args.seq_len
    )


if __name__ == "__main__":
    main()

