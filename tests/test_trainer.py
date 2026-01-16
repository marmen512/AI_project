"""
Unit-тести для Trainer (після callback-рефакторингу)
Mockable компоненти через callbacks
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from tiny_recursive_model.trainer import Trainer
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from train.curriculum import CurriculumScheduler
from train.callbacks.base import Callback
from train.callbacks.curriculum import CurriculumCallback
from train.callbacks.checkpoint import CheckpointCallback
from train.callbacks.logging import LoggingCallback
from train.callbacks.early_stopping import EarlyStoppingCallback
from core.types import TrainState


class MockDataset:
    """Mock dataset для тестування"""
    
    def __init__(self, num_samples=10, seq_len=32):
        self.num_samples = num_samples
        self.seq_len = seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Повернути tuple (input_ids, labels) для backwards compatibility
        input_ids = torch.randint(0, 100, (self.seq_len,))
        labels = torch.randint(0, 100, (self.seq_len,))
        return input_ids, labels


@pytest.fixture
def mock_model():
    """Створити mock модель"""
    network = MLPMixer1D(dim=64, depth=2, seq_len=32)
    model = TinyRecursiveModel(
        dim=64,
        num_tokens=100,
        network=network,
        max_recursion_depth=5
    )
    return model


@pytest.fixture
def mock_dataset():
    """Створити mock dataset"""
    return MockDataset(num_samples=10, seq_len=32)


@pytest.fixture
def trainer(mock_model, mock_dataset):
    """Створити trainer для тестування"""
    return Trainer(
        model=mock_model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=1,
        max_recurrent_steps=3,
        checkpoint_dir=None  # Без checkpoint для тестів
    )


def test_trainer_initialization(trainer):
    """Тест ініціалізації trainer"""
    assert trainer is not None
    assert trainer.model is not None
    assert trainer.batch_size == 2
    assert trainer.epochs == 1
    assert trainer.max_recurrent_steps == 3
    assert trainer.callbacks is not None


def test_trainer_callbacks_integration(trainer):
    """Тест інтеграції callbacks"""
    # Перевірити що callbacks ініціалізовані
    assert trainer.callbacks is not None
    assert hasattr(trainer.callbacks, 'callbacks')
    
    # Додати mock callback
    mock_callback = Mock(spec=Callback)
    trainer.callbacks.add(mock_callback)
    
    # Перевірити що callback додано
    assert len(trainer.callbacks.callbacks) > 0


def test_trainer_with_curriculum_callback(trainer, mock_dataset):
    """Тест trainer з curriculum callback"""
    # Створити curriculum scheduler
    curriculum_stages = [
        {'name': 'stage1', 'seq_len': 32, 'dim': 64, 'batch': 2, 'epochs': 1}
    ]
    scheduler = CurriculumScheduler(stages=curriculum_stages)
    
    # Створити trainer з curriculum
    trainer_with_curriculum = Trainer(
        model=trainer.model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=1,
        max_recurrent_steps=3,
        curriculum_scheduler=scheduler,
        checkpoint_dir=None
    )
    
    # Перевірити що curriculum callback додано
    curriculum_callbacks = [cb for cb in trainer_with_curriculum.callbacks.callbacks 
                           if isinstance(cb, CurriculumCallback)]
    assert len(curriculum_callbacks) > 0


def test_trainer_with_checkpoint_callback(mock_model, mock_dataset, tmp_path):
    """Тест trainer з checkpoint callback"""
    checkpoint_dir = tmp_path / "checkpoints"
    
    trainer = Trainer(
        model=mock_model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=1,
        max_recurrent_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=5
    )
    
    # Перевірити що checkpoint callback додано
    checkpoint_callbacks = [cb for cb in trainer.callbacks.callbacks 
                           if isinstance(cb, CheckpointCallback)]
    assert len(checkpoint_callbacks) > 0
    assert trainer.checkpoint_manager is not None


def test_trainer_with_logging_callback(mock_model, mock_dataset, tmp_path):
    """Тест trainer з logging callback"""
    from train.metrics.trm_metrics import TRMTrainingLogger
    
    log_dir = tmp_path / "logs"
    logger = TRMTrainingLogger(log_dir=log_dir)
    
    trainer = Trainer(
        model=mock_model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=1,
        max_recurrent_steps=3,
        training_logger=logger,
        checkpoint_dir=None
    )
    
    # Перевірити що logging callback додано
    logging_callbacks = [cb for cb in trainer.callbacks.callbacks 
                        if isinstance(cb, LoggingCallback)]
    assert len(logging_callbacks) > 0


def test_trainer_with_early_stopping_callback(mock_model, mock_dataset):
    """Тест trainer з early stopping callback"""
    early_stop = EarlyStoppingCallback(patience=2, entropy_patience=2)
    
    callbacks = [early_stop]
    trainer = Trainer(
        model=mock_model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=10,  # Велика кількість епох
        max_recurrent_steps=3,
        callbacks=callbacks,
        checkpoint_dir=None
    )
    
    # Перевірити що early stopping callback додано
    early_stop_callbacks = [cb for cb in trainer.callbacks.callbacks 
                           if isinstance(cb, EarlyStoppingCallback)]
    assert len(early_stop_callbacks) > 0


def test_train_state_update(trainer):
    """Тест оновлення TrainState"""
    assert trainer.train_state is not None
    
    # Оновити state
    trainer.train_state.update(epoch=1, step=10, loss=0.5)
    
    assert trainer.train_state.epoch == 1
    assert trainer.train_state.step == 10
    assert trainer.train_state.loss == 0.5


def test_trainer_resume(mock_model, mock_dataset, tmp_path):
    """Тест resume функціональності"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    trainer = Trainer(
        model=mock_model,
        dataset=mock_dataset,
        batch_size=2,
        epochs=2,
        max_recurrent_steps=3,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Створити тестовий checkpoint
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    torch.save({
        'epoch': 1,
        'batch_idx': 5,
        'batch_count': 10,
        'model_state_dict': mock_model.state_dict(),
        'optimizer_state_dict': trainer.optim.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_state': trainer.train_state.to_dict()
    }, checkpoint_path)
    
    # Завантажити checkpoint
    epoch, batch_idx, batch_count = trainer.load_checkpoint(checkpoint_path)
    
    assert epoch == 1
    assert batch_idx == 5
    assert batch_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

