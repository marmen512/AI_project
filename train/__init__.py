"""
Модуль навчання TRM моделі
"""
from train.constants import *
from train.exceptions import (
    TrainingError,
    DatasetNotFoundError,
    CheckpointError,
    ModelConfigError,
    ModelLoadError
)
from train.model_factory import create_model, load_model_from_checkpoint, get_model_config_dict
from train.trainer_factory import create_trainer
from train.training_utils import (
    resolve_dataset_path,
    handle_duplicate_training,
    handle_resume_checkpoint,
    save_model_and_config,
    get_final_loss_from_checkpoint
)
from train.train import train_with_auto_config

# Datasets
from train.datasets.trm_dataset import TRMDataset
from train.datasets.splits import split_by_doc_id, validate_split_integrity

# Loss computation
from train.loss import (
    compute_task_loss,
    compute_halt_loss,
    compute_thinking_cost,
    compute_combined_loss
)

# Validation
from train.validation import (
    validate_model,
    validate_with_metrics,
    compute_accuracy
)

__all__ = [
    # Constants
    'DEFAULT_DIM',
    'DEFAULT_DEPTH',
    'DEFAULT_SEQ_LEN',
    'DEFAULT_VOCAB_SIZE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_EPOCHS',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_MAX_RECURRENT_STEPS',
    'DEFAULT_GRADIENT_ACCUMULATION_STEPS',
    'DEFAULT_WARMUP_STEPS',
    'DEFAULT_NUM_REFINEMENT_BLOCKS',
    'DEFAULT_NUM_LATENT_REFINEMENTS',
    'DEFAULT_HALT_LOSS_WEIGHT',
    'DEFAULT_HALT_PROB_THRES',
    'DEFAULT_CHECKPOINT_INTERVAL',
    'DEFAULT_CHECKPOINT_DIR',
    'MAX_NUM_WORKERS',
    'DEFAULT_CACHE_TOKENS',
    'CACHE_SIZE_LIMIT',
    'DEFAULT_TOKENIZER_NAME',
    # Exceptions
    'TrainingError',
    'DatasetNotFoundError',
    'CheckpointError',
    'ModelConfigError',
    'ModelLoadError',
    # Factory functions
    'create_model',
    'load_model_from_checkpoint',
    'get_model_config_dict',
    'create_trainer',
    # Utilities
    'resolve_dataset_path',
    'handle_duplicate_training',
    'handle_resume_checkpoint',
    'save_model_and_config',
    'get_final_loss_from_checkpoint',
    # Main training function
    'train_with_auto_config',
    # Datasets
    'TRMDataset',
    'split_by_doc_id',
    'validate_split_integrity',
    # Loss
    'compute_task_loss',
    'compute_halt_loss',
    'compute_thinking_cost',
    'compute_combined_loss',
    # Validation
    'validate_model',
    'validate_with_metrics',
    'compute_accuracy',
]









