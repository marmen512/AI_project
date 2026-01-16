"""
Загальні константи проекту
"""
from enum import Enum


class TrainingMode(Enum):
    """Режими навчання"""
    NEW = "new"  # Нове навчання
    RESUME = "resume"  # Продовження навчання
    SERVICE = "service"  # Service режим (неперервне навчання)


class InferenceMode(Enum):
    """Режими інференсу"""
    STANDARD = "standard"  # Стандартний інференс
    DETERMINISTIC = "deterministic"  # Детерміністичний режим для debug
    FAST = "fast"  # Швидкий режим (менше recurrent steps)


# Default values
DEFAULT_THINKING_COST_WEIGHT = 0.01
DEFAULT_MAX_RECURRENT_STEPS = 12
DEFAULT_HALT_PROB_THRES = 0.5
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 10

# Checkpoint naming
CHECKPOINT_LATEST = "checkpoint_latest.pt"
CHECKPOINT_BEST_LOSS = "best_loss.ckpt"
CHECKPOINT_BEST_EVAL = "best_eval_score.ckpt"
CHECKPOINT_BEST_ENTROPY = "best_entropy.ckpt"

# Dataset metadata keys
DATASET_DOC_ID = "doc_id"
DATASET_SEGMENT_ID = "segment_id"

# Eval samples directory
EVAL_SAMPLES_DIR = "eval_samples"

