#!/bin/bash
# Централізований конфіг для параметрів навчання
# Використовується в скриптах навчання

# ⚠️ DEPRECATED: Для runtime.bootstrap не використовується
# runtime.bootstrap читає параметри з config/config.yaml
# Цей файл використовується тільки старішим start_training.sh (якщо не використовує runtime.bootstrap)

# Параметри моделі (TRM-оптимальні)
export DEFAULT_DIM=256           # TRM-оптимально (було 1024)
export DEFAULT_DEPTH=4
export DEFAULT_SEQ_LEN=256      # TRM-оптимально (було 4096, curriculum: 64→256)

# Параметри навчання (TRM-оптимальні)
export DEFAULT_BATCH_SIZE=4     # TRM-оптимально (було 1)
export DEFAULT_EPOCHS=10        # TRM-оптимально (було 15)
export DEFAULT_LEARNING_RATE="1e-4"  # TRM-оптимально (було 2e-4)

# Параметри checkpoint
export DEFAULT_CHECKPOINT_DIR="checkpoints"
export DEFAULT_CHECKPOINT_INTERVAL=100

# Датасет за замовчуванням
export DEFAULT_DATASET="datasets/train/openassistant_train.json"

# Функція для завантаження конфігу
load_training_config() {
    if [ -f "$(dirname "$0")/config/training_defaults.sh" ]; then
        source "$(dirname "$0")/config/training_defaults.sh"
    elif [ -f "$(dirname "$0")/../config/training_defaults.sh" ]; then
        source "$(dirname "$0")/../config/training_defaults.sh"
    fi
}



