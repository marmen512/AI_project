#!/bin/bash
# Автоматичне навчання TRM після генерації датасету

# Отримати директорію скрипта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Активація віртуального середовища
if [ ! -d "venv" ]; then
    echo "❌ Помилка: venv не знайдено в $PROJECT_ROOT"
    exit 1
fi

source venv/bin/activate

# Параметри (TRM-оптимальні за замовчуванням)
# Використовуємо існуючий датасет за замовчуванням, або чекаємо на phi3_training_dataset.json
DATASET="${DATASET:-datasets/train/openassistant_train.json}"
DIM="${DIM:-256}"           # TRM-оптимально (було 1024)
DEPTH="${DEPTH:-4}"
SEQ_LEN="${SEQ_LEN:-256}"  # TRM-оптимально (було 4096, curriculum: 64→256)
BATCH_SIZE="${BATCH_SIZE:-4}"  # TRM-оптимально (було 1)
EPOCHS="${EPOCHS:-10}"     # TRM-оптимально (було 15)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"  # TRM-оптимально (було 2e-4)

# Якщо DATASET не вказує на існуючий файл, перевірити альтернативні варіанти
if [ ! -f "$DATASET" ]; then
    # Спробувати знайти phi3_training_dataset.json в корені проекту
    if [ -f "phi3_training_dataset.json" ]; then
        DATASET="phi3_training_dataset.json"
    elif [ -f "datasets/train/openassistant_train.json" ]; then
        DATASET="datasets/train/openassistant_train.json"
    else
        echo "⏳ Очікування завершення генерації датасету..."
        echo "   Шукаю: phi3_training_dataset.json"
        echo ""
        
        # Чекати поки датасет не буде створено (тільки для phi3_training_dataset.json)
        while [ ! -f "phi3_training_dataset.json" ] && [ ! -f "datasets/train/openassistant_train.json" ]; do
            sleep 10
            echo "   Чекаю... ($(date +%H:%M:%S))"
        done
        
        # Використати знайдений датасет
        if [ -f "phi3_training_dataset.json" ]; then
            DATASET="phi3_training_dataset.json"
        elif [ -f "datasets/train/openassistant_train.json" ]; then
            DATASET="datasets/train/openassistant_train.json"
        fi
    fi
fi

echo "✅ Використовується датасет: $DATASET"
if [ -f "$DATASET" ]; then
    echo "📊 Розмір датасету:"
    if command -v wc > /dev/null; then
        wc -l "$DATASET" 2>/dev/null || echo "   (не вдалося підрахувати рядки)"
    fi
else
    echo "⚠️  УВАГА: Датасет не знайдено: $DATASET"
    echo "   Скрипт продовжить, але навчання може не запуститися"
fi
echo ""

echo "🚀 Початок навчання TRM..."
echo "   Параметри: dim=$DIM, depth=$DEPTH, seq_len=$SEQ_LEN"
echo "   batch_size=$BATCH_SIZE, epochs=$EPOCHS, lr=$LEARNING_RATE"
echo ""

# Автоматичний цикл навчання (перезапуск після завершення)
TRAIN_COUNT=0
MAX_RESTARTS=999  # Максимальна кількість перезапусків (практично безмежно)

while [ $TRAIN_COUNT -lt $MAX_RESTARTS ]; do
    TRAIN_COUNT=$((TRAIN_COUNT + 1))
    
    echo "=========================================="
    echo "🔄 Цикл навчання #$TRAIN_COUNT"
    echo "=========================================="
    echo ""
    
    # Перевірка чи вже працює навчання
    if pgrep -f "runtime.bootstrap" > /dev/null; then
        echo "⚠️  Навчання вже працює (PID: $(pgrep -f 'runtime.bootstrap'))"
        echo "   Чекаю завершення..."
        while pgrep -f "runtime.bootstrap" > /dev/null; do
            sleep 30
        done
        echo "   ✅ Попереднє навчання завершено"
        echo ""
    fi
    
    # Створити папку logs якщо не існує
    mkdir -p logs
    
    # Генерувати ім'я лог-файлу з timestamp для кожного циклу
    LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
    LATEST_LOG="logs/training_latest.log"
    
    # Використовувати runtime.bootstrap (service режим)
    python -m runtime.bootstrap \
        --mode service \
        --config config/config.yaml \
        2>&1 | tee "$LOG_FILE" | tee "$LATEST_LOG"
    
    EXIT_CODE=$?
    
    echo ""
    echo "=========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Навчання #$TRAIN_COUNT завершено успішно!"
    else
        echo "⚠️  Навчання #$TRAIN_COUNT завершено з кодом виходу: $EXIT_CODE"
    fi
    echo "=========================================="
    echo ""
    
    # Пауза перед наступним циклом (5 хвилин)
    echo "⏳ Пауза 5 хвилин перед наступним циклом навчання..."
    echo "   (Натисніть Ctrl+C для зупинки)"
    sleep 300
    
    echo ""
done

echo "✅ Досягнуто максимальну кількість циклів ($MAX_RESTARTS)"

