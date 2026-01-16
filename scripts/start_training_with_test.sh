#!/bin/bash
# Скрипт для запуску навчання з інтерактивним тестуванням на Linux
# Після кожного checkpoint питає чи хоче користувач протестувати модель

set -e

# Перейти в директорію скрипта
cd "$(dirname "$0")/.."

echo "=========================================="
echo "[START] Навчання з інтерактивним тестуванням"
echo "=========================================="
echo ""

# Перевірка наявності venv
if [ ! -d "venv" ]; then
    echo "[ERROR] venv не знайдено"
    echo "   Створіть venv: ./setup_venv.sh"
    exit 1
fi

# Активувати venv
echo "[ACTIVATE] Активування віртуального середовища..."
source venv/bin/activate

# Параметри
CHECKPOINT_DIR="checkpoints"
TEST_INTERVAL=5  # Секунд на відповідь
QUICK_TEST_SAMPLES=10  # Кількість прикладів для швидкого тесту

# Функція для швидкого тесту
quick_test() {
    local model_path="$1"
    echo ""
    echo "[TEST] Запуск швидкого тестування..."
    
    python -c "
import sys
sys.path.insert(0, '.')
from train.test_runner import quick_test_model, print_test_results
import os

model_path = '$model_path'
device = os.environ.get('DEVICE', 'cpu')

results = quick_test_model(
    model_path=model_path,
    num_samples=$QUICK_TEST_SAMPLES,
    device=device,
    show_progress=True
)

print_test_results(results, 'quick')
"
}

# Функція для повного тесту
full_test() {
    local model_path="$1"
    echo ""
    echo "[TEST] Запуск повного тестування..."
    
    python scripts/test_model.py --model "$model_path" --device "${DEVICE:-cpu}"
}

# Функція для моніторингу checkpoint'ів
monitor_checkpoints() {
    local last_checkpoint=""
    local last_mtime=0
    
    while true; do
        # Перевірити чи процес навчання ще працює
        if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
            # Процес завершився
            break
        fi
        
        # Знайти останній checkpoint (за часом модифікації)
        local latest_checkpoint=""
        local latest_mtime=0
        
        # Перевірити всі checkpoint файли
        for checkpoint in "$CHECKPOINT_DIR"/*.pt "$CHECKPOINT_DIR"/*.ckpt 2>/dev/null; do
            if [ -f "$checkpoint" ]; then
                local mtime=$(stat -c %Y "$checkpoint" 2>/dev/null || echo 0)
                if [ "$mtime" -gt "$latest_mtime" ]; then
                    latest_mtime=$mtime
                    latest_checkpoint="$checkpoint"
                fi
            fi
        done
        
        # Якщо знайдено новий checkpoint (новіший за останній)
        if [ -n "$latest_checkpoint" ] && [ "$latest_mtime" -gt "$last_mtime" ]; then
            last_checkpoint="$latest_checkpoint"
            last_mtime=$latest_mtime
            
            echo ""
            echo "=========================================="
            echo "[CHECKPOINT] Знайдено новий checkpoint!"
            echo "   Файл: $(basename "$latest_checkpoint")"
            echo "=========================================="
            
            # Запитати користувача
            echo ""
            echo "Протестувати модель? (y/n/f/q)"
            echo "  y - швидкий тест ($QUICK_TEST_SAMPLES прикладів)"
            echo "  f - повний тест"
            echo "  n/q - пропустити"
            echo ""
            read -t $TEST_INTERVAL -p "Відповідь (timeout ${TEST_INTERVAL}с): " answer || answer="n"
            
            case "$answer" in
                y|Y|yes|Yes)
                    quick_test "$latest_checkpoint"
                    ;;
                f|F|full|Full)
                    full_test "$latest_checkpoint"
                    ;;
                n|N|no|No|q|Q|"")
                    echo "[SKIP] Тестування пропущено, продовжуємо навчання..."
                    ;;
                *)
                    echo "[SKIP] Невідома відповідь, продовжуємо навчання..."
                    ;;
            esac
            
            echo ""
            echo "[TRAINING] Навчання продовжується..."
        fi
        
        # Перевіряти кожні 5 секунд
        sleep 5
    done
}

# Перевірка наявності checkpoint'у для продовження
MODE="new"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" ] || [ -n "$(find "$CHECKPOINT_DIR" -name "ckpt_step_*.pt" 2>/dev/null | head -n 1)" ]; then
    echo "[CHECKPOINT] Знайдено checkpoint для продовження"
    read -p "   Продовжити з checkpoint'у? (y/N): " resume
    if [[ "$resume" =~ ^[Yy]$ ]]; then
        MODE="resume"
        echo "   [OK] Буде продовжено з checkpoint'у"
    fi
fi

# Створити папку logs якщо не існує
mkdir -p logs

# Генерувати ім'я лог-файлу
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"
LATEST_LOG="logs/training_latest.log"

# Запуск навчання в фоні
echo ""
echo "[START] Запуск навчання..."
echo "   Скрипт: runtime.bootstrap"
echo "   Логи: $LOG_FILE"
echo "   Checkpoint'и: $CHECKPOINT_DIR/"
echo ""

# Створити папку checkpoints якщо не існує
mkdir -p "$CHECKPOINT_DIR"

# Запустити навчання в фоні
python -m runtime.bootstrap --mode "$MODE" --config config/config.yaml > "$LOG_FILE" 2>&1 &
TRAINING_PID=$!

echo "[INFO] Навчання запущено (PID: $TRAINING_PID)"
echo "[INFO] Моніторинг checkpoint'ів активовано"
echo "[INFO] Після кожного checkpoint буде пропозиція протестувати модель"
echo ""

# Запустити моніторинг checkpoint'ів в окремому процесі
monitor_checkpoints &
MONITOR_PID=$!

# Функція cleanup
cleanup() {
    echo ""
    echo "[STOP] Зупинка процесів..."
    kill $TRAINING_PID 2>/dev/null || true
    kill $MONITOR_PID 2>/dev/null || true
    wait $TRAINING_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    echo "[OK] Процеси зупинено"
    exit 0
}

# Обробка сигналів
trap cleanup SIGINT SIGTERM

# Чекати завершення навчання
if wait $TRAINING_PID 2>/dev/null; then
    TRAINING_EXIT=0
else
    TRAINING_EXIT=$?
fi

# Зупинити моніторинг
if kill $MONITOR_PID 2>/dev/null; then
    wait $MONITOR_PID 2>/dev/null || true
fi

if [ $TRAINING_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[OK] НАВЧАННЯ ЗАВЕРШЕНО"
    echo "=========================================="
else
    echo ""
    echo "[ERROR] Помилка під час навчання"
    echo "   Перевірте логи: $LOG_FILE"
    exit $TRAINING_EXIT
fi

