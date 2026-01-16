#!/bin/bash
# Скрипт для запуску навчання моделі на Linux
# Використання: ./start_training.sh

set -e  # Зупинитися при помилці

# Перейти в директорію скрипта
cd "$(dirname "$0")"

echo "=========================================="
echo "[START] ЗАПУСК НАВЧАННЯ МОДЕЛІ"
echo "=========================================="
echo ""

# Перевірка наявності venv
if [ ! -d "venv" ]; then
    echo "[ERROR] Помилка: venv не знайдено"
    echo "   Створіть venv: ./setup_venv.sh"
    exit 1
fi

# Активувати venv
echo "[ACTIVATE] Активування віртуального середовища..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Помилка активації venv"
    exit 1
fi

# Перевірка чи вже працює навчання
if pgrep -f "runtime.bootstrap" > /dev/null; then
    echo "[WARN] Навчання вже запущено!"
    PID=$(pgrep -f "runtime.bootstrap" | head -n 1)
    echo "   PID: $PID"
    read -p "   Зупинити і запустити нове? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   [STOP] Зупиняю попереднє навчання..."
        kill -9 $PID 2>/dev/null || true
        sleep 2
    else
        echo "   [CANCEL] Скасовано"
        exit 0
    fi
fi

# Перевірка наявності checkpoint'у для продовження
MODE="new"
if [ -f "checkpoints/checkpoint_latest.pt" ] || [ -n "$(ls -A checkpoints/ckpt_step_*.pt 2>/dev/null)" ]; then
    echo "[CHECKPOINT] Знайдено checkpoint для продовження"
    read -p "   Продовжити з checkpoint'у? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        MODE="resume"
        echo "   [OK] Буде продовжено з checkpoint'у"
    fi
fi

# Створити папку logs якщо не існує
mkdir -p logs

# Генерувати ім'я лог-файлу з timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"
LATEST_LOG="logs/training_latest.log"

# Запуск навчання
echo ""
echo "[START] Запуск навчання..."
echo "   Скрипт: runtime.bootstrap"
echo "   Логи зберігаються в: $LOG_FILE"
echo "   Checkpoint'и зберігаються в: checkpoints/"
echo ""

# Використовувати runtime.bootstrap
python -m runtime.bootstrap --mode "$MODE" --config config/config.yaml 2>&1 | tee "$LOG_FILE" | tee "$LATEST_LOG"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[OK] НАВЧАННЯ ЗАВЕРШЕНО"
    echo "=========================================="
else
    echo ""
    echo "[ERROR] Помилка під час навчання"
    echo "   Перевірте логи: $LOG_FILE"
    exit $EXIT_CODE
fi
