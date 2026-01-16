#!/bin/bash
# Скрипт для швидкого запуску навчання "Школа" на Linux
# Використання: ./start_school.sh

set -e

# Перейти в директорію скрипта
cd "$(dirname "$0")"

echo "=========================================="
echo "[START] ЗАПУСК НАВЧАННЯ 'ШКОЛА'"
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

if [ $? -ne 0 ]; then
    echo "[ERROR] Помилка активації venv"
    exit 1
fi

# Параметри за замовчуванням
DAYS=${1:-100}
TEACHER_MODEL=${2:-"gpt2"}
DEVICE=${3:-"cpu"}

echo "[INFO] Параметри:"
echo "   Днів навчання: $DAYS"
echo "   Модель вчительки: $TEACHER_MODEL"
echo "   Пристрій: $DEVICE"
echo ""

# Створити папку для логів якщо не існує
mkdir -p logs
mkdir -p school_progress

# Генерувати ім'я лог-файлу
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/school_${TIMESTAMP}.log"
LATEST_LOG="logs/school_latest.log"

# Запуск навчання
echo "[START] Запуск навчання 'Школа'..."
echo "   Скрипт: scripts/school.py"
echo "   Логи: $LOG_FILE"
echo ""

python scripts/school.py \
    --days "$DAYS" \
    --teacher-model "$TEACHER_MODEL" \
    --device "$DEVICE" \
    --config config/config.yaml \
    2>&1 | tee "$LOG_FILE" | tee "$LATEST_LOG"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[OK] НАВЧАННЯ 'ШКОЛА' ЗАВЕРШЕНО"
    echo "=========================================="
else
    echo ""
    echo "[ERROR] Помилка під час навчання"
    echo "   Перевірте логи: $LOG_FILE"
    exit 1
fi

