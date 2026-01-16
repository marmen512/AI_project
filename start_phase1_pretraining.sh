#!/bin/bash
# Запуск ФАЗИ 1 - Language Pretraining
# Навчання кастомної Transformer моделі з нуля на plain text

set -euo pipefail  # Зупинити при помилці, невизначених змінних, помилках у pipe

echo "🚀 ФАЗА 1 - Language Pretraining"
echo "=================================="

# Перевірити віртуальне середовище
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Активуйте віртуальне середовище:"
    echo "   source venv-linux/bin/activate"
    exit 1
fi

# Перевірити наявність plain text corpus
if [[ ! -f "datasets/pretrain_text.txt" ]]; then
    echo "❌ Plain text corpus не знайдено!"
    echo "   Запустіть: python scripts/prepare_phase1_dataset.py"
    exit 1
fi

# Показати інформацію про dataset
DATASET_FILE="datasets/pretrain_text.txt"
FILE_SIZE=$(stat -c%s "$DATASET_FILE")

# Перевірити розмір файлу
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "❌ Dataset файл занадто малий або порожній"
    echo "   Розмір: $FILE_SIZE байт (потрібно мінімум 1MB)"
    exit 1
fi

# Додаткова перевірка: чи є англійський текст
if ! head -n 5 "$DATASET_FILE" | grep -qi "[a-z]"; then
    echo "❌ Dataset не містить англійського тексту"
    echo "   Перевірте файл: $DATASET_FILE"
    exit 1
fi

# Підрахувати слова та символи (для plain text це більш релевантно)
WORDS=$(wc -w < "$DATASET_FILE")
CHARS=$(wc -c < "$DATASET_FILE")

echo "📊 Інформація про dataset:"
echo "   Файл: $DATASET_FILE"
echo "   Розмір: $((FILE_SIZE / 1024 / 1024))MB ($FILE_SIZE байт)"
echo "   Символів: $CHARS"
echo "   Слів: $WORDS"
echo "   Формат: Plain text (один довгий рядок для language modeling)"

# Показати конфігурацію
echo ""
echo "📋 Конфігурація ФАЗИ 1:"
echo "   Config: config/phase1_pretraining.yaml"
echo "   Model: Small Transformer (~15-25M params)"
echo "   Objective: Causal Language Modeling"
echo "   Initialization: Random (БЕЗ pretrained weights)"
echo "   Max epochs: 3"

# Підтвердження
echo ""
read -p "🤔 Почати ФАЗУ 1? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Скасовано"
    exit 1
fi

# Створити необхідні папки
mkdir -p checkpoints/phase1
mkdir -p logs/phase1

echo ""
echo "🎯 Запуск навчання ФАЗИ 1..."
echo "   Це може зайняти кілька годин на CPU!"
echo "   Моніторинг: tail -f logs/phase1/phase1_pretraining.log"

# Запустити навчання
# Запустити навчання
python scripts/train_phase1_pretraining.py \
    --config config/phase1_pretraining.yaml || {
    ERR_CODE=$?
    echo ""
    echo "⚠️  Training interrupted/failed with code $ERR_CODE"
    exit $ERR_CODE
}

echo ""
echo "✅ ФАЗА 1 завершена!"
echo "   Модель збережена: checkpoints/phase1/best_model.pt"
echo ""
echo "🔄 Наступний крок: ФАЗА 2 - Instruction Tuning"
echo "   Запустіть: ./start_phase2_instruction_tuning.sh"
