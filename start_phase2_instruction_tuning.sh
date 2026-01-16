#!/bin/bash
# Запуск ФАЗИ 2 - Instruction Tuning
# Навчання моделі після ФАЗИ 1 на instruction datasets

set -e  # Зупинити при помилці
set -o pipefail

echo "🚀 ФАЗА 2 - Instruction Tuning"
echo "==============================="

# Environment safety defaults
export LC_ALL=${LC_ALL:-C.UTF-8}
export LANG=${LANG:-C.UTF-8}

# Safer CPU threading defaults (can be overridden)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-6}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-6}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-6}

# Ensure immediate logs without stdbuf wrapper
export PYTHONUNBUFFERED=1

# Перевірити/активувати віртуальне середовище
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -f "venv-linux/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source venv-linux/bin/activate
    fi
fi

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Активуйте віртуальне середовище:"
    echo "   source venv-linux/bin/activate"
    exit 1
fi

# Перевірити наявність моделі з ФАЗИ 1
PHASE1_MODEL="checkpoints/phase1/best_model.pt"
if [[ ! -f "$PHASE1_MODEL" ]]; then
    echo "❌ Модель з ФАЗИ 1 не знайдена!"
    echo "   Очікується: $PHASE1_MODEL"
    echo "   Спочатку завершіть ФАЗУ 1: ./start_phase1_pretraining.sh"
    exit 1
fi

# Перевірити instruction datasets
echo "📚 Перевірка instruction datasets:"
DATASETS_FOUND=0

for dataset in "datasets/alpaca.json" "datasets/squad.json" "datasets/squad_v2.json" "datasets/dailydialog_minimal.json"; do
    if [[ -f "$dataset" ]]; then
        echo "   ✅ $dataset"
        DATASETS_FOUND=$((DATASETS_FOUND + 1))
    else
        echo "   ❌ $dataset (не знайдено)"
    fi
done

if [[ $DATASETS_FOUND -eq 0 ]]; then
    echo "❌ Жоден instruction dataset не знайдено!"
    echo "   Перевірте папку datasets/"
    exit 1
fi

echo "   📊 Знайдено $DATASETS_FOUND instruction datasets"

# Показати інформацію про модель з ФАЗИ 1
echo ""
echo "🔗 Модель з ФАЗИ 1:"
echo "   Файл: $PHASE1_MODEL"

# Надійна перевірка розміру файлу
if [ -f "$PHASE1_MODEL" ]; then
    MODEL_SIZE=$(stat -c%s "$PHASE1_MODEL")
    echo "   Розмір: $((MODEL_SIZE / 1024 / 1024))MB ($MODEL_SIZE байт)"
    
    # Перевірити що це дійсно checkpoint файл
    if [ "$MODEL_SIZE" -lt 1000000 ]; then
        echo "⚠️  Модель занадто мала (< 1MB), можливо пошкоджена"
    fi
else
    echo "   ❌ ПОМИЛКА: Файл не існує!"
    exit 1
fi

# Показати конфігурацію
echo ""
echo "📋 Конфігурація ФАЗИ 2:"
echo "   Config: config/phase2_instruction_tuning.yaml"
echo "   Base model: З ФАЗИ 1 (pretrained на Simple Wikipedia)"
echo "   Objective: Instruction Following"
echo "   Max epochs: 1-2 (з early stopping)"
echo "   Datasets: Alpaca, SQuAD, DailyDialog"

# Створити необхідні папки
mkdir -p checkpoints/phase2
mkdir -p logs/phase2

echo ""
echo "🎯 Запуск instruction tuning ФАЗИ 2..."
echo "   Це займе менше часу ніж ФАЗА 1"
echo "   Моніторинг: tail -f logs/phase2/phase2_instruction_tuning.log"
echo "   Live stdout: tail -f logs/phase2/live_stdout.log"

# Запустити навчання
set +e
python -u scripts/train_phase2_instruction_tuning.py \
    --config config/phase2_instruction_tuning.yaml \
    --phase1-model "$PHASE1_MODEL" \
    2>&1 | tee -a logs/phase2/live_stdout.log
PY_STATUS=${PIPESTATUS[0]}
set -e

echo ""
if [[ $PY_STATUS -eq 0 ]]; then
    echo "✅ ФАЗА 2 завершена!"
    echo "   Фінальна модель: checkpoints/phase2/best_instruction_model.pt"
    echo ""
    echo "🎉 Двофазне навчання завершено!"
    echo "   📚 ФАЗА 1: Language pretraining ✅"
    echo "   🎯 ФАЗА 2: Instruction tuning ✅"
    echo ""
    echo "🤖 Модель готова для використання!"
    echo "   Тестування: python scripts/test_model.py --model checkpoints/phase2/best_instruction_model.pt"
elif [[ $PY_STATUS -eq 130 ]]; then
    echo "⚠️  ФАЗА 2 зупинена (Ctrl+C). Прогрес збережено."
    echo "   Resume: checkpoints/phase2/last_checkpoint.pt"
    exit 130
else
    echo "❌ ФАЗА 2 завершилась з помилкою (exit code: $PY_STATUS)"
    echo "   Перевір логи: logs/phase2/phase2_instruction_tuning.log"
    echo "   Live stdout: logs/phase2/live_stdout.log"
fi

exit $PY_STATUS
