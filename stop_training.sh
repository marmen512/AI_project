#!/bin/bash
# Скрипт для зупинки навчання моделі

cd "$(dirname "$0")"

# Якщо передано "check" або "status", тільки показати статус
if [ "$1" = "check" ] || [ "$1" = "status" ]; then
    echo "🔍 Перевірка статусу навчання..."
    echo ""
    
    PIDS=$(pgrep -f "train_model.py\|runtime.bootstrap" 2>/dev/null)
    
    if [ -n "$PIDS" ]; then
        echo "✅ Навчання працює"
        echo ""
        echo "📋 Процеси:"
        for pid in $PIDS; do
            ps -p "$pid" -o pid,etime,pcpu,pmem,cmd --no-headers 2>/dev/null | awk '{
                printf "   PID: %s | Час: %s | CPU: %s%% | Память: %s%%\n   Команда: %s\n", $1, $2, $3, $4, substr($0, index($0,$5))
            }'
        done
    else
        echo "❌ Навчання не працює"
    fi
    
    exit 0
fi

echo "=========================================="
echo "🛑 ЗУПИНКА НАВЧАННЯ"
echo "=========================================="
echo ""

# Знайти всі процеси навчання (підтримка обох архітектур)
PIDS=$(pgrep -f "train_code_model.py\|train_model.py\|train_two_models\|runtime.bootstrap" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✅ Навчання не працює"
    exit 0
fi

echo "📋 Знайдено процеси навчання:"
for pid in $PIDS; do
    ps -p "$pid" -o pid,etime,pcpu,pmem,cmd --no-headers 2>/dev/null | awk '{
        printf "   PID: %s | Час: %s | CPU: %s%% | Память: %s%%\n   Команда: %s\n", $1, $2, $3, $4, substr($0, index($0,$5))
    }'
done

echo ""
read -p "⚠️  Зупинити ці процеси? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "   ❌ Скасовано"
    exit 0
fi

echo ""
echo "🛑 Зупиняю процеси..."

for pid in $PIDS; do
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "   Зупиняю PID: $pid"
        kill "$pid" 2>/dev/null
        
        # Чекати трохи
        sleep 1
        
        # Якщо не зупинився, вбити примусово
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "   Примусове завершення PID: $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    fi
done

sleep 2

# Перевірити результат
REMAINING=$(pgrep -f "train_code_model.py\|train_model.py\|train_two_models\|runtime.bootstrap" 2>/dev/null)

if [ -z "$REMAINING" ]; then
    echo ""
    echo "✅ Всі процеси зупинено успішно!"
    echo ""
    
    # Показати інформацію про checkpoint'и
    if [ -f "checkpoints/checkpoint_latest.pt" ]; then
        echo "💾 Останній checkpoint збережено:"
        echo "   checkpoints/checkpoint_latest.pt"
        echo "   Можна продовжити навчання пізніше"
    fi
else
    echo ""
    echo "⚠️  Деякі процеси не зупинилися:"
    for pid in $REMAINING; do
        echo "   PID: $pid"
    done
fi

echo ""
echo "=========================================="

