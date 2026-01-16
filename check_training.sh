#!/bin/bash
# Скрипт перевірки статусу навчання

cd "$(dirname "$0")"

echo "🔍 Перевірка статусу навчання..."
echo ""

# Перевірити чи працює навчання (підтримка обох архітектур)
if pgrep -f "train_model.py\|runtime.bootstrap" > /dev/null; then
    echo "✅ Навчання працює!"
    echo ""
    
    # Показати PID процесів
    echo "📋 Процеси навчання:"
    pgrep -f "train_model.py\|runtime.bootstrap" | while read pid; do
        ps -p "$pid" -o pid,etime,pcpu,pmem,cmd --no-headers | awk '{printf "   PID: %s | Час: %s | CPU: %s%% | Память: %s%%\n   Команда: %s\n", $1, $2, $3, $4, substr($0, index($0,$5))}'
    done
    echo ""
    
    # Перевірити checkpoint'и
    if [ -d "checkpoints" ] && [ -f "checkpoints/checkpoint_latest.pt" ]; then
        CHECKPOINT_SIZE=$(du -h checkpoints/checkpoint_latest.pt 2>/dev/null | cut -f1)
        CHECKPOINT_TIME=$(stat -c %y checkpoints/checkpoint_latest.pt 2>/dev/null | cut -d'.' -f1)
        echo "💾 Останній checkpoint:"
        echo "   Файл: checkpoints/checkpoint_latest.pt ($CHECKPOINT_SIZE)"
        echo "   Час: $CHECKPOINT_TIME"
    fi
    
    # Перевірити логи
    LATEST_LOG=""
    if [ -f "logs/training_latest.log" ]; then
        LATEST_LOG="logs/training_latest.log"
    elif [ -d "logs" ]; then
        LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    fi
    
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        LAST_LINE=$(tail -1 "$LATEST_LOG" 2>/dev/null | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
        echo ""
        echo "📝 Останній рядок логу ($(basename "$LATEST_LOG")):"
        echo "   $LAST_LINE"
    fi
else
    echo "❌ Навчання не працює"
    echo ""
    
    # Перевірити чи є checkpoint'и
    if [ -f "checkpoints/checkpoint_latest.pt" ]; then
        echo "💾 Знайдено checkpoint для продовження:"
        echo "   checkpoints/checkpoint_latest.pt"
    fi
fi

echo ""
echo "=========================================="

