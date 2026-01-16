#!/bin/bash
# Скрипт моніторингу навчання (уніфікований)

cd "$(dirname "$0")"

INTERVAL=${1:-30}  # За замовчуванням 30 секунд

# Якщо передано "once", показати один раз
if [ "$1" = "once" ]; then
    INTERVAL=0
fi

# Знайти PID процесу навчання (підтримка обох архітектур)
TRAIN_PID=$(pgrep -f "train_model.py\|runtime.bootstrap" | head -1)

# Знайти лог-файл
LATEST_LOG=""
# Спочатку шукаємо символічне посилання
if [ -f "logs/training_latest.log" ]; then
    LATEST_LOG="logs/training_latest.log"
# Потім шукаємо останній лог з timestamp
elif [ -d "logs" ]; then
    LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
fi

while true; do
    clear
    echo "📊 Моніторинг навчання - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Оновити PID якщо потрібно
    if [ -z "$TRAIN_PID" ] || ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        TRAIN_PID=$(pgrep -f "train_model.py\|runtime.bootstrap" | head -1)
    fi
    
    if [ -n "$TRAIN_PID" ] && ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo "✅ Навчання працює (PID: $TRAIN_PID)"
        echo ""
        
        # Інформація про процес
        ps -p "$TRAIN_PID" -o pid,etime,pcpu,pmem,vsz,rss,cmd --no-headers 2>/dev/null | awk '{
            printf "   Час: %s | CPU: %s%% | Память: %s%% (%s MB)\n", $2, $3, $4, int($6/1024)
        }'
        echo ""
        
        # Checkpoint'и
        if [ -d "checkpoints" ]; then
            echo "💾 Останні checkpoint'и:"
            ls -lht checkpoints/*.pt 2>/dev/null | head -3 | awk '{printf "   %s (%s) - %s\n", $9, $5, $6" "$7" "$8}'
            echo ""
        fi
        
        # Прогрес з логу або checkpoint
        PROGRESS_LINE=""
        TQDM_LINE=""
        
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            # Спочатку шукаємо детальний прогрес з 📊
            PROGRESS_LINE=$(grep "📊 Прогрес" "$LATEST_LOG" 2>/dev/null | tail -1)
            # Якщо немає, шукаємо tqdm прогрес-бар
            if [ -z "$PROGRESS_LINE" ]; then
                TQDM_LINE=$(grep -E "Епоха [0-9]+/[0-9]+:|Навчання:" "$LATEST_LOG" 2>/dev/null | tail -1)
            fi
        fi
        
        if [ -z "$PROGRESS_LINE" ] && [ -z "$TQDM_LINE" ] && [ -f "checkpoints/checkpoint_latest.pt" ]; then
            PROGRESS_LINE=$(./venv/bin/python3 -c "
import torch
from pathlib import Path
try:
    checkpoint = torch.load('checkpoints/checkpoint_latest.pt', map_location='cpu')
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    batch_count = checkpoint.get('batch_count', 0)
    epochs = checkpoint.get('epochs', 0)
    loss = checkpoint.get('loss', None)
    
    total_batches_in_dataset = checkpoint.get('total_batches_per_epoch', 1800)
    if total_batches_in_dataset == 0:
        total_batches_in_dataset = 1800
    
    if epochs > 0:
        total_batches = epochs * total_batches_in_dataset
        progress_pct = (batch_count / total_batches) * 100 if total_batches > 0 else 0
        loss_str = f'{loss:.3f}' if loss is not None else 'N/A'
        current_epoch_batch = batch_idx + 1
        epoch_progress_pct = (current_epoch_batch / total_batches_in_dataset) * 100 if total_batches_in_dataset > 0 else 0
        print(f'📊 Прогрес: {batch_count}/{total_batches} батчів ({progress_pct:.1f}%) | Епоха: {epoch}/{epochs} ({epoch_progress_pct:.1f}%) | Батч: {current_epoch_batch}/{total_batches_in_dataset} | Loss: {loss_str}')
except:
    pass
" 2>/dev/null)
        fi
        
        echo "📝 Прогрес навчання:"
        if [ -n "$PROGRESS_LINE" ]; then
            echo "   $PROGRESS_LINE"
        elif [ -n "$TQDM_LINE" ]; then
            CLEAN_LINE=$(echo "$TQDM_LINE" | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
            echo "   📈 $CLEAN_LINE"
        elif [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            echo "   Останні рядки логу:"
            tail -2 "$LATEST_LOG" 2>/dev/null | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^/      /'
        fi
        
        # Показати попередження якщо є
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            WARNING_LINE=$(grep "⚠️ УВАГА:" "$LATEST_LOG" 2>/dev/null | tail -1)
            if [ -n "$WARNING_LINE" ]; then
                CLEAN_WARNING=$(echo "$WARNING_LINE" | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
                echo ""
                echo "   ⚠️  Попередження: $CLEAN_WARNING"
            fi
        fi
    else
        echo "❌ Навчання не працює"
        echo ""
        
        # Перевірити чи є checkpoint'и
        if [ -f "checkpoints/checkpoint_latest.pt" ]; then
            echo "💾 Знайдено checkpoint для продовження:"
            CHECKPOINT_SIZE=$(du -h checkpoints/checkpoint_latest.pt 2>/dev/null | cut -f1)
            CHECKPOINT_TIME=$(stat -c %y checkpoints/checkpoint_latest.pt 2>/dev/null | cut -d'.' -f1)
            echo "   Файл: checkpoints/checkpoint_latest.pt ($CHECKPOINT_SIZE)"
            echo "   Час: $CHECKPOINT_TIME"
        fi
    fi
    
    echo ""
    echo "=========================================="
    
    # Якщо "once", вийти
    if [ "$INTERVAL" = "0" ]; then
        break
    fi
    
    echo "Оновлення через $INTERVAL секунд... (Ctrl+C для виходу)"
    sleep "$INTERVAL"
done

