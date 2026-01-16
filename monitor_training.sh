#!/bin/bash
# Скрипт моніторингу навчання з детальним логуванням та виявленням проблем

cd "$(dirname "$0")" || exit 1

# Створити папки якщо не існують
mkdir -p temp
mkdir -p logs

TRAIN_PID_FILE="temp/train_pid.txt"
LOG_DIR="logs"
CHECK_INTERVAL=30  # Перевіряти кожні 30 секунд

# Лог моніторингу
MONITORING_LOG="$LOG_DIR/monitoring_$(date +%Y%m%d_%H%M%S).log"

# Спробувати знайти PID процесу навчання
TRAIN_PID=""
if [ -f "$TRAIN_PID_FILE" ]; then
    TRAIN_PID=$(cat "$TRAIN_PID_FILE" 2>/dev/null)
fi

# Якщо PID файл не знайдено, спробувати знайти процес через pgrep
if [ -z "$TRAIN_PID" ] || ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    TRAIN_PID=$(pgrep -f "train_model.py\|runtime.bootstrap" | head -1)
fi

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Процес навчання не знайдено"
    echo "   Перевірте чи запущено навчання: ./start_training.sh"
    exit 1
fi

# Знайти лог-файл
LATEST_LOG=""
# Спочатку шукаємо символічне посилання
if [ -f "$LOG_DIR/training_latest.log" ]; then
    LATEST_LOG="$LOG_DIR/training_latest.log"
# Потім шукаємо останній лог з timestamp
elif [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
fi

# НЕ використовуємо training_service_error.log як основний лог
# (це тільки для помилок, не для прогресу)

echo "📊 Моніторинг навчання (PID: $TRAIN_PID)"
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "📝 Лог файл: $LATEST_LOG"
else
    echo "📝 Лог файл: не знайдено (шукаю в logs/training_*.log)"
fi
echo "📋 Лог моніторингу: $MONITORING_LOG"
echo "=========================================="

# Записати початок моніторингу
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Початок моніторингу (PID: $TRAIN_PID)" >> "$MONITORING_LOG"

while true; do
    if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo ""
        echo "⚠️  Процес завершено!"
        echo "Перевіряю причину..."
        
        if [ -f "$LATEST_LOG" ]; then
            # Перевірити чи була OOM помилка
            if grep -q "killed\|OOM\|out of memory\|MemoryError" "$LATEST_LOG"; then
                echo "❌ Виявлено OOM помилку!" | tee -a "$MONITORING_LOG"
                echo "   Деталі:" | tee -a "$MONITORING_LOG"
                grep -i "killed\|OOM\|out of memory\|MemoryError" "$LATEST_LOG" | tail -5 | tee -a "$MONITORING_LOG"
                echo "   Час: $(date)" >> "$MONITORING_LOG"
                exit 1
            fi
            
            # Перевірити інші помилки
            if grep -q "Error\|Exception\|Traceback" "$LATEST_LOG"; then
                echo "❌ Виявлено помилку!" | tee -a "$MONITORING_LOG"
                echo "   Останні 50 рядків логу:" | tee -a "$MONITORING_LOG"
                tail -50 "$LATEST_LOG" | tee -a "$MONITORING_LOG"
                echo "   Час: $(date)" >> "$MONITORING_LOG"
                exit 1
            fi
            
            # Якщо завершено успішно
            if grep -q "НАВЧАННЯ ЗАВЕРШЕНО\|завершено" "$LATEST_LOG"; then
                echo "✅ Навчання завершено успішно!" | tee -a "$MONITORING_LOG"
                tail -20 "$LATEST_LOG" | tee -a "$MONITORING_LOG"
                echo "   Час: $(date)" >> "$MONITORING_LOG"
                exit 0
            fi
        fi
        
        echo "❌ Процес завершено невідомою причиною" | tee -a "$MONITORING_LOG"
        echo "   Час: $(date)" >> "$MONITORING_LOG"
        if [ -f "$LATEST_LOG" ]; then
            echo "   Останні рядки логу:" >> "$MONITORING_LOG"
            tail -20 "$LATEST_LOG" >> "$MONITORING_LOG"
        fi
        exit 1
    fi
    
    # Перевірити використання пам'яті та CPU
    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        MEM_USAGE=$(ps -p "$TRAIN_PID" -o %mem --no-headers 2>/dev/null | tr -d ' ')
        CPU_USAGE=$(ps -p "$TRAIN_PID" -o %cpu --no-headers 2>/dev/null | tr -d ' ')
        ETIME=$(ps -p "$TRAIN_PID" -o etime --no-headers 2>/dev/null | tr -d ' ')
        RSS_MB=$(ps -p "$TRAIN_PID" -o rss --no-headers 2>/dev/null | awk '{print int($1/1024)}')
        
        # Перевірити чи оновлюється лог (виявлення підвисань)
        LAST_LOG_UPDATE=""
        if [ -f "$LATEST_LOG" ]; then
            if command -v stat > /dev/null 2>&1; then
                # Linux
                LAST_LOG_UPDATE=$(stat -c %Y "$LATEST_LOG" 2>/dev/null)
            else
                # macOS/BSD
                LAST_LOG_UPDATE=$(stat -f %m "$LATEST_LOG" 2>/dev/null)
            fi
            if [ -n "$LAST_LOG_UPDATE" ]; then
                CURRENT_TIME=$(date +%s)
                TIME_SINCE_UPDATE=$((CURRENT_TIME - LAST_LOG_UPDATE))
                
                # Якщо лог не оновлювався більше 5 хвилин - можливе зависання
                if [ "$TIME_SINCE_UPDATE" -gt 300 ]; then
                    echo ""
                    echo "⚠️  УВАГA: Лог не оновлювався ${TIME_SINCE_UPDATE} секунд (> 5 хв)!" | tee -a "$MONITORING_LOG"
                    echo "   Можливе зависання процесу!" | tee -a "$MONITORING_LOG"
                    if command -v date > /dev/null 2>&1; then
                        if [ "$(uname)" = "Linux" ]; then
                            LAST_UPDATE_STR=$(date -d "@$LAST_LOG_UPDATE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                        else
                            LAST_UPDATE_STR=$(date -r "$LAST_LOG_UPDATE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                        fi
                        echo "   Останнє оновлення: $LAST_UPDATE_STR" | tee -a "$MONITORING_LOG"
                    fi
                fi
            fi
        fi
        
        # Перевірити обмеження ресурсів
        if [ -n "$CPU_USAGE" ] && [ -n "$MEM_USAGE" ]; then
            # Використати awk для порівняння (працює на всіх системах)
            CPU_CHECK=$(echo "$CPU_USAGE" | awk '{if ($1 > 95) print "high"}')
            MEM_CHECK=$(echo "$MEM_USAGE" | awk '{if ($1 > 90) print "high"}')
            
            if [ "$CPU_CHECK" = "high" ]; then
                echo "⚠️  ВИСОКЕ ВИКОРИСТАННЯ CPU: ${CPU_USAGE}%" | tee -a "$MONITORING_LOG"
            fi
            
            if [ "$MEM_CHECK" = "high" ]; then
                echo "⚠️  ВИСОКЕ ВИКОРИСТАННЯ ПАМ'ЯТІ: ${MEM_USAGE}% (${RSS_MB} MB)" | tee -a "$MONITORING_LOG"
            fi
        fi
        
        # Форматований вивід статусу
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🕐 $(date '+%H:%M:%S') | CPU: ${CPU_USAGE}% | RAM: ${MEM_USAGE}% (${RSS_MB} MB) | Час роботи: ${ETIME}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Показати останній прогрес з логу
        # Спробувати знайти лог-файл знову (на випадок якщо він створився)
        if [ -z "$LATEST_LOG" ] || [ ! -f "$LATEST_LOG" ]; then
            if [ -f "$LOG_DIR/training_latest.log" ]; then
                LATEST_LOG="$LOG_DIR/training_latest.log"
            elif [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
            fi
        fi
        
        PROGRESS_LINE=""
        TQDM_LINE=""
        
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            # Шукати прогрес тільки в правильному лог-файлі (не в error log)
            if [[ "$LATEST_LOG" != *"error"* ]] && [[ "$LATEST_LOG" != *"service"* ]]; then
                # Спочатку шукаємо детальний прогрес з 📊
                PROGRESS_LINE=$(grep "📊 Прогрес" "$LATEST_LOG" 2>/dev/null | tail -1)
                if [ -n "$PROGRESS_LINE" ]; then
                    echo "   $PROGRESS_LINE"
                else
                    # Шукаємо tqdm прогрес-бар (формат: Епоха X/Y: Z%|...| N/TOTAL [...])
                    TQDM_LINE=$(grep -E "Епоха [0-9]+/[0-9]+:|Навчання:" "$LATEST_LOG" 2>/dev/null | tail -1)
                    if [ -n "$TQDM_LINE" ]; then
                        # Видалити зайві символи та показати чисто
                        CLEAN_LINE=$(echo "$TQDM_LINE" | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
                        echo "   📈 $CLEAN_LINE"
                    else
                        # Показати останній рядок з прогресом
                        LAST_LINE=$(tail -1 "$LATEST_LOG" 2>/dev/null | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
                        if [ -n "$LAST_LINE" ]; then
                            # Показати тільки якщо це не просто порожній рядок
                            if [[ "$LAST_LINE" =~ (Епоха|batch|Прогрес|Loss) ]]; then
                                echo "   📝 ${LAST_LINE:0:120}"
                            fi
                        fi
                    fi
                fi
            fi
        fi
        
        # Якщо прогрес не знайдено в логах, спробувати прочитати з checkpoint
        if [ -z "$TQDM_LINE" ] && [ -z "$PROGRESS_LINE" ] && [ -f "checkpoints/checkpoint_latest.pt" ]; then
            CHECKPOINT_INFO=$(./venv/bin/python3 -c "
import torch
from pathlib import Path
try:
    checkpoint = torch.load('checkpoints/checkpoint_latest.pt', map_location='cpu')
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    batch_count = checkpoint.get('batch_count', 0)
    epochs = checkpoint.get('epochs', 0)
    loss = checkpoint.get('loss', None)
    
    # Отримати кількість батчів на епоху з checkpoint або з dataloader
    total_batches_in_dataset = checkpoint.get('total_batches_per_epoch', 1800)
    if total_batches_in_dataset == 0:
        total_batches_in_dataset = 1800  # Значення за замовчуванням
    
    if epochs > 0:
        total_batches = epochs * total_batches_in_dataset
        progress_pct = (batch_count / total_batches) * 100 if total_batches > 0 else 0
        loss_str = f'{loss:.3f}' if loss is not None else 'N/A'
        current_epoch_batch = batch_idx + 1
        epoch_progress_pct = (current_epoch_batch / total_batches_in_dataset) * 100 if total_batches_in_dataset > 0 else 0
        print(f'📊 Прогрес: {batch_count}/{total_batches} батчів ({progress_pct:.1f}%) | Епоха: {epoch}/{epochs} ({epoch_progress_pct:.1f}%) | Батч в епосі: {current_epoch_batch}/{total_batches_in_dataset} | Loss: {loss_str}')
except Exception as e:
    pass
" 2>/dev/null)
            if [ -n "$CHECKPOINT_INFO" ]; then
                echo "   $CHECKPOINT_INFO"
            fi
        fi
        
        # Показати попередження про зависання якщо вони є
        if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
            WARNING_LINE=$(grep "⚠️ УВАГА:" "$LATEST_LOG" 2>/dev/null | tail -1)
            if [ -n "$WARNING_LINE" ]; then
                CLEAN_WARNING=$(echo "$WARNING_LINE" | sed 's/\r//g' | sed 's/\x1b\[[0-9;]*m//g')
                echo "   ⚠️  $CLEAN_WARNING"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Попередження: $CLEAN_WARNING" >> "$MONITORING_LOG"
            fi
        fi
    else
        echo "[$(date +%H:%M:%S)] ⚠️  Процес не знайдено!"
        break
    fi
    
    sleep "$CHECK_INTERVAL"
done
