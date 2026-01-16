# Логи навчання

Ця папка містить логи навчання моделі.

## Структура

- `training_YYYYMMDD_HHMMSS.log` - логи навчання з timestamp
- `training_latest.log` - символічне посилання на останній лог
- `training_metrics_*.json` - метрики навчання (зберігаються через train/logging.py)
- `monitoring_*.log` - логи моніторингу (створюються monitor_training.sh)

## Використання

Переглянути останній лог:
```bash
tail -f logs/training_latest.log
```

Переглянути всі логи:
```bash
ls -lht logs/training_*.log
```

