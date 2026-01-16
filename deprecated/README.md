# Deprecated Files

Ця папка містить застарілі файли, які зберігаються для backwards compatibility, але не рекомендуються до використання.

## 🎯 Поточна архітектура

**Головний entry point:** `scripts/train_model.py` → `runtime.bootstrap`

**Config:** `config/config.yaml` (єдине джерело істини)

**Shell скрипти:** Використовують `runtime.bootstrap`

## ⚠️ Deprecated файли

### Config файли:

- `config/training_resume.py` - DEPRECATED
  - **Використовувати:** `runtime.resume.find_latest_checkpoint()`
  - **Залишено для:** backwards compatibility з `train/train.py`

- `config/training_config.py` - ЧАСТКОВО DEPRECATED
  - `AutoTrainingConfig` - DEPRECATED, використовувати `config.yaml`
  - `TrainingConfig` - може використовуватися, але рекомендується `config.yaml`

- `config/training_defaults.sh` - DEPRECATED для runtime.bootstrap
  - Використовується тільки в `start_training.sh` (старий спосіб)
  - `runtime.bootstrap` читає `config.yaml` напряму

### Entry points:

- `train/train.py` - DEPRECATED як прямий entry point
  - Функція `train_with_auto_config()` може використовуватися старим кодом
  - **Використовувати:** `runtime.bootstrap` через `scripts/train_model.py`

- `train/train_code_model.py` - DEPRECATED
  - Не використовується runtime.bootstrap
  - Можна заархівувати

- `train/train_trm_with_phi3.py` - DEPRECATED
  - Не використовується runtime.bootstrap
  - Можна заархівувати

## ✅ Рекомендації

1. **Для навчання:** Використовуйте `python scripts/train_model.py --config config/config.yaml`
2. **Для config:** Всі параметри в `config/config.yaml`
3. **Для resume:** Використовуйте `runtime.resume.find_latest_checkpoint()`

## 📅 План міграції

Файли в цій папці будуть видалені в майбутніх версіях після повної міграції на `runtime.bootstrap`.

