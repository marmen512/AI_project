# 🚀 GPT-2 Two-Phase Training System

Повна система для двофазного навчання GPT-2: Phase 1 (Language Pretraining) та Phase 2 (Instruction Tuning) з підтримкою Google Colab.

---

## 🎯 Особливості

- **Двофазне навчання**: Спочатку мовна модель, потім інструкційна
- **CPU-оптимізовано**: Працює на звичайних комп'ютерах без GPU
- **Google Colab Ready**: Повна інтеграція з Google Colab notebook
- **Безпечне навчання**: Gradient clipping, loss guard, sanity checks
- **Інтерактивний чат**: Готовий інтерфейс для тестування моделі

---

## 📁 Структура проекту

```
AI-27.12.2025-20_00/
├── README.md                    # Цей файл
├── requirements.txt             # Залежності Python
├── .gitignore                 # Виключення Git
├── colab_setup.ipynb          # 🆕 Google Colab notebook
├── GOOGLE_COLAB_GUIDE.md      # 🆕 Інструкція для Colab
├── GITHUB_SETUP.md           # 🆕 Гід по налаштуванню GitHub
├── config/                   # Конфігурації
│   ├── phase1_pretraining.yaml
│   ├── phase2_instruction_tuning.yaml
│   └── colab_phase2.yaml     # 🆕 Оптимізовано для Colab
├── scripts/                  # Скрипти навчання та тестування
│   ├── train_phase1_pretraining.py    # Phase 1: Language Pretraining
│   ├── train_phase2_instruction_tuning.py  # Phase 2: Instruction Tuning
│   ├── test_model.py                 # Тестування моделі
│   ├── chat.py                      # Інтерактивний чат
│   └── check_training_status.py      # Моніторинг статусу
├── datasets/                 # Датасети
│   ├── alpaca.json
│   ├── squad.json
│   └── dailydialog_minimal.json
├── checkpoints/              # Моделі (з gitignore)
├── logs/                    # Логи (з gitignore)
└── core/                    # Основні модулі
```

---

## 🚀 Швидкий старт

### Локальна машина

1. **Клонуйте репозиторій**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Налаштуйте віртуальне середовище**
   ```bash
   # Linux/Mac
   chmod +x setup_venv.sh
   ./setup_venv.sh
   source venv-linux/bin/activate
   
   # Windows
   setup_venv.bat
   venv\Scripts\activate
   ```

3. **Запустіть Phase 1 (Language Pretraining)**
   ```bash
   python scripts/train_phase1_pretraining.py \
     --config config/phase1_pretraining.yaml
   ```

4. **Запустіть Phase 2 (Instruction Tuning)**
   ```bash
   python scripts/train_phase2_instruction_tuning.py \
     --config config/phase2_instruction_tuning.yaml \
     --phase1-model checkpoints/phase1/best_model.pt
   ```

5. **Протестуйте модель**
   ```bash
   python scripts/test_model.py \
     --model checkpoints/phase2/best_instruction_model.pt \
     --prompt "What is 2+2?"
   ```

### Google Colab (🆕)

1. **Відкрийте в Colab**: [colab_setup.ipynb](colab_setup.ipynb)
2. **Виконуйте клітинки** послідовно
3. **Результати автоматично зберігаються** на Google Drive

Детальна інструкція: [GOOGLE_COLAB_GUIDE.md](GOOGLE_COLAB_GUIDE.md)

---

## 📋 Вимоги

- Python 3.8+
- PyTorch 2.9.1+
- Transformers 4.57.3+
- 8GB+ RAM для навчання
- CPU (GPU опціонально)

---

## 🔧 Конфігурація

### Phase 1: Language Pretraining
```yaml
model:
  n_embd: 320
  n_layer: 6
  n_head: 8
  n_positions: 256

training:
  batch_size: 4
  epochs: 3
  learning_rate: 1e-4
```

### Phase 2: Instruction Tuning
```yaml
training:
  batch_size: 4
  epochs: 1
  learning_rate: 5e-5
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
```

---

## 🎮 Використання

### Тестування моделі
```bash
python scripts/test_model.py \
  --model checkpoints/phase2/best_instruction_model.pt \
  --prompt "Your question here" \
  --max-tokens 50
```

### Інтерактивний чат
```bash
python scripts/chat.py \
  --model checkpoints/phase2/best_instruction_model.pt \
  --max-tokens 100
```

### Моніторинг навчання
```bash
python scripts/check_training_status.py
```

---

## 📊 Архітектура

### Phase 1: Language Pretraining
- **Модель**: GPT-2 архітектура (HuggingFace)
- **Мета**: Навчити базову мовну модель
- **Датасети**: Відкриті текстові корпуси
- **Результат**: `checkpoints/phase1/best_model.pt`

### Phase 2: Instruction Tuning
- **Модель**: Fine-tuning Phase 1 моделі
- **Мета**: Навчити виконувати інструкції
- **Датасети**: Alpaca, SQuAD, DailyDialog
- **Результат**: `checkpoints/phase2/best_instruction_model.pt`

### Ключові техніки
- **Gradient Clipping**: Запобігання вибуху градієнтів
- **Label Masking**: Тільки response токени для навчання
- **Loss Guard**: EMA моніторинг для раннього виявлення проблем
- **Sanity Checks**: Періодична перевірка якості генерації

---

## 🛠️ Налаштування для Google Colab

### Оптимізації
- **Batch Size**: 2 (замість 4)
- **Gradient Accumulation**: 4 (ефективний batch = 8)
- **CPU Threads**: 2 (обмеження Colab)
- **Checkpoint Interval**: 50 (частіше збереження)

### Автоматизація
- **Google Drive Backup**: Автоматичне збереження результатів
- **Resource Monitoring**: Моніторинг CPU/RAM
- **Session Recovery**: Відновлення після переривання

---

## 🔍 Troubleshooting

### Проблема: Out of Memory
```yaml
# Зменште в конфігурації
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Проблема: Повільне навчання
```yaml
# Зменште частоту логування
training:
  monitoring_interval: 50
  checkpoint_interval: 100
```

### Проблема: Погана якість генерації
- Перевірте learning rate (занадто високий)
- Перевірте label masking (може бути неправильним)
- Запустіть sanity checks для діагностики

---

## 📚 Документація

- [GOOGLE_COLAB_GUIDE.md](GOOGLE_COLAB_GUIDE.md) - Повна інструкція для Colab
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Налаштування GitHub репозиторію
- [TWO_PHASE_TRAINING_README.md](TWO_PHASE_TRAINING_README.md) - Детальна архітектура

---

## 🤝 Contributing

1. Fork репозиторій
2. Створіть feature branch
3. Зробіть changes
4. Створіть Pull Request

---

## 📄 Ліцензія

Цей проект ліцензовано під MIT License - дивіться [LICENSE](LICENSE) файл.

---

## 🙏 Подяки

- HuggingFace за transformers бібліотеку
- OpenAI за GPT-2 архітектуру
- Stanford Alpaca проект за датасет
- Спільноті за відгуки та покращення

---

## 📞 Підтримка

- **Issues**: [GitHub Issues](https://github.com/your-username/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/your-repo/discussions)
- **Email**: your-email@example.com

---

**🚀 Готово до навчання вашої GPT-2 моделі!**

**Windows (Command Prompt):**
```cmd
setup_venv.bat
```

**Windows (PowerShell):**
```powershell
.\setup_venv.ps1
```

**Або вручну:**
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Активація віртуального середовища

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

> **Примітка для PowerShell:** Якщо отримуєте помилку "execution of scripts is disabled", виконайте:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Опціональні залежності

Для роботи з GGUF моделями (Phi-3, тощо):
```bash
pip install llama-cpp-python
```

Або розкоментуйте рядок в `requirements.txt`:
```txt
llama-cpp-python  # Розкоментуйте якщо потрібно
```

---

## 🎯 Швидкий старт

### 1. Запуск навчання

**Основний спосіб:**
```bash
./start_training.sh
```

**Або напряму:**
```bash
cd /media/sony/641bf160-e2a6-47a2-b335-1da24af98536/ai/tiny_recursive_model-0.0.12
source venv/bin/activate

python scripts/train_model.py \
    --dataset datasets/train/openassistant_train.json \
    --dim 1024 \
    --seq-len 4096 \
    --batch-size 1 \
    --epochs 15 \
    --learning-rate 2e-4 \
    --checkpoint-dir checkpoints \
    --checkpoint-interval 100
```

### 2. Перевірка статусу навчання

```bash
python scripts/check_training_status.py
```

**Або через скрипт:**
```bash
./check_training.sh
```

### 3. Моніторинг

**Простий моніторинг:**
```bash
./monitor.sh
```

**Детальний моніторинг з виявленням проблем:**
```bash
./monitor_training.sh
```

Детальний моніторинг автоматично:
- Виявляє підвисання (якщо лог не оновлюється > 5 хв)
- Виявляє обмеження ресурсів (високе використання CPU/пам'яті)
- Логує всі виявлені проблеми в `logs/monitoring_*.log`

### 4. Зупинка навчання

```bash
./stop_training.sh
```

---

## 📊 Де що знаходиться

### 🚀 Запуск навчання

| Файл | Опис |
|------|------|
| `start_training.sh` | **Головний скрипт запуску** - запускає навчання з правильними параметрами |
| `scripts/train_model.py` | **Основний Python скрипт** - використовує `train_with_auto_config()` |
| `train/train_code_model.py` | Альтернативний скрипт навчання (для коду) |

### 🧪 Тестування

| Файл | Опис |
|------|------|
| `scripts/test_model.py` | Тестування навченої моделі |
| `scripts/test_model_capabilities.py` | Розширене тестування можливостей |

### 📈 Статус та моніторинг

| Файл | Опис |
|------|------|
| `scripts/check_training_status.py` | **Перевірка статусу** - показує чи працює навчання, прогрес, checkpoint'и |
| `check_training.sh` | Shell скрипт для перевірки статусу |
| `monitor.sh` | Простий моніторинг процесу навчання |
| `monitor_training.sh` | Детальний моніторинг з виявленням підвисань та обмежень ресурсів |

### 💾 Checkpoint'и та моделі

| Папка/Файл | Опис |
|------------|------|
| `checkpoints/` | Автоматично створюється, містить checkpoint'и навчання |
| `checkpoints/checkpoint_latest.pt` | Останній checkpoint для продовження |
| `models/trained/` | Навчені моделі зберігаються тут |

---

## 🔧 Основні команди

### Навчання

```bash
# Запуск навчання
./start_training.sh

# Або напряму
python scripts/train_model.py --dataset datasets/train/openassistant_train.json
```

### Продовження після перерви

```bash
python scripts/train_model.py \
    --dataset datasets/train/openassistant_train.json \
    --resume checkpoints/checkpoint_latest.pt \
    --checkpoint-dir checkpoints
```

**Або через скрипт:**
```bash
./start_training.sh
# Скрипт автоматично запропонує продовжити з checkpoint'у якщо він є
```

### Перевірка статусу

```bash
# Python скрипт (детальна інформація)
python scripts/check_training_status.py

# Shell скрипт (швидка перевірка)
./check_training.sh

# Перевірка процесів
ps aux | grep train_model
```

### Моніторинг

```bash
# Простий моніторинг (швидкий перегляд)
./monitor.sh

# Детальний моніторинг (виявлення проблем, логування)
./monitor_training.sh

# Або Python скрипт (детальна інформація)
python scripts/check_training_status.py
```

### Зупинка

```bash
./stop_training.sh
```

### Збереження моделі для донавчання

```bash
# Зберегти модель з автоматичним пошуком останньої моделі та чекпоінтів
python scripts/save_model.py --model-name my_model --dataset-path datasets/train/openassistant_train.json

# Зберегти конкретну модель
python scripts/save_model.py --model-name my_model --model-path models/trained/model.pt --checkpoint-dir checkpoints --dataset-path datasets/train/openassistant_train.json
```

Скрипт створить структуру в `saved_models/{model_name}/` з:
- Моделлю (`model.pt`)
- Всіма чекпоінтами (`checkpoints/`)
- Датасетом (`dataset/`)
- Конфігурацією (`model_config.json`, `training_config.json`)
- Інструкціями для донавчання (`README.md`)

---

## 📝 Параметри навчання

### Поточні параметри (в `start_training.sh`):

- **Датасет:** `datasets/train/openassistant_train.json`
- **dim:** 1024
- **seq_len:** 4096
- **batch_size:** 1
- **epochs:** 15
- **learning_rate:** 2e-4
- **checkpoint_interval:** 100 (зберігає кожні 100 батчів)

### Зміна параметрів

Відредагуйте `start_training.sh` або запускайте напряму:

```bash
python scripts/train_model.py \
    --dataset datasets/train/openassistant_train.json \
    --dim 512 \
    --seq-len 2048 \
    --batch-size 4 \
    --epochs 20 \
    --learning-rate 1e-4
```

---

## 🔬 Порівняння моделей (GGUF та TRM)

Скрипт `scripts/compare_gguf_models.py` дозволяє порівнювати:
- **GGUF моделі** (Phi-3, TinyLlama, DeepSeek тощо)
- **Навчені TRM моделі** (ваші кастомні моделі)
- **Змішані порівняння** (GGUF з TRM)

### Основні можливості:

1. **Вибір конкретних моделей** - ви можете вказати які саме моделі порівнювати
2. **Порівняння GGUF з TRM** - порівняйте свою навчену модель з GGUF моделями
3. **Автоматичне визначення типу** - скрипт автоматично розпізнає тип моделі

### Приклади використання:

**Порівняти всі знайдені GGUF моделі:**

```bash
python scripts/compare_gguf_models.py --all
```

**Порівняти конкретні GGUF моделі:**

```bash
python scripts/compare_gguf_models.py \
    --models models/gguf/phi-3.5-mini-instruct-q4_k_m.gguf \
              models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
```

**Порівняти TRM модель з GGUF моделями:**

```bash
# Порівняти вашу навчену TRM модель з GGUF моделями
python scripts/compare_gguf_models.py \
    --models models/trained/my_model.pt \
              models/gguf/phi-3.5-mini-instruct-q4_k_m.gguf \
    --trm-config models/trained/my_model_config.json
```

**Порівняти тільки TRM моделі:**

```bash
# Порівняти всі знайдені TRM моделі
python scripts/compare_gguf_models.py --all-trm

# Або конкретні TRM моделі
python scripts/compare_gguf_models.py \
    --trm-models models/trained/model1.pt \
                 models/trained/model2.pt
```

**Змішане порівняння (GGUF + TRM):**

```bash
# Скрипт автоматично визначить тип кожної моделі
python scripts/compare_gguf_models.py \
    --models models/gguf/phi-3.5-mini-instruct-q4_k_m.gguf \
              models/trained/my_model.pt \
              models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
```

### Метрики порівняння:

Скрипт порівнює:
- ⚡ **Швидкість генерації** (токенів/сек)
- 🎯 **Якість відповідей** (відповідність очікуваним ключовим словам)
- ✅ **Надійність** (успішність тестів)
- 🔄 **Кроки уточнення** (для TRM моделей)

Результати зберігаються в JSON звіті з детальними метриками для кожної моделі.

---

## 📚 Документація

Всі детальні документи проекту знаходяться в папці [`README/`](README/README.md):

- [Аудит проекту](README/AUDIT_REPORT.md)
- [Керівництво з checkpoint'ів](README/CHECKPOINT_GUIDE.md)
- [Підсумок рефакторингу](README/REFACTORING_SUMMARY.md)
- [Аудит скриптів](README/SCRIPTS_AUDIT_REPORT.md)
- [Перевірка параметрів](README/PARAMETERS_CHECK.md)

---

## ⚠️ Важливо

1. **Віртуальне середовище:** Завжди активуйте `venv` перед роботою:
   ```bash
   source venv/bin/activate
   ```

2. **Checkpoint'и:** Навчання автоматично зберігає checkpoint'и в `checkpoints/`. Можна продовжити після перерви.

3. **Моделі:** Навчені моделі зберігаються в `models/trained/`

4. **Датасети:** Датасети знаходяться в `datasets/train/` та `datasets/eval/`

---

## 📝 Логування та моніторинг

### Структура логів

Всі логи зберігаються в папці `logs/`:

- **`logs/training_*.log`** - основні логи навчання (з timestamp)
- **`logs/training_latest.log`** - символічне посилання на останній лог
- **`logs/training_detailed_*.log`** - детальні логи з інформацією про ресурси
- **`logs/resource_monitor_*.log`** - логи моніторингу ресурсів (CPU, пам'ять, GPU)
- **`logs/monitoring_*.log`** - логи з моніторингових скриптів
- **`logs/*.json`** - метрики та статистика в JSON форматі

### Перегляд логів

```bash
# Останній лог навчання
tail -f logs/training_latest.log

# Детальний лог з ресурсами
tail -f logs/training_detailed_*.log

# Лог моніторингу ресурсів
tail -f logs/resource_monitor_*.log

# Лог моніторингового скрипта
tail -f logs/monitoring_*.log
```

### Моніторинг ресурсів

Під час навчання автоматично відстежуються:
- **CPU використання** - виявлення високого навантаження (>95%)
- **Пам'ять (RAM)** - виявлення високого використання (>90%)
- **GPU пам'ять** - якщо GPU доступний
- **Тривалість батчів** - виявлення повільних батчів (>60 секунд)
- **Підвисання** - виявлення завислих процесів

Всі попередження автоматично логуються в `logs/resource_monitor_*.log`.

### Налаштування логування

```bash
# Змінити інтервал логування ресурсів (кожні N батчів)
python scripts/train_model.py --log-interval 20

# Вимкнути моніторинг ресурсів
python scripts/train_model.py --disable-resource-monitor

# Вимкнути логування рекурсії
python scripts/train_model.py --disable-recursion-logging
```

## 🆘 Допомога

Якщо щось не працює:

1. Перевірте чи активовано venv: `source venv/bin/activate`
2. Перевірте статус: `python scripts/check_training_status.py`
3. Перевірте процеси: `ps aux | grep train_model`
4. Перевірте логи: `tail -f logs/training_latest.log`
5. Перевірте моніторинг ресурсів: `tail -f logs/resource_monitor_*.log`

---

**Останнє оновлення:** 18 грудня 2025
# AI_project

## ✅ GitHub sanity check
Commit from mobile Debian terminal (Pixel 7)
