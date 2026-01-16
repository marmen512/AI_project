# Google Colab Integration Guide

## Огляд

Цей проект можна експортувати та запускати в Google Colab з певними обмеженнями. Нижче наведено повну інструкцію.

## 🚀 Швидкий старт

### 1. Завантажте проект в Colab

```python
# Клонуйте репозиторій
!git clone https://github.com/your-username/your-repo.git
%cd your-repo

# Встановіть залежності
!pip install torch==2.9.1 transformers==4.57.3 datasets==4.4.2 PyYAML==6.0.3 tqdm==4.67.1 loguru==0.7.3
```

### 2. Використовуйте готовий notebook

Відкрийте `colab_setup.ipynb` в Google Colab та виконуйте клітинки послідовно.

## 📋 Обмеження Colab

| Обмеження | Вплив на проект | Рішення |
|------------|----------------|----------|
| **CPU-only** | Повільніше навчання | Менші batch sizes, gradient accumulation |
| **12 годин сесія** | Переривання навчання | Автозбереження, Google Drive backup |
| **Обмежений диск** | Неможливість великих датасетів | Мінімальні датасети для тестування |
| **Обмежена RAM** |OOM помилки | Зменшення batch_size, gradient_accumulation |

## 🛠️ Конфігурація для Colab

Створено спеціальну конфігурацію `config/colab_phase2.yaml`:

```yaml
training:
  batch_size: 2              # Менший batch
  gradient_accumulation_steps: 4  # Ефективний batch = 8
  epochs: 1                  # Швидке тестування
  learning_rate: 5e-5
  max_grad_norm: 1.0
  
cpu_optimization:
  num_threads: 2             # Обмеження для Colab
  num_workers: 0
  pin_memory: False
```

## 📦 Структура файлів для Colab

```
your-repo/
├── colab_setup.ipynb         # Головний notebook
├── config/
│   └── colab_phase2.yaml    # Оптимізована конфігурація
├── datasets/
│   └── minimal_test.json    # Мінімальний датасет
├── checkpoints/             # Автоматично створюється
└── logs/                   # Логи навчання
```

## 🔄 Процес запуску

### Крок 1: Підготовка
```python
# Встановлення
!pip install -r requirements.txt

# Підготовка датасетів
!mkdir -p datasets
# Завантаження мінімального датасету для тестування
```

### Крок 2: Створення базової моделі
```python
# Якщо у вас немає Phase 1 checkpoint
import torch
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config(
    vocab_size=50257, n_embd=320, n_layer=6, n_head=8, n_positions=256
)
model = GPT2LMHeadModel(config)
torch.save(model.state_dict(), 'checkpoints/phase1/best_model.pt')
```

### Крок 3: Запуск Phase 2
```python
!python scripts/train_phase2_instruction_tuning.py \
  --config config/colab_phase2.yaml \
  --phase1-model checkpoints/phase1/best_model.pt
```

### Крок 4: Тестування
```python
!python scripts/test_model.py \
  --model checkpoints/phase2/best_instruction_model.pt \
  --prompt "What is 2+2?"
```

## 💾 Збереження на Google Drive

Автоматичне збереження результатів:

```python
from google.colab import drive
import shutil
from datetime import datetime

drive.mount('/content/drive')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"/content/drive/MyDrive/GPT2_Training_{timestamp}"

# Копіювання результатів
shutil.copytree('checkpoints/', f"{results_dir}/checkpoints")
shutil.copytree('logs/', f"{results_dir}/logs")
```

## ⚡ Оптимізації для Colab

### 1. Зменшення розміру даних
```python
# Використовуйте підмножину датасету
import json

with open('datasets/alpaca.json') as f:
    data = json.load(f)

# Взяти перші 100 прикладів
data['data'] = data['data'][:100]

with open('datasets/alpaca_small.json', 'w') as f:
    json.dump(data, f)
```

### 2. Моніторинг ресурсів
```python
import psutil
import time

def monitor():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"RAM: {psutil.virtual_memory().percent}%")

# Моніторинг кожні 30 секунд
for i in range(10):
    monitor()
    time.sleep(30)
```

### 3. Автоматичне збереження
```python
# Додайте в конфігурацію
training:
  checkpoint_interval: 50    # Частіше checkpoints
  save_every_steps: 10     # Частіше автозбереження
```

## 🔧 Вирішення проблем

### Проблема: Out of Memory
```yaml
# Зменште в конфігурації
training:
  batch_size: 1
  gradient_accumulation_steps: 8
```

### Проблема: Час сесії вичерпано
```python
# Автоматичне збереження кожні 30 хвилин
import time
import shutil

start_time = time.time()
while time.time() - start_time < 3600:  # 1 година
    # Ваш код навчання
    time.sleep(1800)  # 30 хвилин
    shutil.copytree('checkpoints/', '/content/drive/MyDrive/backup')
```

### Проблема: Повільне навчання
```python
# Зменште розмір датасету
# Використовуйте gradient accumulation
# Зменште частоту логування
```

## 📊 Очікувані результати в Colab

| Метрика | Значення в Colab | Локальна машина |
|----------|------------------|-----------------|
| Batch size | 2 | 4-8 |
| Epoch time | ~30-60 хв | ~10-20 хв |
| Total training time | 1-2 години | 30-60 хв |
| Quality | Базова | Повна |

## 🎯 Рекомендації

1. **Для тестування**: Використовуйте мінімальні датасети та конфігурацію Colab
2. **Для повного навчання**: Рекомендується локальна машина або VPS
3. **Збереження**: Регулярно копіюйте checkpoints на Google Drive
4. **Моніторинг**: Слідкуйте за використанням RAM/CPU
5. **Оптимізація**: Зменшуйте розміри даних для швидких експериментів

## 📝 Додаткові ресурси

- [Google Colab FAQ](https://research.google.com/colab/faq.html)
- [PyTorch на Colab](https://colab.research.google.com/notebooks/intro_to_pytorch.ipynb)
- [HuggingFace на Colab](https://huggingface.co/docs/transformers/notebooks)

## 🚨 Важливі зауваження

- Цей проект оптимізований для CPU. Для GPU потрібні зміни в коді.
- Colab не підходить для production навчання великих моделей.
- Завжди зберігайте прогрес перед закриттям сесії.
- Використовуйте Google Drive для надійного зберігання результатів.
