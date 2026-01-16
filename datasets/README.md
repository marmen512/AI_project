# 📚 Структура датасетів

Ця папка містить датасети для навчання та тестування моделі.

## 📁 Структура папок

```
datasets/
├── train/          # Training датасети (для навчання моделі)
├── eval/           # Eval датасети (для тестування моделі)
└── raw/            # Сирі датасети (необроблені файли)
```

## 🚀 Швидкий старт

### 1. Завантажити датасет

```bash
# Завантажити OpenAssistant датасет
python scripts/download_openassistant.py
```

Це створить:
- `datasets/train/openassistant_train.json` - для навчання
- `datasets/eval/openassistant_eval.json` - для тестування

### 2. Навчити модель

```bash
# Навчання з автоматичною конфігурацією
python scripts/train_model.py --dataset datasets/train/openassistant_train.json

# Або з кастомними параметрами
python scripts/train_model.py \
    --dataset datasets/train/openassistant_train.json \
    --epochs 10 \
    --batch-size 4
```

### 3. Протестувати модель

```bash
# Тестування на eval датасеті
python scripts/test_model.py \
    --model models/trained/trm_openassistant_train.pt \
    --dataset datasets/eval/openassistant_eval.json
```

## 📊 Формат датасету

Датасети мають формат JSON:

```json
{
  "metadata": {
    "source": "OpenAssistant/oasst_top1_2023-08-25",
    "split": "train",
    "num_examples": 2000
  },
  "data": [
    {
      "context": "Some context text",
      "query": "User query",
      "completion": "Expected completion"
    }
  ]
}
```

Або простіший формат (список):

```json
[
  {
    "context": "...",
    "query": "...",
    "completion": "..."
  }
]
```

## 🔧 Додаткові інструменти

### Створити датасет з TinyLlama

```bash
python scripts/phi3_to_trm.py \
    --create \
    --phi3-model models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \
    --output datasets/train/tinyllama_dataset.json \
    --num-examples 800
```

### Перевірити наявність датасетів

```bash
ls -lh datasets/train/
ls -lh datasets/eval/
```

## 📝 Примітки

- **Training датасети** використовуються для навчання моделі
- **Eval датасети** використовуються для оцінки якості після навчання
- **Raw датасети** - сирі файли перед обробкою












