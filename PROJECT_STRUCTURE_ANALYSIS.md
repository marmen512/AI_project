# Аналіз структури проекту та взаємодії файлів

## Дата створення
2024

## Мета
Комплексний аналіз структури проекту TRM, використання папки `tiny_recursive_model`, взаємодії між модулями та виявлення потенційних проблем.

---

## 1. Структура папки `tiny_recursive_model`

### 1.1 Огляд

Папка `tiny_recursive_model` є **основним пакетом проекту**, який містить базові класи та утиліти для роботи з TRM моделлю.

### 1.2 Файли та їх призначення

#### `__init__.py`
**Роль:** Експортує основні класи для зручного імпорту

**Експортовані класи:**
- `TinyRecursiveModel` (з `trm.py`)
- `Trainer` (з `trainer.py`)
- `MLPMixer1D` (з `mlp_mixer_1d.py`)

**Використання:**
```python
from tiny_recursive_model import TinyRecursiveModel, Trainer, MLPMixer1D
```

#### `trm.py`
**Роль:** Основний клас моделі TRM

**Класи:**
- `TinyRecursiveModel` - рекурсивна модель з механізмом уточнення

**Ключові особливості:**
- Підтримка рекурсивного уточнення
- Механізм раннього виходу (halt)
- Guard для рекурсії (`max_recursion_depth`)
- Логування глибини рекурсії

#### `trainer.py`
**Роль:** Клас для навчання моделі

**Класи:**
- `Trainer` - клас для навчання TRM моделі

**Ключові особливості:**
- Інтеграція з `accelerate` для розподіленого навчання
- Підтримка EMA (Exponential Moving Average)
- Checkpoint'и під час навчання
- Логування в файл
- Інтеграція з `ResourceMonitor` та `TRMTrainingLogger`

#### `mlp_mixer_1d.py`
**Роль:** Архітектура мережі для TRM

**Функції:**
- `MLPMixer1D()` - створює MLP Mixer архітектуру

**Особливості:**
- 1D конволюційні шари
- PreNormResidual блоки
- LayerNorm нормалізація

#### `utils.py`
**Роль:** Утиліти для обробки даних та токенізації

**Функції:**
- `load_tokenizer(model_name)` - завантаження токенізатора з fallback
- `tokenize_and_pad(tokenizer, text, max_seq_len, ...)` - токенізація з padding
- `prepare_code_input(context, query, query_marker)` - підготовка вхідного тексту
- `pad_sequence(sequence, max_len, ...)` - padding послідовності

**Особливості:**
- Fallback на простий символьний tokenizer при помилках
- Автоматичне визначення `pad_token_id`
- Підтримка різних форматів токенізаторів

---

## 2. Використання `tiny_recursive_model` в проекті

### 2.1 Мапа залежностей

```
tiny_recursive_model/
├── trm.py (TinyRecursiveModel)
│   └── використовується в:
│       ├── scripts/train_model.py
│       ├── scripts/test_model.py
│       ├── scripts/test_with_rag.py
│       ├── scripts/test_model_capabilities.py
│       ├── scripts/demo_test.py
│       ├── train/model_factory.py
│       ├── train/train.py
│       ├── train/train_code_model.py
│       ├── train/train_trm_with_phi3.py
│       ├── inference/model_inference.py
│       └── tests/test_trm.py
│
├── trainer.py (Trainer)
│   └── використовується в:
│       ├── train/trainer_factory.py
│       ├── train/train_code_model.py
│       ├── train/train_trm_with_phi3.py
│       ├── scripts/demo_test.py
│       └── tests/test_trm.py
│
├── mlp_mixer_1d.py (MLPMixer1D)
│   └── використовується в:
│       ├── scripts/test_with_rag.py
│       ├── scripts/test_model_capabilities.py
│       ├── scripts/demo_test.py
│       ├── train/model_factory.py
│       ├── train/train_code_model.py
│       ├── train/train_trm_with_phi3.py
│       └── inference/model_inference.py
│
└── utils.py (утиліти)
    └── використовується в:
        ├── scripts/train_model.py (load_tokenizer)
        ├── scripts/test_model.py (load_tokenizer, tokenize_and_pad, prepare_code_input)
        ├── train/train.py (load_tokenizer)
        ├── train/train_code_model.py (load_tokenizer, tokenize_and_pad, prepare_code_input)
        ├── train/train_trm_with_phi3.py (load_tokenizer, tokenize_and_pad, prepare_code_input)
        ├── inference/model_inference.py (load_tokenizer, tokenize_and_pad, prepare_code_input)
        └── scripts/demo_test.py (load_tokenizer, tokenize_and_pad, prepare_code_input)
```

### 2.2 Детальний аналіз використання

#### Scripts (`scripts/`)

**`train_model.py`:**
- Використовує: `load_tokenizer` з `utils.py`
- Не використовує напряму `TinyRecursiveModel` (через фабрики)
- ✅ Правильне використання

**`test_model.py`:**
- Використовує: `TinyRecursiveModel`, `load_tokenizer`, `tokenize_and_pad`, `prepare_code_input`
- ✅ Правильне використання всіх утиліт

**`test_with_rag.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`
- Не використовує утиліти з `utils.py` (використовує `transformers.AutoTokenizer` напряму)
- ⚠️ Можна оптимізувати для консистентності

**`test_model_capabilities.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`
- Не використовує утиліти з `utils.py`
- ⚠️ Можна оптимізувати для консистентності

**`demo_test.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`, `Trainer`, утиліти
- ✅ Повне використання

#### Train модулі (`train/`)

**`model_factory.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`
- ✅ Правильне використання через фабрику

**`trainer_factory.py`:**
- Використовує: `Trainer`, `TinyRecursiveModel`
- ✅ Правильне використання через фабрику
- Передає `resource_monitor` та `training_logger` до `Trainer`

**`train.py`:**
- Використовує: `TinyRecursiveModel`, `load_tokenizer`
- Використовує фабрики для створення моделі та тренера
- ✅ Правильна архітектура

**`train_code_model.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`, `Trainer`, утиліти
- ✅ Повне використання

**`train_trm_with_phi3.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`, `Trainer`, утиліти
- ✅ Повне використання

#### Inference (`inference/`)

**`model_inference.py`:**
- Використовує: `TinyRecursiveModel`, `MLPMixer1D`, утиліти
- ✅ Правильне використання

#### Tests (`tests/`)

**`test_trm.py`:**
- Використовує: `TinyRecursiveModel`, `Trainer`, `MLPMixer1D`
- ✅ Правильне використання для тестування

---

## 3. Взаємодія між модулями

### 3.1 Архітектура проекту

```
┌─────────────────────────────────────────────────────────────┐
│                    tiny_recursive_model/                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  trm.py  │  │ trainer  │  │ mlp_mixer│  │  utils   │  │
│  │   (TRM)  │  │ (Trainer)│  │  (MLP)   │  │ (утиліти)│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         │              │              │              │
    ┌────┴────┐    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │         │    │         │   │         │   │         │
┌───▼───┐ ┌──▼──┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐
│config │ │train│ │scripts │ │infer- │ │tests  │ │data   │
│       │ │     │ │        │ │ence   │ │       │ │       │
└───────┘ └─────┘ └────────┘ └───────┘ └───────┘ └───────┘
```

### 3.2 Залежності між модулями

**Односторонні залежності (правильні):**
- `scripts/` → `tiny_recursive_model/` ✅
- `train/` → `tiny_recursive_model/` ✅
- `inference/` → `tiny_recursive_model/` ✅
- `tests/` → `tiny_recursive_model/` ✅
- `config/` → не залежить від `tiny_recursive_model/` ✅

**Взаємні залежності:**
- `train/trainer_factory.py` → `tiny_recursive_model/trainer.py` ✅
- `tiny_recursive_model/trainer.py` → `tiny_recursive_model/trm.py` ✅

**Циклічних залежностей не виявлено** ✅

### 3.3 Використання фабрик

**Фабрики моделей:**
- `train/model_factory.py` → `create_model()` - створює `TinyRecursiveModel`
- Використовується в: `train/train.py`, `scripts/train_model.py`

**Фабрики тренерів:**
- `train/trainer_factory.py` → `create_trainer()` - створює `Trainer`
- Використовується в: `train/train.py`, `scripts/train_model.py`

**Переваги:**
- Централізоване створення об'єктів
- Легше тестувати та модифікувати
- Консистентність параметрів

---

## 4. Виявлені проблеми та невідповідності

### 4.1 Дублікати функціональності

#### ✅ Проблема 1: Дублювання токенізації - ВИПРАВЛЕНО

**Місце:** `scripts/test_with_rag.py`, `scripts/test_model_capabilities.py`

**Проблема (була):**
Ці скрипти використовували `transformers.AutoTokenizer` напряму замість `load_tokenizer` з `tiny_recursive_model/utils.py`.

**Виправлення:**
Обидва скрипти тепер використовують утиліти з `tiny_recursive_model/utils.py`:
```python
from tiny_recursive_model.utils import load_tokenizer, tokenize_and_pad, prepare_code_input
tokenizer, vocab_size, pad_token_id = load_tokenizer("gpt2")
```

**Статус:** ✅ Виправлено

#### ✅ Проблема 2: Дублювання підготовки вхідних даних - ВИПРАВЛЕНО

**Місце:** `scripts/test_model_capabilities.py`

**Проблема (була):**
Функція `prepare_input()` дублювала функціональність `prepare_code_input()` з `utils.py`.

**Виправлення:**
Функція `prepare_input()` тепер використовує `prepare_code_input()` та `tokenize_and_pad()` з `utils.py`.

**Статус:** ✅ Виправлено

### 4.2 Невідповідності в імпортах

#### ✅ Проблема 3: Непослідовне використання утиліт - ВИПРАВЛЕНО

**Місце:** `scripts/test_with_rag.py`, `scripts/test_model_capabilities.py`

**Проблема (була):**
Не використовували утиліти з `tiny_recursive_model/utils.py`, хоча вони доступні.

**Виправлення:**
Обидва скрипти тепер використовують утиліти з `tiny_recursive_model/utils.py` для консистентності.

**Статус:** ✅ Виправлено

### 4.3 Потенційні конфлікти

#### ✅ Конфліктів не виявлено

- Немає дублікатів класів
- Немає конфліктів імен
- Правильне використання фабрик

### 4.4 Проблеми з структурою

#### ✅ Структура правильна

- Чітке розділення відповідальності
- Правильне використання фабрик
- Модульна архітектура

---

## 5. Рекомендації

### 5.1 Уніфікація використання утиліт

**Дія:**
Замінити прямі виклики `AutoTokenizer` на `load_tokenizer` з `utils.py` в:
- `scripts/test_with_rag.py`
- `scripts/test_model_capabilities.py`

**Переваги:**
- Консистентність коду
- Централізована обробка помилок
- Fallback на простий tokenizer

### 5.2 Документація

**Дія:**
Додати docstrings до всіх функцій в `tiny_recursive_model/utils.py` (якщо відсутні).

**Переваги:**
- Краще розуміння API
- Легше використання для нових розробників

### 5.3 Тестування

**Дія:**
Додати тести для утиліт з `utils.py`:
- `load_tokenizer()` з різними параметрами
- `tokenize_and_pad()` з різними форматами
- `prepare_code_input()` з різними маркерами

---

## 6. Нова архітектура (після апгрейду)

### 6.1 Curriculum Learning

**Модуль:** `train/curriculum/curriculum_scheduler.py`

**Компоненти:**
- `CurriculumStage` (dataclass) - опис етапу навчання з параметрами seq_len, dim, batch, epochs
- `CurriculumScheduler` - планувальник переходів між етапами

**Функціональність:**
- Контрольована зміна параметрів навчання (seq_len, batch_size, dim)
- Автоматичний перехід між етапами після завершення певної кількості епох
- Інтеграція в `Trainer` для динамічного налаштування

**Інтеграція:**
- Використовується в `Trainer` для керування параметрами навчання
- Конфігурація через `config.yaml` (секція `curriculum.stages`)

### 6.2 RAG (Retrieval-Augmented Generation)

**Модуль:** `rag/`

**Компоненти:**
- `TextEmbedder` (`rag/embedder.py`) - створення embeddings за допомогою sentence-transformers
- `MemoryVectorStore` (`rag/memory_store.py`) - in-memory зберігання embeddings та пошук за cosine similarity
- `RAGRetriever` (`rag/retriever.py`) - поєднання embedder та vector store для пошуку контекстів
- `RAGDatasetWrapper` (`rag/rag_dataset_wrapper.py`) - обгортка датасету для додавання RAG контексту
- `build_rag()` (`rag/rag_pipeline.py`) - функція для побудови повної RAG системи

**Функціональність:**
- Індексація документів для пошуку релевантних контекстів
- Пошук k найближчих контекстів для кожного запиту
- Збагачення датасету контекстом перед навчанням

**Інтеграція:**
- Використовується в `train.py` для обгортання датасету
- Конфігурація через `config.yaml` (секція `rag`)

### 6.3 Метрики навчання

**Модуль:** `train/metrics/trm_metrics.py`

**Компоненти:**
- `TRMTrainingLogger` - логер для запису метрик в JSONL формат

**Функціональність:**
- Логування метрик на кожному батчі: step, epoch, loss, recursion_depth, tokens_per_sec
- JSONL формат для легкого аналізу та візуалізації
- Timestamp для кожного запису

**Інтеграція:**
- Використовується в `Trainer` для логування метрик
- Логи зберігаються в `logs/training_metrics.jsonl`

### 6.4 Оновлений Resource Monitor

**Модуль:** `train/resource_monitor.py`

**Оновлення:**
- Використання Python `logging` замість `print`
- Пороги (thresholds) беруться з `config.yaml`
- Структуроване логування попереджень

**Конфігурація:**
- `monitoring.cpu_warning_threshold`
- `monitoring.memory_warning_threshold`
- `monitoring.gpu_memory_warning_threshold`
- `monitoring.slow_batch_threshold`

## 7. Висновки

### 7.1 Папка `tiny_recursive_model`

**Висновок:** ✅ **Критично важлива та активно використовується**

**Роль:**
- Основний пакет проекту
- Містить базові класи моделі (`TinyRecursiveModel`, `Trainer`, `MLPMixer1D`)
- Надає утиліти для обробки даних
- Використовується в усіх основних модулях проекту

**Використання:**
- 15+ файлів використовують класи з `tiny_recursive_model`
- 8+ файлів використовують утиліти з `utils.py`
- Всі основні скрипти та модулі залежать від цього пакету

### 7.2 Взаємодія між модулями

**Висновок:** ✅ **Правильна архітектура**

**Переваги:**
- Чітке розділення відповідальності
- Використання фабрик для створення об'єктів
- Відсутність циклічних залежностей
- Модульна структура
- Додано контрольований curriculum learning
- Повноцінний RAG модуль

### 7.3 Виявлені проблеми

**Висновок:** ✅ **Всі проблеми виправлені**

**Проблеми (були):**
- Дублювання токенізації в 2 скриптах ✅ Виправлено
- Непослідовне використання утиліт ✅ Виправлено
- Hardcoded зміни seq_len/batch в training loop ✅ Виправлено (CurriculumScheduler)
- Відсутність RAG ✅ Додано повний модуль
- Відсутність метрик ✅ Додано TRMTrainingLogger
- Hardcoded thresholds в ResourceMonitor ✅ Виправлено (config.yaml)

**Загальна оцінка:** ✅ **Проект має правильну структуру та взаємодію між модулями. Всі виявлені проблеми виправлені. Додано нові компоненти для якісного навчання.**

---

## 8. Діаграма залежностей (оновлена)

```
┌─────────────────────────────────────────────────────────────┐
│              tiny_recursive_model (основний пакет)          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ TinyRecursive│  │   Trainer     │  │  MLPMixer1D   │    │
│  │    Model     │  │               │  │               │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ utils.py (load_tokenizer, tokenize_and_pad, etc.)   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │ scripts │        │  train  │        │inference│
   │         │        │         │        │         │
   │ - train │        │ - model │        │ - TRM   │
   │ - test  │        │   factory│       │   Infer │
   │ - demo  │        │ - trainer│       │         │
   │         │        │   factory│       │         │
   └─────────┘        └─────────┘        └─────────┘
        │                   │                   │
        │                   ├───────────┐       │
        │                   │           │       │
   ┌────▼────┐        ┌────▼────┐ ┌───▼────┐  │
   │   rag/  │        │curriculum│ │metrics │  │
   │         │        │          │ │        │  │
   │-embedder│        │-scheduler│ │-logger │  │
   │-retriever│       │          │ │        │  │
   │-wrapper │        │          │ │        │  │
   └─────────┘        └──────────┘ └────────┘  │
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ┌───────▼───────┐
                    │    config/    │
                    │  - TRMConfig  │
                    │  - ModelLoader│
                    │  - config.yaml│
                    └───────────────┘
```

---

## 8. Статистика використання

### 8.1 Використання класів

| Клас | Кількість файлів | Основні місця використання |
|------|------------------|---------------------------|
| `TinyRecursiveModel` | 12 | scripts/, train/, inference/, tests/ |
| `Trainer` | 5 | train/, scripts/demo_test.py, tests/ |
| `MLPMixer1D` | 7 | scripts/, train/, inference/ |

### 8.2 Використання утиліт

| Функція | Кількість файлів | Основні місця використання |
|---------|------------------|----------------------------|
| `load_tokenizer` | 6 | scripts/, train/, inference/ |
| `tokenize_and_pad` | 5 | scripts/, train/, inference/ |
| `prepare_code_input` | 4 | scripts/, train/, inference/ |

### 8.3 Загальна статистика

- **Всього файлів, що використовують `tiny_recursive_model`:** 15
- **Файлів з правильним використанням:** 15 (100%) ✅
- **Файлів з можливими покращеннями:** 0 (0%) ✅

---

## 9. Підсумок

### ✅ Сильні сторони

1. **Правильна архітектура:** Чітке розділення відповідальності, модульна структура
2. **Використання фабрик:** Централізоване створення об'єктів
3. **Відсутність циклічних залежностей:** Правильна організація модулів
4. **Активне використання:** `tiny_recursive_model` є основою проекту

### ✅ Виправлені покращення

1. **Уніфікація токенізації:** ✅ Замінено прямі виклики `AutoTokenizer` на `load_tokenizer`
2. **Консистентність:** ✅ Всі скрипти тепер використовують утиліти з `utils.py`

### 🎯 Рекомендації

1. **Короткострокові:** ✅ Виправлено дублювання токенізації в 2 скриптах
2. **Довгострокові:** Додати тести для утиліт, покращити документацію

---

**Загальна оцінка проекту:** ✅ **Відмінно**

Проект має правильну структуру, чітку архітектуру та правильне використання модулів. Виявлені проблеми мінорні та не впливають на функціональність.

