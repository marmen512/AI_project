# Встановлення та запуск на Linux (Debian/Ubuntu)

## Швидкий старт

### 1. Встановлення Python та середовища

```bash
# Зробити скрипт виконуваним
chmod +x setup_venv.sh

# Запустити встановлення
./setup_venv.sh
```

Скрипт автоматично:
- Перевірить наявність Python 3.9+
- Встановить Python та необхідні залежності (якщо потрібно)
- Створить віртуальне середовище `venv`
- Встановить всі залежності з `requirements.txt`

### 2. Активація середовища

```bash
source venv/bin/activate
```

### 3. Запуск навчання

```bash
# Зробити скрипт виконуваним
chmod +x start_training.sh

# Запустити навчання
./start_training.sh
```

Або вручну:

```bash
source venv/bin/activate
python -m runtime.bootstrap --mode new --config config/config.yaml
```

## Ручне встановлення (якщо скрипт не працює)

### Встановлення Python та залежностей

```bash
# Оновити список пакетів
sudo apt-get update

# Встановити Python 3.9+ та необхідні залежності
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git
```

### Створення віртуального середовища

```bash
# Створити venv
python3 -m venv venv

# Активувати venv
source venv/bin/activate

# Оновити pip
python -m pip install --upgrade pip setuptools wheel

# Встановити залежності
pip install -r requirements.txt
```

## Продовження навчання з checkpoint

Якщо є checkpoint, скрипт `start_training.sh` автоматично запропонує продовжити:

```bash
./start_training.sh
# Відповісти 'y' коли запитає про checkpoint
```

Або вручну:

```bash
source venv/bin/activate
python -m runtime.bootstrap --mode resume --config config/config.yaml
```

## Деактивація середовища

```bash
deactivate
```

## Перевірка встановлення

```bash
source venv/bin/activate
python --version
pip list
```

## Вирішення проблем

### Помилка: "Permission denied"
```bash
chmod +x setup_venv.sh
chmod +x start_training.sh
```

### Помилка: "Python 3.9+ required"
Встановіть Python 3.9 або новіший:
```bash
sudo apt-get install python3.9 python3.9-venv python3.9-dev
```

### Помилка: "pip install failed"
Спробуйте оновити pip:
```bash
python -m pip install --upgrade pip
```

### Помилка: "build-essential not found"
Встановіть build tools:
```bash
sudo apt-get install build-essential
```

