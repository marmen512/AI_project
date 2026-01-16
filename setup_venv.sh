#!/bin/bash
# Скрипт для встановлення Python та створення віртуального середовища на Debian/Ubuntu
# Використання: ./setup_venv.sh

set -e  # Зупинитися при помилці

echo "=========================================="
echo "[START] Встановлення Python та середовища"
echo "=========================================="
echo ""

# Перевірка чи запущено від root (не потрібно для встановлення в систему)
if [ "$EUID" -eq 0 ]; then 
   echo "[WARN] Не рекомендується запускати від root"
   echo "   Використовуйте звичайного користувача"
   read -p "   Продовжити? (y/N): " -n 1 -r
   echo
   if [[ ! $REPLY =~ ^[Yy]$ ]]; then
       exit 1
   fi
fi

# Перевірка наявності Python
if ! command -v python3 &> /dev/null; then
    echo "[INSTALL] Python3 не знайдено, встановлюю..."
    
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
    
    echo "[OK] Python3 встановлено"
else
    PYTHON_VERSION=$(python3 --version)
    echo "[OK] $PYTHON_VERSION знайдено"
fi

# Перевірка версії Python (потрібен 3.9+)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "[ERROR] Потрібен Python 3.9 або новіший"
    echo "   Поточна версія: $(python3 --version)"
    exit 1
fi

echo ""

# Перевірка чи venv вже існує
if [ -d "venv-linux" ]; then
    echo "[WARN] venv-linux вже існує"
    read -p "   Перезаписати? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[DELETE] Видалення старого venv-linux..."
        rm -rf venv-linux
    else
        echo "[INFO] Використовується існуючий venv-linux"
        echo ""
        echo "=========================================="
        echo "[OK] Готово!"
        echo "=========================================="
        echo ""
        echo "Для активації виконайте:"
        echo "  source venv-linux/bin/activate"
        echo ""
        exit 0
    fi
fi

# Створення venv
echo "[CREATE] Створення віртуального середовища venv-linux..."
python3 -m venv venv-linux

if [ $? -ne 0 ]; then
    echo "[ERROR] Помилка при створенні venv-linux"
    exit 1
fi

# Активувати venv
echo "[ACTIVATE] Активування venv-linux..."
source venv-linux/bin/activate

# Оновити pip
echo "[UPDATE] Оновлення pip..."
python -m pip install --upgrade pip setuptools wheel

# Встановити залежності
echo ""
echo "[INSTALL] Встановлення залежностей з requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[WARN] requirements.txt не знайдено!"
    exit 1
fi

echo ""
echo "=========================================="
echo "[OK] Віртуальне середовище venv-linux створено!"
echo "=========================================="
echo ""
echo "Для активації виконайте:"
echo "  source venv-linux/bin/activate"
echo ""

