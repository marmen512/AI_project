@echo off
REM Скрипт для створення віртуального середовища та встановлення залежностей
REM Використання: setup_venv.bat

chcp 65001 >nul
echo ==========================================
echo [START] Створення віртуального середовища
echo ==========================================
echo.

REM Перевірка наявності Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Помилка: Python не знайдено
    echo    Встановіть Python 3.9 або новіший з https://www.python.org/
    pause
    exit /b 1
)

REM Перевірка версії Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% знайдено
echo.

REM Перевірка чи venv вже існує
if exist "venv" (
    echo [WARN] venv вже існує
    set /p OVERWRITE="   Перезаписати? (y/N): "
    if /i "%OVERWRITE%"=="y" (
        echo [DELETE] Видалення старого venv...
        rmdir /s /q venv
    ) else (
        echo [INFO] Використовується існуючий venv
        echo.
        echo ==========================================
        echo [OK] Готово!
        echo ==========================================
        echo.
        echo Для активації виконайте:
        echo   venv\Scripts\activate.bat
        echo.
        pause
        exit /b 0
    )
)

REM Створення venv
echo [CREATE] Створення віртуального середовища...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Помилка при створенні venv
    pause
    exit /b 1
)

REM Активувати venv
echo [ACTIVATE] Активування venv...
call venv\Scripts\activate.bat

REM Оновити pip
echo [UPDATE] Оновлення pip...
python -m pip install --upgrade pip setuptools wheel

REM Встановити залежності
echo.
echo [INSTALL] Встановлення залежностей з requirements.txt...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo [WARN] requirements.txt не знайдено, встановлюю базові залежності...
    pip install accelerate adam-atan2-pytorch einops ema-pytorch torch x-transformers transformers datasets tqdm numpy psutil
)

echo.
echo ==========================================
echo [OK] Віртуальне середовище створено!
echo ==========================================
echo.
echo Для активації виконайте:
echo   venv\Scripts\activate.bat
echo.
echo Для деактивації виконайте:
echo   deactivate
echo.
pause

