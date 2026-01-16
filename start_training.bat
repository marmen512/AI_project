@echo off
REM Скрипт для запуску навчання моделі на Windows
REM Використання: start_training.bat

cd /d "%~dp0"

chcp 65001 >nul
echo ==========================================
echo [START] ЗАПУСК НАВЧАННЯ МОДЕЛІ
echo ==========================================
echo.

REM Перевірка наявності venv
if not exist "venv" (
    echo [ERROR] Помилка: venv не знайдено
    echo    Створіть venv: setup_venv.bat
    pause
    exit /b 1
)

REM Активувати venv
echo [ACTIVATE] Активування віртуального середовища...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Помилка активації venv
    pause
    exit /b 1
)

REM Перевірка чи вже працює навчання (спрощена версія)
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if not errorlevel 1 (
    echo [WARN] Python процеси знайдено
    echo    Переконайтеся що навчання не запущено в іншому вікні
)

REM Перевірка наявності checkpoint'у для продовження
set MODE=new
if exist "checkpoints\checkpoint_latest.pt" (
    echo [CHECKPOINT] Знайдено checkpoint для продовження
    set /p RESUME="   Продовжити з checkpoint'у? (y/N): "
    if /i "%RESUME%"=="y" (
        set MODE=resume
        echo    [OK] Буде продовжено з checkpoint'у
    )
)

REM Створити папку logs якщо не існує
if not exist "logs" mkdir logs

REM Генерувати ім'я лог-файлу з timestamp
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
set mytime=%mytime: =0%
set LOG_FILE=logs\training_%mydate%_%mytime%.log
set LATEST_LOG=logs\training_latest.log

REM Запуск навчання
echo.
echo [START] Запуск навчання...
echo    Скрипт: runtime.bootstrap
echo    Логи зберігаються в: %LOG_FILE%
echo    Checkpoint'и зберігаються в: checkpoints/
echo.

REM Використовувати runtime.bootstrap
python -m runtime.bootstrap --mode %MODE% --config config/config.yaml > "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Помилка під час навчання
    echo    Перевірте логи: %LOG_FILE%
) else (
    echo.
    echo ==========================================
    echo [OK] НАВЧАННЯ ЗАВЕРШЕНО
    echo ==========================================
)

pause

