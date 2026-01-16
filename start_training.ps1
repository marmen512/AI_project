# Скрипт для запуску навчання моделі на Windows (PowerShell)
# Використання: .\start_training.ps1

Set-Location $PSScriptRoot

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[START] ЗАПУСК НАВЧАННЯ МОДЕЛІ" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Перевірка наявності venv
if (-not (Test-Path "venv")) {
    Write-Host "[ERROR] Помилка: venv не знайдено" -ForegroundColor Red
    Write-Host "   Створіть venv: .\setup_venv.ps1" -ForegroundColor Yellow
    Read-Host "Натисніть Enter для виходу"
    exit 1
}

# Активувати venv
Write-Host "[ACTIVATE] Активування віртуального середовища..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Помилка активації venv" -ForegroundColor Red
    Read-Host "Натисніть Enter для виходу"
    exit 1
}

# Перевірка чи вже працює навчання
$running = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*runtime.bootstrap*" -or $_.MainWindowTitle -like "*runtime.bootstrap*"
}
if ($running) {
    Write-Host "[WARN] Навчання вже запущено!" -ForegroundColor Yellow
    Write-Host "   PID: $($running.Id)" -ForegroundColor Yellow
    $stop = Read-Host "   Зупинити і запустити нове? (y/N)"
    if ($stop -eq "y" -or $stop -eq "Y") {
        Write-Host "   [STOP] Зупиняю попереднє навчання..." -ForegroundColor Yellow
        Stop-Process -Id $running.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } else {
        Write-Host "   [CANCEL] Скасовано" -ForegroundColor Red
        Read-Host "Натисніть Enter для виходу"
        exit 0
    }
}

# Перевірка наявності checkpoint'у для продовження
$MODE = "new"
if ((Test-Path "checkpoints\checkpoint_latest.pt") -or (Get-ChildItem "checkpoints\ckpt_step_*.pt" -ErrorAction SilentlyContinue)) {
    Write-Host "[CHECKPOINT] Знайдено checkpoint для продовження" -ForegroundColor Cyan
    $resume = Read-Host "   Продовжити з checkpoint'у? (y/N)"
    if ($resume -eq "y" -or $resume -eq "Y") {
        $MODE = "resume"
        Write-Host "   [OK] Буде продовжено з checkpoint'у" -ForegroundColor Green
    }
}

# Створити папку logs якщо не існує
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Генерувати ім'я лог-файлу з timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = "logs\training_$timestamp.log"
$LATEST_LOG = "logs\training_latest.log"

# Запуск навчання
Write-Host ""
Write-Host "[START] Запуск навчання..." -ForegroundColor Green
Write-Host "   Скрипт: runtime.bootstrap" -ForegroundColor Cyan
Write-Host "   Логи зберігаються в: $LOG_FILE" -ForegroundColor Cyan
Write-Host "   Checkpoint'и зберігаються в: checkpoints/" -ForegroundColor Cyan
Write-Host ""

# Використовувати runtime.bootstrap
python -m runtime.bootstrap --mode $MODE --config config/config.yaml 2>&1 | Tee-Object -FilePath $LOG_FILE | Tee-Object -FilePath $LATEST_LOG

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "[OK] НАВЧАННЯ ЗАВЕРШЕНО" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] Помилка під час навчання" -ForegroundColor Red
    Write-Host "   Перевірте логи: $LOG_FILE" -ForegroundColor Yellow
}

Read-Host "Натисніть Enter для виходу"

