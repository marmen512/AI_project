# Скрипт для створення віртуального середовища та встановлення залежностей
# Використання: .\setup_venv.ps1

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[START] Створення віртуального середовища" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Перевірка наявності Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion знайдено" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Помилка: Python не знайдено" -ForegroundColor Red
    Write-Host "   Встановіть Python 3.9 або новіший з https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Перевірка версії Python
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Host "[ERROR] Помилка: Потрібен Python 3.9 або новіший" -ForegroundColor Red
        Write-Host "   Поточна версія: $pythonVersion" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""

# Перевірка чи venv вже існує
if (Test-Path "venv") {
    Write-Host "[WARN] venv вже існує" -ForegroundColor Yellow
    $overwrite = Read-Host "   Перезаписати? (y/N)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        Write-Host "[DELETE] Видалення старого venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "[INFO] Використовується існуючий venv" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "[OK] Готово!" -ForegroundColor Green
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Для активації виконайте:" -ForegroundColor Yellow
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host ""
        exit 0
    }
}

# Створення venv
Write-Host "[CREATE] Створення віртуального середовища..." -ForegroundColor Cyan
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Помилка при створенні venv" -ForegroundColor Red
    exit 1
}

# Активувати venv
Write-Host "[ACTIVATE] Активування venv..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Оновити pip
Write-Host "[UPDATE] Оновлення pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Встановити залежності
Write-Host ""
Write-Host "[INSTALL] Встановлення залежностей з requirements.txt..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "[WARN] requirements.txt не знайдено, встановлюю базові залежності..." -ForegroundColor Yellow
    pip install accelerate adam-atan2-pytorch einops ema-pytorch torch x-transformers transformers datasets tqdm numpy psutil
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[OK] Віртуальне середовище створено!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Для активації виконайте:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Для деактивації виконайте:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "[TIP] Порада: Якщо отримуєте помилку 'execution of scripts is disabled', виконайте:" -ForegroundColor Cyan
Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
Write-Host ""

