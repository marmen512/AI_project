@echo off
@chcp 65001 >nul
setlocal

echo 🚀 ФАЗА 1 - Language Pretraining (Windows)
echo ===========================================

:: Auto-activate venv-win if found and not active
if "%VIRTUAL_ENV%"=="" (
    if exist "venv-win\Scripts\activate.bat" (
        echo 🔌 Auto-activating venv-win...
        call venv-win\Scripts\activate.bat
    ) else (
        echo ⚠️  Virtual Environment not active / found!
        echo    Please run: setup_windows.bat
        echo    Then: venv-win\Scripts\activate
        pause
        exit /b 1
    )
)

:: Validate dataset
if not exist "datasets\pretrain_text.txt" (
    echo ❌ Dataset not found: datasets\pretrain_text.txt
    echo    Run: python scripts/prepare_phase1_dataset.py
    pause
    exit /b 1
)

:: Config
echo.
echo 📋 Configuration:
echo    Config: config/phase1_pretraining.yaml
echo    Model: Small Transformer (Phase 1)
echo.

set /p START="🤔 Start Phase 1? (y/N): "
if /i not "%START%"=="y" goto :EOF

:: Folders
if not exist "checkpoints\phase1" mkdir "checkpoints\phase1"
if not exist "logs\phase1" mkdir "logs\phase1"

echo.
echo 🎯 Starting training...
echo    Monitoring: logs/phase1/phase1_pretraining.log
echo.

python scripts/train_phase1_pretraining.py --config config/phase1_pretraining.yaml
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️  Training failed or interrupted (Code: %ERRORLEVEL%)
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ Phase 1 Completed!
echo    Checkpoints saved in checkpoints/phase1
pause
