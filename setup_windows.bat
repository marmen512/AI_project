@echo off
@chcp 65001 >nul
echo 🪟 Windows Environment Setup
echo ==========================

:: 1. Create venv-win if not exists
if not exist "venv-win" (
    echo 📦 Creating python venv: venv-win...
    python -m venv venv-win
) else (
    echo ✅ venv-win already exists
)

:: 2. Activate
call venv-win\Scripts\activate

:: 3. Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install dependencies
if exist "requirements.txt" (
    echo 📥 Installing requirements...
    python -m pip install -r requirements.txt
) else (
    echo ⚠️  requirements.txt not found!
    pause
    exit /b 1
)

echo.
echo ✅ Windows setup complete!
echo    Run: start_phase1_pretraining.bat
pause
