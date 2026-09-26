@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 -m venv .venv
) else (
  python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
  echo Failed to create the virtual environment.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Setup complete.
echo Run: run.bat
pause
