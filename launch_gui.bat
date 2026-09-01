@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local virtual environment not found. Run: py -m venv .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "launch_gui.py"
endlocal
