@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local virtual environment not found. Run setup_gui.bat first.
  echo Python 3.12 is required. You can pass its full path to setup_gui.bat.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "launch_gui.py" %*
set "GUI_EXIT_CODE=%ERRORLEVEL%"
if not "%GUI_EXIT_CODE%"=="0" (
  echo GUI startup failed. See the error above; run setup_gui.bat to install dependencies.
  pause
)
exit /b %GUI_EXIT_CODE%
