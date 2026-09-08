@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  if "%~1"=="" (
    py -3.12 -m venv .venv
  ) else (
    "%~1" -m venv .venv
  )
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-gpu.txt -c requirements-performance-lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "from PySide6.QtWidgets import QApplication; import numpy, scipy, matplotlib, yaml; print('GUI dependencies are ready.')"
if errorlevel 1 goto failed
echo Setup complete. Run launch_gui.bat and select GPU surface in the calculation mode.
echo CUDA requires a compatible NVIDIA GPU and driver; CPU modes remain available.
exit /b 0

:failed
echo Setup failed. See the error above.
echo Install Python 3.12 or run: setup_gui.bat "C:\path\to\python.exe"
echo This script only changes the local .venv, not your system Python.
pause
exit /b 1
