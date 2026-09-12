@echo off
setlocal
cd /d "%~dp0"
set "MOON_DIAGNOSTICS_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%MOON_DIAGNOSTICS_PYTHON%" goto launch

rem An isolated Git worktree can reuse the main checkout's installed packages.
rem The working directory and imported application code remain this worktree.
for /f "delims=" %%G in ('git rev-parse --path-format=absolute --git-common-dir 2^>nul') do set "MOON_DIAGNOSTICS_GIT_DIR=%%G"
if not defined MOON_DIAGNOSTICS_GIT_DIR goto missing
for %%G in ("%MOON_DIAGNOSTICS_GIT_DIR%\..\.venv\Scripts\python.exe") do set "MOON_DIAGNOSTICS_PYTHON=%%~fG"
if not exist "%MOON_DIAGNOSTICS_PYTHON%" goto missing

:launch
"%MOON_DIAGNOSTICS_PYTHON%" "%~dp0launch_gui.py"
if errorlevel 1 pause
exit /b

:missing
echo Python environment not found in this worktree or the main checkout.
echo See ASSIGNMENT_INTEGRATION.md for launch instructions.
pause
exit /b 1
