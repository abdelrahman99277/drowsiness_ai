@echo off
setlocal
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 pause & exit /b 1
venv\Scripts\python.exe train.py
pause
