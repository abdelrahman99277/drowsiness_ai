@echo off
setlocal
cd /d "%~dp0"
title AI Drowsiness Detection

call setup_env.bat
if errorlevel 1 pause & exit /b 1

if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    (echo [general]&echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)

if not exist model.pkl (
    echo.
    echo [INFO] model.pkl not found.
    echo Collect data with collect_normal.bat and collect_drowsy.bat, then run train.bat.
    echo The application will not make a fake rule-based decision.
    echo.
)

venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false
pause
