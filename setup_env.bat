@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo Drowsiness AI - Environment Check
echo ========================================

REM 1) Require Python 3.12. Never silently fall back to Python 3.14.
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 is not installed.
    echo Trying Python Install Manager...
    py install 3.12 >nul 2>&1
    py -3.12 --version >nul 2>&1
)
if errorlevel 1 (
    echo Python 3.12 still not found. Trying winget...
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    py -3.12 --version >nul 2>&1
)
if errorlevel 1 (
    echo [ERROR] Python 3.12 could not be found.
    echo Run: py install 3.12
    exit /b 1
)

REM 2) Create venv if needed. Existing Python 3.14 is untouched.
if not exist "venv\Scripts\python.exe" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create venv.
        exit /b 1
    )
)

REM 3) Verify the venv really uses Python 3.12.
venv\Scripts\python.exe --version
if errorlevel 1 (
    echo [ERROR] venv Python is not working.
    exit /b 1
)
venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)"
if errorlevel 1 (
    echo [ERROR] Existing venv is not Python 3.12.
    echo Delete only the venv folder and run this again.
    exit /b 1
)

REM 4) Only install dependencies when an import check says something is missing.
venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, sklearn, pandas, joblib, streamlit" >nul 2>&1
if errorlevel 1 (
    echo Required libraries are missing. Installing them now...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Package installation failed.
        exit /b 1
    )
)

REM 5) Final import test.
venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, sklearn, pandas, joblib, streamlit; print('ALL OK - Python 3.12 + required libraries')"
if errorlevel 1 (
    echo [ERROR] Library import test failed.
    exit /b 1
)

exit /b 0
