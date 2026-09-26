@echo off
echo =========================================================
echo      AMAZON ML CHALLENGE: GPU PREPROCESSING ENGINE
echo =========================================================
echo.
echo Hello! This script will automatically set up everything you need 
echo to run the heavy math on your RTX 5050. You don't need to know Python!
echo.
echo [Step 1] Checking if Python is installed...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.10+ and try again.
    pause
    exit /b
)

echo [Step 2] Installing necessary AI libraries (PyTorch, Pandas, Sentence-Transformers)...
pip install torch pandas sentence-transformers numpy --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install libraries. Check your internet connection.
    pause
    exit /b
)

echo.
echo [Step 3] Booting up the RTX 5050 and starting the Preprocessor...
echo Please do not close this window until it says "Complete!"
echo.
python role1_gpu_preprocessor.py

echo.
echo =========================================================
echo Processing Complete! You can now close this window.
echo Give the 'data/clean_data/' folder to Role 2!
echo =========================================================
pause
